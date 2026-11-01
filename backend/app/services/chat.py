"""
MEMOS — Chat Service
Orchestrates the full request lifecycle:
Input → Retrieval → Response → Memory Extraction → Evolution
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import get_settings
from app.database.chroma import VectorRepository
from app.database.mongo import ConversationRepository, MemoryRepository, UserRepository
from app.database.neo4j_db import GraphRepository
from app.memory.evolution import MemoryEvolutionService
from app.memory.extractor import extract_memories_from_turn
from app.models.domain import Conversation, Memory, Message, MessageRole
from app.retrieval.graph import GraphRetrievalService
from app.retrieval.hybrid import HybridRetrievalService, HybridCandidate
from app.retrieval.semantic import SemanticRetrievalService
from app.schemas.api import (
    ChatRequest,
    ChatResponse,
    MemorySummary,
    RetrievalDiagnostics,
)
from app.services.response import ResponseGenerationService

logger = logging.getLogger(__name__)
settings = get_settings()


class ChatService:
    """
    Full end-to-end chat orchestration service.
    """

    def __init__(
        self,
        memory_repo: MemoryRepository,
        conv_repo: ConversationRepository,
        user_repo: UserRepository,
        vector_repo: VectorRepository,
        graph_repo: GraphRepository,
    ):
        self.memory_repo = memory_repo
        self.conv_repo = conv_repo
        self.user_repo = user_repo

        # Build service layer
        semantic_service = SemanticRetrievalService(vector_repo, memory_repo)
        graph_service = GraphRetrievalService(graph_repo, memory_repo)
        self.hybrid_service = HybridRetrievalService(
            semantic_service, graph_service, memory_repo
        )
        self.evolution_service = MemoryEvolutionService(
            memory_repo, vector_repo, graph_repo
        )
        self.response_service = ResponseGenerationService()

    async def handle_chat(self, request: ChatRequest) -> ChatResponse:
        """
        Process a chat request end-to-end.
        """
        start_time = time.monotonic()
        user_id = request.user_id

        # ── 1. Ensure user exists ──────────────────────────────
        await self.user_repo.get_or_create(user_id, settings.demo_user_name)

        # ── 2. Get or create conversation ─────────────────────
        conversation = await self._get_or_create_conversation(
            conversation_id=request.conversation_id,
            user_id=user_id,
        )

        # ── 3. Load conversation history ──────────────────────
        history = await self.conv_repo.get_messages(
            conversation.conversation_id, limit=20
        )

        # ── 4. Retrieve relevant memories ─────────────────────
        retrieval_start = time.monotonic()
        hybrid_candidates = await self.hybrid_service.retrieve(
            query=request.message,
            user_id=user_id,
            mode=request.retrieval_mode,
            n_results=settings.final_context_limit,
        )
        retrieval_ms = int((time.monotonic() - retrieval_start) * 1000)

        memories = [c.memory for c in hybrid_candidates if c.memory]

        # ── 5. Update access timestamps ───────────────────────
        for memory in memories:
            memory.touch()
        if memories:
            # Batch update access timestamps (fire and forget)
            asyncio.create_task(self._batch_touch(memories))

        # ── 6. Generate response ───────────────────────────────
        try:
            response_text = await self.response_service.generate(
                user_message=request.message,
                conversation_history=history,
                memories=memories,
            )
        except Exception as e:
            logger.warning("LLM response generation failed (%s), generating memory-aware fallback", e)
            if memories:
                m_bullets = "; ".join(f"'{m.content}'" for m in memories[:3])
                response_text = (
                    f"I recall from memory that {m_bullets}. "
                    f"I've registered your latest input and updated your semantic profile."
                )
            else:
                response_text = (
                    f"Got it! I've noted down: \"{request.message}\". "
                    f"I've saved this into your persistent memory so I can recall it in future conversations."
                )

        total_ms = int((time.monotonic() - start_time) * 1000)

        # ── 7. Persist messages ───────────────────────────────
        used_memory_ids = [c.memory_id for c in hybrid_candidates if c.memory]

        user_msg = Message(
            conversation_id=conversation.conversation_id,
            user_id=user_id,
            role=MessageRole.USER,
            content=request.message,
            memory_ids_used=[],
            processing_ms=total_ms,
        )
        assistant_msg = Message(
            conversation_id=conversation.conversation_id,
            user_id=user_id,
            role=MessageRole.ASSISTANT,
            content=response_text,
            memory_ids_used=used_memory_ids,
            retrieval_diagnostics=self.hybrid_service.get_diagnostics(hybrid_candidates),
            processing_ms=total_ms,
        )
        await self.conv_repo.save_message(user_msg)
        await self.conv_repo.save_message(assistant_msg)

        # ── 8. Background memory extraction ───────────────────
        history_dicts = [{"role": m.role.value, "content": m.content} for m in history]
        asyncio.create_task(
            self._extract_and_store_memories(
                user_message=request.message,
                assistant_response=response_text,
                history=history_dicts,
                user_id=user_id,
                conversation_id=conversation.conversation_id,
                message_ids=[user_msg.message_id, assistant_msg.message_id],
            )
        )

        # ── 9. Build response ──────────────────────────────────
        memory_summaries = [
            MemorySummary(
                memory_id=c.memory_id,
                content=c.memory.content if c.memory else "",
                category=c.memory.category.value if c.memory else "unknown",
                importance=c.memory.importance if c.memory else 0.0,
                confidence=c.memory.confidence if c.memory else 0.0,
                rrf_score=c.rrf_score,
            )
            for c in hybrid_candidates
            if c.memory
        ]

        return ChatResponse(
            conversation_id=conversation.conversation_id,
            message_id=assistant_msg.message_id,
            response=response_text,
            memories_used=memory_summaries,
            diagnostics=RetrievalDiagnostics(
                semantic_candidates=sum(1 for c in hybrid_candidates if "semantic" in c.sources),
                graph_candidates=sum(1 for c in hybrid_candidates if "graph" in c.sources),
                final_memories=len(memories),
                retrieval_mode=request.retrieval_mode,
                timing_ms=retrieval_ms,
            ),
            processing_ms=total_ms,
        )

    async def _get_or_create_conversation(
        self, conversation_id: Optional[str], user_id: str
    ) -> Conversation:
        """Get existing conversation or create a new one."""
        if conversation_id:
            existing = await self.conv_repo.get(conversation_id, user_id)
            if existing:
                return existing
        # Create new conversation
        convo = Conversation(user_id=user_id)
        await self.conv_repo.create(convo)
        return convo

    async def _batch_touch(self, memories: List[Memory]) -> None:
        """Update last_accessed_at for retrieved memories."""
        for m in memories:
            try:
                await self.memory_repo.update(m)
            except Exception:
                pass

    async def _extract_and_store_memories(
        self,
        user_message: str,
        assistant_response: str,
        history: List[Dict[str, str]],
        user_id: str,
        conversation_id: str,
        message_ids: List[str],
    ) -> None:
        """
        Background task: extract and store memories from completed interaction.
        Failure here must not affect the already-sent chat response.
        """
        try:
            # Get existing memory summaries for deduplication context
            active_memories = await self.memory_repo.list_active(user_id, limit=20)
            existing_summaries = [m.content[:80] for m in active_memories[:10]]

            candidates = await extract_memories_from_turn(
                user_message=user_message,
                assistant_response=assistant_response,
                conversation_history=history,
                existing_memory_summaries=existing_summaries,
                source_message_ids=message_ids,
            )

            if not candidates:
                return

            logger.info(
                "Processing %d memory candidates for user %s", len(candidates), user_id
            )

            for candidate in candidates:
                try:
                    await self.evolution_service.process_candidate(
                        candidate=candidate,
                        user_id=user_id,
                        conversation_id=conversation_id,
                    )
                except Exception as e:
                    logger.error("Failed to process memory candidate: %s", e)

        except Exception as e:
            logger.error("Background memory extraction failed: %s", e)
