"""
MEMOS — Memory Evolution Service (Phase 8)
Manages memory lifecycle: creation, updates, supersession, conflict detection,
redundancy reduction, and version history.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import get_settings
from app.database.chroma import VectorRepository
from app.database.mongo import MemoryRepository
from app.database.neo4j_db import GraphRepository
from app.llm.embeddings import embed_text
from app.llm.provider import llm_json
from app.memory.extractor import ExtractedMemoryCandidate
from app.models.domain import (
    EvolutionAction,
    Memory,
    MemoryCategory,
    MemoryEntity,
    MemoryRelationship,
    MemoryStatus,
    MemoryVersion,
)

logger = logging.getLogger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────────────────────
# Conflict / Evolution Analysis Prompt
# ─────────────────────────────────────────────────────────────

EVOLUTION_PROMPT = """You are analyzing whether a new piece of information should update or replace an existing memory.

Existing memory:
{existing}

New information:
{new_info}

Determine the relationship. Respond in JSON:
{{
  "action": "<duplicate|update|conflict|new|merge>",
  "reason": "<brief explanation>",
  "resolved_content": "<if update or merge: the best combined/updated statement, else null>",
  "confidence": <0.0-1.0>
}}

Actions:
- duplicate: same fact, no new information
- update: new info refines/replaces specific aspect of existing (same subject, same predicate, new value)
- conflict: contradictory facts that cannot both be true at once without scope clarification
- merge: both facts are true but can be expressed more clearly as one statement
- new: different topic, both facts can coexist independently"""


CONFLICT_RESOLUTION_PROMPT = """Two memories may be in conflict. Analyze them carefully.

Memory A: {memory_a}
Memory B: {memory_b}

Consider:
1. Same subject and predicate?
2. Different time scope or context?
3. Explicit correction language?
4. Can both be true simultaneously?

Respond in JSON:
{{
  "is_genuine_conflict": <true|false>,
  "resolution": "<keep_both|supersede_a|supersede_b|needs_clarification>",
  "reason": "<explanation>",
  "resolved_content": "<if supersede: the authoritative statement, else null>"
}}"""


class MemoryEvolutionService:
    """
    Orchestrates memory lifecycle:
    - Deduplication
    - Updates and supersession
    - Conflict detection and resolution
    - Version history recording
    - Cross-database synchronization
    """

    def __init__(
        self,
        memory_repo: MemoryRepository,
        vector_repo: VectorRepository,
        graph_repo: GraphRepository,
    ):
        self.memory_repo = memory_repo
        self.vector_repo = vector_repo
        self.graph_repo = graph_repo

    async def process_candidate(
        self,
        candidate: ExtractedMemoryCandidate,
        user_id: str,
        conversation_id: str,
    ) -> Optional[Memory]:
        """
        Process one extracted memory candidate through the full evolution pipeline.
        Returns the resulting Memory (created or updated), or None if rejected.
        """
        # Step 1: Generate embedding for similarity comparison
        try:
            embedding = embed_text(candidate.content)
        except Exception as e:
            logger.error("Embedding failed for candidate: %s", e)
            return None

        # Step 2: Find similar existing memories via vector search
        similar_results = self.vector_repo.search(
            query_embedding=embedding,
            user_id=user_id,
            n_results=5,
            include_statuses=["active", "needs_review"],
        )

        # Step 3: Filter by similarity threshold
        close_matches = [
            r for r in similar_results
            if r["score"] >= settings.memory_dedup_threshold
        ]

        if close_matches:
            # Very close match — check for duplicate/update/conflict
            existing_ids = [m["memory_id"] for m in close_matches]
            existing_memories = await self.memory_repo.get_by_ids(existing_ids, user_id)

            if existing_memories:
                return await self._handle_similar(
                    candidate=candidate,
                    existing_memories=existing_memories,
                    embedding=embedding,
                    user_id=user_id,
                    conversation_id=conversation_id,
                )

        # Step 4: Also check moderate similarity (possible conflict with different content)
        moderate_results = [
            r for r in similar_results
            if 0.6 <= r["score"] < settings.memory_dedup_threshold
        ]
        if moderate_results:
            moderate_ids = [m["memory_id"] for m in moderate_results]
            moderate_memories = await self.memory_repo.get_by_ids(moderate_ids, user_id)
            if moderate_memories:
                conflict_result = await self._check_conflicts(
                    candidate, moderate_memories, user_id
                )
                if conflict_result:
                    return conflict_result

        # Step 5: New memory — create it
        return await self._create_memory(
            candidate=candidate,
            embedding=embedding,
            user_id=user_id,
            conversation_id=conversation_id,
        )

    async def _handle_similar(
        self,
        candidate: ExtractedMemoryCandidate,
        existing_memories: List[Memory],
        embedding: List[float],
        user_id: str,
        conversation_id: str,
    ) -> Optional[Memory]:
        """Determine whether close match is duplicate, update, or conflict."""
        # Use the most similar (first) as primary comparison
        primary = existing_memories[0]

        # Fast path: identical content → duplicate
        if candidate.content.strip().lower() == primary.content.strip().lower():
            logger.debug("Duplicate detected: %s", candidate.content[:60])
            primary.touch()
            await self.memory_repo.update(primary)
            return primary

        # LLM analysis for nuanced comparison
        try:
            analysis = await llm_json(
                messages=[{
                    "role": "user",
                    "content": EVOLUTION_PROMPT.format(
                        existing=primary.content,
                        new_info=candidate.content,
                    )
                }],
                temperature=0.1,
            )
        except Exception as e:
            logger.warning("LLM evolution analysis failed: %s — treating as new memory", e)
            return await self._create_memory(candidate, embedding, user_id, conversation_id)

        action = analysis.get("action", "new")
        reason = analysis.get("reason", "")
        resolved_content = analysis.get("resolved_content")

        logger.info("Evolution action for '%s': %s", candidate.content[:50], action)

        if action == "duplicate":
            primary.touch()
            await self.memory_repo.update(primary)
            return primary

        elif action in ("update", "merge"):
            return await self._update_memory(
                existing=primary,
                candidate=candidate,
                resolved_content=resolved_content or candidate.content,
                embedding=embedding,
                evolution_action=(
                    EvolutionAction.UPDATED if action == "update"
                    else EvolutionAction.MERGED
                ),
                reason=reason,
                user_id=user_id,
                conversation_id=conversation_id,
            )

        elif action == "conflict":
            return await self._handle_conflict(
                existing=primary,
                candidate=candidate,
                embedding=embedding,
                reason=reason,
                user_id=user_id,
                conversation_id=conversation_id,
            )

        else:  # "new" or unknown
            return await self._create_memory(candidate, embedding, user_id, conversation_id)

    async def _check_conflicts(
        self,
        candidate: ExtractedMemoryCandidate,
        existing: List[Memory],
        user_id: str,
    ) -> Optional[Memory]:
        """Check moderate-similarity memories for genuine conflicts."""
        for existing_mem in existing[:2]:
            try:
                result = await llm_json(
                    messages=[{
                        "role": "user",
                        "content": CONFLICT_RESOLUTION_PROMPT.format(
                            memory_a=existing_mem.content,
                            memory_b=candidate.content,
                        )
                    }],
                    temperature=0.1,
                )
                if result.get("is_genuine_conflict") and result.get("resolution") in (
                    "supersede_a", "supersede_b"
                ):
                    logger.info("Conflict detected between memories, resolution: %s", result.get("resolution"))
                    # Mark existing as needs_review
                    existing_mem.status = MemoryStatus.NEEDS_REVIEW
                    existing_mem.conflict_with_memory_ids.append("pending")
                    await self.memory_repo.update(existing_mem)
                    self.vector_repo.mark_status(existing_mem.memory_id, "needs_review")
            except Exception as e:
                logger.warning("Conflict check failed: %s", e)
        return None  # Will be handled as new memory

    async def _create_memory(
        self,
        candidate: ExtractedMemoryCandidate,
        embedding: List[float],
        user_id: str,
        conversation_id: str,
    ) -> Memory:
        """Create a brand-new memory and persist to all three databases."""
        memory = Memory(
            user_id=user_id,
            content=candidate.content,
            category=candidate.category,
            entities=[
                MemoryEntity(
                    name=e.get("name", ""),
                    entity_type=e.get("entity_type", "entity"),
                )
                for e in candidate.entities if e.get("name")
            ],
            relationships=[
                MemoryRelationship(
                    subject=r.get("subject", ""),
                    predicate=r.get("predicate", "related"),
                    obj=r.get("obj", ""),
                )
                for r in candidate.relationships if r.get("subject") and r.get("obj")
            ],
            importance=candidate.importance,
            confidence=candidate.confidence,
            memory_type=candidate.memory_type,
            temporal_scope=candidate.temporal_scope,
            source_conversation_id=conversation_id,
            source_message_ids=candidate.source_message_ids,
            extraction_reason=candidate.extraction_reason,
        )

        # MongoDB
        await self.memory_repo.insert(memory)

        # Save initial version
        await self._save_version(memory, EvolutionAction.CREATED, "New memory created")

        # ChromaDB
        try:
            self.vector_repo.upsert(
                memory_id=memory.memory_id,
                embedding=embedding,
                content=memory.content,
                user_id=user_id,
                category=memory.category.value,
                status=memory.status.value,
                importance=memory.importance,
            )
            memory.vector_synced = True
            await self.memory_repo.update(memory)
        except Exception as e:
            logger.error("ChromaDB sync failed for new memory %s: %s", memory.memory_id, e)

        # Neo4j
        try:
            await self.graph_repo.upsert_memory(
                memory_id=memory.memory_id,
                user_id=user_id,
                content=memory.content,
                category=memory.category.value,
                importance=memory.importance,
                status=memory.status.value,
                entities=[e.model_dump() for e in memory.entities],
                relationships=[r.model_dump() for r in memory.relationships],
            )
            memory.graph_synced = True
            await self.memory_repo.update(memory)
        except Exception as e:
            logger.warning("Neo4j sync failed for new memory %s: %s", memory.memory_id, e)

        logger.info("Created memory %s: %s", memory.memory_id, memory.content[:60])
        return memory

    async def _update_memory(
        self,
        existing: Memory,
        candidate: ExtractedMemoryCandidate,
        resolved_content: str,
        embedding: List[float],
        evolution_action: EvolutionAction,
        reason: str,
        user_id: str,
        conversation_id: str,
    ) -> Memory:
        """Update an existing memory with new information."""
        # Save version snapshot before update
        await self._save_version(existing, evolution_action, reason)

        # Update fields
        existing.content = resolved_content
        existing.importance = max(existing.importance, candidate.importance)
        existing.confidence = max(existing.confidence, candidate.confidence)
        if candidate.category != MemoryCategory.GENERAL:
            existing.category = candidate.category
        existing.version += 1
        existing.source_conversation_id = conversation_id
        existing.mark_updated()

        await self.memory_repo.update(existing)

        # Update ChromaDB
        try:
            self.vector_repo.upsert(
                memory_id=existing.memory_id,
                embedding=embedding,
                content=existing.content,
                user_id=user_id,
                category=existing.category.value,
                status=existing.status.value,
                importance=existing.importance,
            )
        except Exception as e:
            logger.error("ChromaDB update failed: %s", e)

        # Update Neo4j
        try:
            await self.graph_repo.upsert_memory(
                memory_id=existing.memory_id,
                user_id=user_id,
                content=existing.content,
                category=existing.category.value,
                importance=existing.importance,
                status=existing.status.value,
                entities=[e.model_dump() for e in existing.entities],
                relationships=[r.model_dump() for r in existing.relationships],
            )
        except Exception as e:
            logger.warning("Neo4j update failed: %s", e)

        logger.info("Updated memory %s (%s): %s", existing.memory_id, evolution_action.value, existing.content[:60])
        return existing

    async def _handle_conflict(
        self,
        existing: Memory,
        candidate: ExtractedMemoryCandidate,
        embedding: List[float],
        reason: str,
        user_id: str,
        conversation_id: str,
    ) -> Optional[Memory]:
        """Handle a conflicting memory. Creates new memory, marks existing as needs_review."""
        # Mark existing as needing review
        existing.status = MemoryStatus.NEEDS_REVIEW
        await self.memory_repo.update(existing)
        self.vector_repo.mark_status(existing.memory_id, "needs_review")
        await self._save_version(existing, EvolutionAction.CONFLICT_FLAGGED, reason)

        # Create new memory with conflict reference
        new_memory = await self._create_memory(candidate, embedding, user_id, conversation_id)
        new_memory.conflict_with_memory_ids.append(existing.memory_id)
        existing.conflict_with_memory_ids.append(new_memory.memory_id)
        await self.memory_repo.update(new_memory)
        await self.memory_repo.update(existing)

        logger.info(
            "Conflict: memory %s conflicts with new %s",
            existing.memory_id, new_memory.memory_id,
        )
        return new_memory

    async def supersede_memory(
        self,
        old_memory_id: str,
        new_memory: Memory,
        reason: str,
        user_id: str,
    ) -> None:
        """Mark an old memory as superseded by a new one."""
        old = await self.memory_repo.get_by_id(old_memory_id, user_id)
        if not old:
            return

        await self._save_version(old, EvolutionAction.SUPERSEDED, reason)
        old.status = MemoryStatus.SUPERSEDED
        old.superseded_by_memory_id = new_memory.memory_id
        old.mark_updated()
        await self.memory_repo.update(old)

        new_memory.supersedes_memory_id = old_memory_id
        await self.memory_repo.update(new_memory)

        # Sync
        self.vector_repo.mark_status(old_memory_id, "superseded")
        try:
            await self.graph_repo.mark_superseded(old_memory_id, new_memory.memory_id)
        except Exception as e:
            logger.warning("Neo4j supersede sync failed: %s", e)

    async def archive_memory(self, memory_id: str, user_id: str, reason: str = "") -> bool:
        """Archive a memory (soft removal from active retrieval)."""
        memory = await self.memory_repo.get_by_id(memory_id, user_id)
        if not memory:
            return False
        await self._save_version(memory, EvolutionAction.ARCHIVED, reason)
        memory.status = MemoryStatus.ARCHIVED
        memory.mark_updated()
        await self.memory_repo.update(memory)
        self.vector_repo.mark_status(memory_id, "archived")
        await self.graph_repo.delete_memory(memory_id)
        return True

    async def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """Soft delete a memory."""
        memory = await self.memory_repo.get_by_id(memory_id, user_id)
        if not memory:
            return False
        await self._save_version(memory, EvolutionAction.DELETED, "User requested deletion")
        success = await self.memory_repo.soft_delete(memory_id, user_id)
        if success:
            self.vector_repo.delete(memory_id)
            await self.graph_repo.delete_memory(memory_id)
        return success

    async def restore_memory(
        self, memory_id: str, version_number: int, user_id: str
    ) -> Optional[Memory]:
        """Restore a memory to a previous version."""
        memory = await self.memory_repo.get_by_id(memory_id, user_id)
        if not memory:
            return None

        version = await self.memory_repo.get_version_by_number(memory_id, version_number, user_id)
        if not version:
            return None

        # Save current as a version snapshot
        await self._save_version(memory, EvolutionAction.RESTORED, f"Restored to v{version_number}")

        # Apply snapshot
        snapshot = version.snapshot
        if snapshot:
            memory.content = snapshot.get("content", memory.content)
            memory.category = MemoryCategory(snapshot.get("category", memory.category.value))
            memory.importance = snapshot.get("importance", memory.importance)
            memory.confidence = snapshot.get("confidence", memory.confidence)

        memory.status = MemoryStatus.ACTIVE
        memory.version += 1
        memory.mark_updated()
        await self.memory_repo.update(memory)

        # Sync embeddings
        try:
            embedding = embed_text(memory.content)
            self.vector_repo.upsert(
                memory_id=memory.memory_id,
                embedding=embedding,
                content=memory.content,
                user_id=user_id,
                category=memory.category.value,
                status="active",
                importance=memory.importance,
            )
        except Exception as e:
            logger.error("Embedding sync failed after restore: %s", e)

        return memory

    async def _save_version(
        self, memory: Memory, action: EvolutionAction, reason: str = ""
    ) -> None:
        """Save a version snapshot of the current memory state."""
        version = MemoryVersion(
            memory_id=memory.memory_id,
            user_id=memory.user_id,
            version=memory.version,
            content=memory.content,
            category=memory.category,
            importance=memory.importance,
            confidence=memory.confidence,
            status=memory.status,
            evolution_action=action,
            change_reason=reason,
            snapshot=memory.model_dump(mode="json"),
        )
        await self.memory_repo.save_version(version)
