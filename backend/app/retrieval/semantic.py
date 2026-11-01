"""
MEMOS — Semantic Retrieval Service (Phase 5)
ChromaDB-based semantic similarity retrieval with lifecycle filtering.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.core.config import get_settings
from app.database.chroma import VectorRepository
from app.database.mongo import MemoryRepository
from app.llm.embeddings import embed_text
from app.models.domain import Memory, MemoryStatus

logger = logging.getLogger(__name__)
settings = get_settings()


class SemanticRetrievalCandidate:
    def __init__(
        self,
        memory_id: str,
        score: float,
        memory: Optional[Memory] = None,
        excluded: bool = False,
        exclusion_reason: str = "",
    ):
        self.memory_id = memory_id
        self.score = score
        self.memory = memory
        self.excluded = excluded
        self.exclusion_reason = exclusion_reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "score": self.score,
            "excluded": self.excluded,
            "exclusion_reason": self.exclusion_reason,
        }


class SemanticRetrievalService:
    """
    Semantic retrieval pipeline:
    1. Embed query
    2. Search ChromaDB
    3. Apply lifecycle filters
    4. Fetch canonical records from MongoDB
    5. Return ranked candidates
    """

    def __init__(self, vector_repo: VectorRepository, memory_repo: MemoryRepository):
        self.vector_repo = vector_repo
        self.memory_repo = memory_repo

    async def retrieve(
        self,
        query: str,
        user_id: str,
        n_results: int = None,
        threshold: float = None,
        include_diagnostics: bool = False,
    ) -> List[SemanticRetrievalCandidate]:
        """
        Retrieve semantically relevant memories for a query.
        Returns SemanticRetrievalCandidate list, sorted by score descending.
        """
        n_results = n_results or settings.semantic_retrieval_limit
        threshold = threshold or settings.similarity_threshold

        # Step 1: Generate embedding
        try:
            query_embedding = embed_text(query)
        except Exception as e:
            logger.error("Embedding generation failed: %s", e)
            return []

        # Step 2: ChromaDB search
        raw_results = self.vector_repo.search(
            query_embedding=query_embedding,
            user_id=user_id,
            n_results=n_results,
            include_statuses=["active"],
        )

        if not raw_results:
            return []

        # Step 3: Apply threshold filter
        memory_ids = []
        score_map: Dict[str, float] = {}
        for r in raw_results:
            mid = r["memory_id"]
            score = r["score"]
            if score >= threshold:
                memory_ids.append(mid)
                score_map[mid] = score
            elif include_diagnostics:
                pass  # Could track excluded-by-threshold

        if not memory_ids:
            return []

        # Step 4: Fetch canonical records
        memories = await self.memory_repo.get_by_ids(memory_ids, user_id)
        memory_map: Dict[str, Memory] = {m.memory_id: m for m in memories}

        # Step 5: Build candidates with eligibility checks
        candidates = []
        for mid in memory_ids:
            score = score_map[mid]
            memory = memory_map.get(mid)

            if memory is None:
                # Stale vector record
                candidates.append(SemanticRetrievalCandidate(
                    memory_id=mid, score=score, excluded=True,
                    exclusion_reason="not_found_in_mongodb",
                ))
                continue

            # Lifecycle eligibility
            if memory.status != MemoryStatus.ACTIVE:
                candidates.append(SemanticRetrievalCandidate(
                    memory_id=mid, score=score, memory=memory, excluded=True,
                    exclusion_reason=f"status_{memory.status.value}",
                ))
                continue

            # Confidence filter
            if memory.confidence < settings.min_confidence_score:
                candidates.append(SemanticRetrievalCandidate(
                    memory_id=mid, score=score, memory=memory, excluded=True,
                    exclusion_reason="low_confidence",
                ))
                continue

            candidates.append(SemanticRetrievalCandidate(
                memory_id=mid, score=score, memory=memory, excluded=False,
            ))

        # Sort by score descending
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    async def retrieve_active(
        self,
        query: str,
        user_id: str,
        n_results: int = None,
        threshold: float = None,
    ) -> List[Memory]:
        """Convenience: returns only active, eligible Memory objects."""
        candidates = await self.retrieve(query, user_id, n_results, threshold)
        return [c.memory for c in candidates if not c.excluded and c.memory]
