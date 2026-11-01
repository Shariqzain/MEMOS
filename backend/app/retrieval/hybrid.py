"""
MEMOS — Hybrid Retrieval using Reciprocal Rank Fusion (Phase 7)
Combines semantic (ChromaDB) and graph (Neo4j) retrieval via RRF.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.config import get_settings
from app.database.mongo import MemoryRepository
from app.models.domain import Memory
from app.retrieval.graph import GraphRetrievalCandidate, GraphRetrievalService
from app.retrieval.semantic import SemanticRetrievalCandidate, SemanticRetrievalService

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class HybridCandidate:
    """A memory candidate with RRF score and source rank information."""
    memory_id: str
    rrf_score: float = 0.0
    semantic_rank: Optional[int] = None
    graph_rank: Optional[int] = None
    semantic_score: Optional[float] = None
    graph_score: Optional[float] = None
    memory: Optional[Memory] = None
    sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "rrf_score": round(self.rrf_score, 6),
            "semantic_rank": self.semantic_rank,
            "graph_rank": self.graph_rank,
            "semantic_score": self.semantic_score,
            "graph_score": self.graph_score,
            "sources": self.sources,
        }


def compute_rrf(
    ranked_lists: List[List[str]],
    k: int = 60,
) -> Dict[str, float]:
    """
    Compute Reciprocal Rank Fusion scores.

    RRF(d) = Σ_{r ∈ R} 1 / (k + rank_r(d))

    Args:
        ranked_lists: Each inner list is an ordered list of memory_ids (rank 1 = index 0)
        k: RRF smoothing constant (default 60)

    Returns:
        Dict mapping memory_id → RRF score (higher is better)
    """
    scores: Dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank_idx, mem_id in enumerate(ranked_list):
            rank = rank_idx + 1  # 1-based ranking
            scores[mem_id] = scores.get(mem_id, 0.0) + 1.0 / (k + rank)
    return scores


class HybridRetrievalService:
    """
    Hybrid retrieval combining semantic and graph results via RRF.
    Supports semantic-only, graph-only, and full hybrid modes.
    """

    def __init__(
        self,
        semantic_service: SemanticRetrievalService,
        graph_service: GraphRetrievalService,
        memory_repo: MemoryRepository,
    ):
        self.semantic = semantic_service
        self.graph = graph_service
        self.memory_repo = memory_repo

    async def retrieve(
        self,
        query: str,
        user_id: str,
        mode: str = "hybrid",
        n_results: int = None,
        rrf_k: int = None,
    ) -> List[HybridCandidate]:
        """
        Perform hybrid retrieval using RRF.

        modes:
        - "hybrid": combine semantic + graph (default)
        - "semantic": semantic only
        - "graph": graph only
        - "none": return no memories (baseline)
        """
        n_results = n_results or settings.final_context_limit
        rrf_k = rrf_k or settings.rrf_k

        if mode == "none":
            return []

        semantic_candidates: List[SemanticRetrievalCandidate] = []
        graph_candidates: List[GraphRetrievalCandidate] = []

        # ── Retrieval ─────────────────────────────────────────
        if mode in ("hybrid", "semantic"):
            try:
                semantic_candidates = await self.semantic.retrieve(
                    query=query,
                    user_id=user_id,
                    n_results=settings.semantic_retrieval_limit,
                    include_diagnostics=True,
                )
            except Exception as e:
                logger.error("Semantic retrieval failed: %s", e)

        if mode in ("hybrid", "graph"):
            try:
                graph_candidates = await self.graph.retrieve(
                    query=query,
                    user_id=user_id,
                    n_results=settings.graph_retrieval_limit,
                )
            except Exception as e:
                logger.error("Graph retrieval failed: %s", e)

        # ── Filter active-only candidates ─────────────────────
        active_semantic = [c for c in semantic_candidates if not c.excluded]
        active_graph = [c for c in graph_candidates if not c.excluded]

        # ── Build ranked lists for RRF ─────────────────────────
        semantic_ranked = [c.memory_id for c in active_semantic]
        graph_ranked = [c.memory_id for c in active_graph]

        semantic_rank_map: Dict[str, int] = {mid: i + 1 for i, mid in enumerate(semantic_ranked)}
        graph_rank_map: Dict[str, int] = {mid: i + 1 for i, mid in enumerate(graph_ranked)}
        semantic_score_map: Dict[str, float] = {c.memory_id: c.score for c in active_semantic}
        graph_score_map: Dict[str, float] = {c.memory_id: c.score for c in active_graph}

        # ── RRF ───────────────────────────────────────────────
        ranked_lists = []
        if semantic_ranked:
            ranked_lists.append(semantic_ranked)
        if graph_ranked:
            ranked_lists.append(graph_ranked)

        if not ranked_lists:
            return []

        rrf_scores = compute_rrf(ranked_lists, k=rrf_k)

        # ── Build HybridCandidates ─────────────────────────────
        all_memory_ids = list(rrf_scores.keys())

        # Collect pre-fetched memories
        memory_map: Dict[str, Memory] = {}
        for c in active_semantic:
            if c.memory:
                memory_map[c.memory_id] = c.memory
        for c in active_graph:
            if c.memory:
                memory_map[c.memory_id] = c.memory

        # Fetch any missing
        missing_ids = [mid for mid in all_memory_ids if mid not in memory_map]
        if missing_ids:
            fetched = await self.memory_repo.get_by_ids(missing_ids, user_id)
            for m in fetched:
                memory_map[m.memory_id] = m

        # ── Assemble final candidates ──────────────────────────
        hybrid_candidates: List[HybridCandidate] = []
        for mid, rrf_score in rrf_scores.items():
            sources = []
            if mid in semantic_rank_map:
                sources.append("semantic")
            if mid in graph_rank_map:
                sources.append("graph")

            candidate = HybridCandidate(
                memory_id=mid,
                rrf_score=rrf_score,
                semantic_rank=semantic_rank_map.get(mid),
                graph_rank=graph_rank_map.get(mid),
                semantic_score=semantic_score_map.get(mid),
                graph_score=graph_score_map.get(mid),
                memory=memory_map.get(mid),
                sources=sources,
            )
            hybrid_candidates.append(candidate)

        # Sort by RRF score descending
        hybrid_candidates.sort(key=lambda c: c.rrf_score, reverse=True)

        # Apply final context limit
        return hybrid_candidates[:n_results]

    async def retrieve_memories(
        self,
        query: str,
        user_id: str,
        mode: str = "hybrid",
        n_results: int = None,
    ) -> List[Memory]:
        """Convenience: returns Memory objects from hybrid retrieval."""
        candidates = await self.retrieve(query, user_id, mode, n_results)
        return [c.memory for c in candidates if c.memory]

    def get_diagnostics(self, candidates: List[HybridCandidate]) -> Dict[str, Any]:
        """Return diagnostic information about a retrieval result."""
        return {
            "total_candidates": len(candidates),
            "semantic_only": sum(1 for c in candidates if c.sources == ["semantic"]),
            "graph_only": sum(1 for c in candidates if c.sources == ["graph"]),
            "both_sources": sum(1 for c in candidates if len(c.sources) == 2),
            "candidates": [c.to_dict() for c in candidates],
        }
