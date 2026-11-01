"""
MEMOS — Graph Retrieval Service (Phase 6)
Neo4j-based entity and relationship traversal for memory retrieval.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from app.core.config import get_settings
from app.database.mongo import MemoryRepository
from app.database.neo4j_db import GraphRepository, is_available
from app.models.domain import Memory, MemoryStatus

logger = logging.getLogger(__name__)
settings = get_settings()


class GraphRetrievalCandidate:
    def __init__(
        self,
        memory_id: str,
        path_length: int,
        matched_entities: List[str],
        memory: Optional[Memory] = None,
        excluded: bool = False,
        exclusion_reason: str = "",
    ):
        self.memory_id = memory_id
        self.path_length = path_length
        self.matched_entities = matched_entities
        self.memory = memory
        self.excluded = excluded
        self.exclusion_reason = exclusion_reason
        # Graph score: inverse path length, normalized by entity matches
        self.score = self._compute_score()

    def _compute_score(self) -> float:
        entity_bonus = min(len(self.matched_entities) * 0.1, 0.3)
        path_score = 1.0 / max(self.path_length, 1)
        return min(path_score + entity_bonus, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "path_length": self.path_length,
            "matched_entities": self.matched_entities,
            "score": self.score,
            "excluded": self.excluded,
        }


def extract_entities_from_query(query: str) -> List[str]:
    """
    Simple rule-based entity extraction from query text.
    Identifies capitalized words and common noun phrases.
    For production use, replace with NER or LLM-based extraction.
    """
    # Capitalized words (potential proper nouns/entities)
    capitalized = re.findall(r'\b[A-Z][a-z]{2,}\b', query)
    # Words in quotes
    quoted = re.findall(r'"([^"]+)"', query)
    # Technical terms (camelCase or ALLCAPS)
    technical = re.findall(r'\b[A-Z][A-Za-z]{3,}\b|\b[A-Z]{3,}\b', query)

    entities = list(set(capitalized + quoted + technical))
    # Filter common English words
    stopwords = {"The", "This", "That", "What", "When", "Where", "Who", "How", "Why", "Can", "Are", "You", "Your"}
    entities = [e for e in entities if e not in stopwords and len(e) >= 3]
    return entities[:8]  # Cap to prevent overly broad queries


class GraphRetrievalService:
    """
    Graph traversal retrieval pipeline:
    1. Extract entities from query
    2. Traverse Neo4j graph
    3. Filter by lifecycle status
    4. Fetch canonical records
    """

    def __init__(self, graph_repo: GraphRepository, memory_repo: MemoryRepository):
        self.graph_repo = graph_repo
        self.memory_repo = memory_repo

    async def retrieve(
        self,
        query: str,
        user_id: str,
        entity_names: Optional[List[str]] = None,
        n_results: int = None,
        traversal_depth: int = 2,
    ) -> List[GraphRetrievalCandidate]:
        """
        Retrieve memories via graph traversal.
        Falls back gracefully if Neo4j is unavailable.
        """
        if not is_available():
            logger.debug("Neo4j unavailable, skipping graph retrieval.")
            return []

        n_results = n_results or settings.graph_retrieval_limit

        # Step 1: Extract entities
        entities = entity_names or extract_entities_from_query(query)
        if not entities:
            logger.debug("No entities extracted from query for graph retrieval.")
            return []

        logger.debug("Graph retrieval entities: %s", entities)

        # Step 2: Traverse graph
        try:
            graph_results = await self.graph_repo.retrieve_by_entities(
                user_id=user_id,
                entity_names=entities,
                depth=traversal_depth,
                limit=n_results,
            )
        except Exception as e:
            logger.error("Graph traversal failed: %s", e)
            return []

        if not graph_results:
            return []

        # Step 3: Fetch canonical records
        memory_ids = [r["memory_id"] for r in graph_results]
        memories = await self.memory_repo.get_by_ids(memory_ids, user_id)
        memory_map = {m.memory_id: m for m in memories}

        # Step 4: Build candidates with eligibility checks
        candidates = []
        for gr in graph_results:
            mid = gr["memory_id"]
            memory = memory_map.get(mid)

            if memory is None:
                candidates.append(GraphRetrievalCandidate(
                    memory_id=mid,
                    path_length=gr.get("path_length", 99),
                    matched_entities=gr.get("matched_entities", []),
                    excluded=True,
                    exclusion_reason="not_found_in_mongodb",
                ))
                continue

            if memory.status != MemoryStatus.ACTIVE:
                candidates.append(GraphRetrievalCandidate(
                    memory_id=mid,
                    path_length=gr.get("path_length", 99),
                    matched_entities=gr.get("matched_entities", []),
                    memory=memory,
                    excluded=True,
                    exclusion_reason=f"status_{memory.status.value}",
                ))
                continue

            candidates.append(GraphRetrievalCandidate(
                memory_id=mid,
                path_length=gr.get("path_length", 1),
                matched_entities=gr.get("matched_entities", []),
                memory=memory,
            ))

        # Sort by graph score descending
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    async def retrieve_active(
        self,
        query: str,
        user_id: str,
        entity_names: Optional[List[str]] = None,
        n_results: int = None,
    ) -> List[Memory]:
        """Convenience: returns only eligible Memory objects."""
        candidates = await self.retrieve(query, user_id, entity_names, n_results)
        return [c.memory for c in candidates if not c.excluded and c.memory]
