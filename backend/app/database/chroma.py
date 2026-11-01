"""
MEMOS — ChromaDB Vector Database Adapter
Handles embedding storage, semantic similarity search, and metadata filtering.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_chroma_client = None
_collection = None
COLLECTION_NAME = "memos_memories"


def _get_client():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        if settings.chroma_mode == "embedded":
            import os
            os.makedirs(settings.chroma_persist_path, exist_ok=True)
            _chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_path)
        else:
            _chroma_client = chromadb.HttpClient(
                host=settings.chroma_host, port=settings.chroma_port
            )
        logger.info("ChromaDB client initialized (mode=%s)", settings.chroma_mode)
    return _chroma_client


def get_collection():
    global _collection
    if _collection is None:
        client = _get_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection '%s' ready.", COLLECTION_NAME)
    return _collection


# ─────────────────────────────────────────────────────────────
# ChromaDB Repository
# ─────────────────────────────────────────────────────────────

class VectorRepository:
    """Thread-safe wrapper around ChromaDB for memory embeddings."""

    def upsert(
        self,
        memory_id: str,
        embedding: List[float],
        content: str,
        user_id: str,
        category: str,
        status: str = "active",
        importance: float = 0.5,
        metadata_extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        col = get_collection()
        meta: Dict[str, Any] = {
            "user_id": user_id,
            "category": category,
            "status": status,
            "importance": importance,
            "content_preview": content[:200],
        }
        if metadata_extra:
            meta.update(metadata_extra)
        try:
            col.upsert(
                ids=[memory_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[meta],
            )
        except Exception as e:
            logger.error("ChromaDB upsert failed for %s: %s", memory_id, e)
            raise

    def delete(self, memory_id: str) -> None:
        try:
            col = get_collection()
            col.delete(ids=[memory_id])
        except Exception as e:
            logger.warning("ChromaDB delete failed for %s: %s", memory_id, e)

    def search(
        self,
        query_embedding: List[float],
        user_id: str,
        n_results: int = 20,
        status_filter: str = "active",
        include_statuses: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns list of dicts: {memory_id, score, metadata, content_preview}
        Score is cosine similarity (0 to 1, higher = more similar).
        """
        col = get_collection()
        count = col.count()
        if count == 0:
            return []

        status_clause: Dict[str, Any]
        if include_statuses:
            if len(include_statuses) == 1:
                status_clause = {"status": {"$eq": include_statuses[0]}}
            else:
                status_clause = {"status": {"$in": include_statuses}}
        else:
            status_clause = {"status": {"$eq": status_filter}}

        where_filter = {
            "$and": [
                {"user_id": {"$eq": user_id}},
                status_clause,
            ]
        }

        try:
            results = col.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, count),
                where=where_filter,
                include=["distances", "metadatas", "documents"],
            )
        except Exception as e:
            logger.error("ChromaDB query failed: %s", e)
            return []

        candidates = []
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        documents = results.get("documents", [[]])[0]

        for i, mem_id in enumerate(ids):
            # ChromaDB cosine distance → similarity: score = 1 - distance
            distance = distances[i] if i < len(distances) else 1.0
            similarity = max(0.0, 1.0 - distance)
            candidates.append({
                "memory_id": mem_id,
                "score": similarity,
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "content_preview": documents[i] if i < len(documents) else "",
            })

        return candidates

    def get_count(self, user_id: Optional[str] = None) -> int:
        try:
            col = get_collection()
            if user_id:
                results = col.get(where={"user_id": {"$eq": user_id}})
                return len(results.get("ids", []))
            return col.count()
        except Exception:
            return 0

    def mark_status(self, memory_id: str, new_status: str) -> None:
        """Update status metadata for a memory (e.g., when superseded/archived)."""
        try:
            col = get_collection()
            col.update(ids=[memory_id], metadatas=[{"status": new_status}])
        except Exception as e:
            logger.warning("ChromaDB status update failed for %s: %s", memory_id, e)

    def rebuild_from_memories(
        self,
        memories: List[Dict[str, Any]],
        embedding_fn,
    ) -> int:
        """Rebuild the entire vector index from a list of memory dicts."""
        count = 0
        for mem in memories:
            try:
                emb = embedding_fn(mem["content"])
                self.upsert(
                    memory_id=mem["memory_id"],
                    embedding=emb,
                    content=mem["content"],
                    user_id=mem["user_id"],
                    category=mem.get("category", "general"),
                    status=mem.get("status", "active"),
                    importance=mem.get("importance", 0.5),
                )
                count += 1
            except Exception as e:
                logger.error("Rebuild failed for memory %s: %s", mem.get("memory_id"), e)
        return count
