"""
MEMOS — Memory Management API Routes (Phase 10)
"""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_memory_repo, get_vector_repo, get_graph_repo
from app.database.chroma import VectorRepository
from app.database.mongo import MemoryRepository
from app.database.neo4j_db import GraphRepository
from app.llm.embeddings import embed_text
from app.memory.evolution import MemoryEvolutionService
from app.models.domain import MemoryStatus
from app.schemas.api import MemoryOut, MemoryUpdate, MemoryVersionOut

logger = logging.getLogger(__name__)
router = APIRouter()


def _memory_to_out(memory) -> MemoryOut:
    return MemoryOut(
        memory_id=memory.memory_id,
        user_id=memory.user_id,
        content=memory.content,
        category=memory.category.value,
        entities=[e.model_dump() for e in memory.entities],
        importance=memory.importance,
        confidence=memory.confidence,
        memory_type=memory.memory_type.value,
        status=memory.status.value,
        version=memory.version,
        source_conversation_id=memory.source_conversation_id,
        supersedes_memory_id=memory.supersedes_memory_id,
        superseded_by_memory_id=memory.superseded_by_memory_id,
        created_at=memory.created_at,
        updated_at=memory.updated_at,
        last_accessed_at=memory.last_accessed_at,
    )


@router.get("/memories", response_model=List[MemoryOut])
async def list_memories(
    user_id: str = "demo_user",
    category: Optional[str] = Query(default=None),
    limit: int = Query(default=50, le=200),
    skip: int = Query(default=0),
    memory_repo: MemoryRepository = Depends(get_memory_repo),
) -> List[MemoryOut]:
    """List active memories for a user, with optional category filter."""
    memories = await memory_repo.list_active(
        user_id=user_id, category=category, limit=limit, skip=skip
    )
    return [_memory_to_out(m) for m in memories]


@router.get("/memories/{memory_id}", response_model=MemoryOut)
async def get_memory(
    memory_id: str,
    user_id: str = "demo_user",
    memory_repo: MemoryRepository = Depends(get_memory_repo),
) -> MemoryOut:
    """Get a specific memory by ID."""
    memory = await memory_repo.get_by_id(memory_id, user_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    return _memory_to_out(memory)


@router.patch("/memories/{memory_id}", response_model=MemoryOut)
async def update_memory(
    memory_id: str,
    update: MemoryUpdate,
    user_id: str = "demo_user",
    memory_repo: MemoryRepository = Depends(get_memory_repo),
    vector_repo: VectorRepository = Depends(get_vector_repo),
    graph_repo: GraphRepository = Depends(get_graph_repo),
) -> MemoryOut:
    """Update a memory's content, category, or scores."""
    memory = await memory_repo.get_by_id(memory_id, user_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    if memory.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    evolution_svc = MemoryEvolutionService(memory_repo, vector_repo, graph_repo)

    # Apply updates
    if update.content is not None:
        memory.content = update.content
    if update.category is not None:
        memory.category = update.category
    if update.importance is not None:
        memory.importance = update.importance
    if update.confidence is not None:
        memory.confidence = update.confidence
    if update.status is not None:
        if update.status == MemoryStatus.ARCHIVED:
            await evolution_svc.archive_memory(memory_id, user_id, "User archived")
            return _memory_to_out(await memory_repo.get_by_id(memory_id, user_id))
        memory.status = update.status

    memory.mark_updated()
    await memory_repo.update(memory)

    # Re-sync vector if content changed
    if update.content is not None:
        try:
            embedding = embed_text(memory.content)
            vector_repo.upsert(
                memory_id=memory.memory_id,
                embedding=embedding,
                content=memory.content,
                user_id=user_id,
                category=memory.category.value,
                status=memory.status.value,
                importance=memory.importance,
            )
        except Exception as e:
            logger.warning("Vector re-sync failed after update: %s", e)

    return _memory_to_out(memory)


@router.delete("/memories/{memory_id}")
async def delete_memory(
    memory_id: str,
    user_id: str = "demo_user",
    memory_repo: MemoryRepository = Depends(get_memory_repo),
    vector_repo: VectorRepository = Depends(get_vector_repo),
    graph_repo: GraphRepository = Depends(get_graph_repo),
):
    """Soft-delete a memory (marks as deleted, removes from retrieval)."""
    memory = await memory_repo.get_by_id(memory_id, user_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    if memory.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    evolution_svc = MemoryEvolutionService(memory_repo, vector_repo, graph_repo)
    success = await evolution_svc.delete_memory(memory_id, user_id)
    if not success:
        raise HTTPException(status_code=500, detail="Deletion failed")
    return {"message": "Memory deleted", "memory_id": memory_id}


@router.get("/memories/{memory_id}/history", response_model=List[MemoryVersionOut])
async def get_memory_history(
    memory_id: str,
    user_id: str = "demo_user",
    memory_repo: MemoryRepository = Depends(get_memory_repo),
) -> List[MemoryVersionOut]:
    """Get the full version history for a memory."""
    memory = await memory_repo.get_by_id(memory_id, user_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")

    versions = await memory_repo.get_versions(memory_id, user_id)
    return [
        MemoryVersionOut(
            version_id=v.version_id,
            memory_id=v.memory_id,
            version=v.version,
            content=v.content,
            category=v.category.value,
            evolution_action=v.evolution_action.value,
            change_reason=v.change_reason,
            created_at=v.created_at,
        )
        for v in versions
    ]


@router.post("/memories/{memory_id}/restore", response_model=MemoryOut)
async def restore_memory(
    memory_id: str,
    version: int,
    user_id: str = "demo_user",
    memory_repo: MemoryRepository = Depends(get_memory_repo),
    vector_repo: VectorRepository = Depends(get_vector_repo),
    graph_repo: GraphRepository = Depends(get_graph_repo),
) -> MemoryOut:
    """Restore a memory to a previous version."""
    evolution_svc = MemoryEvolutionService(memory_repo, vector_repo, graph_repo)
    restored = await evolution_svc.restore_memory(memory_id, version, user_id)
    if not restored:
        raise HTTPException(status_code=404, detail="Memory or version not found")
    return _memory_to_out(restored)
