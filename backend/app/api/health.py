"""
MEMOS — Health and Evaluation API Routes
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from app.api.deps import get_conv_repo, get_memory_repo
from app.core.config import get_settings
from app.database.mongo import ConversationRepository, MemoryRepository
from app.database.neo4j_db import is_available as neo4j_available
from app.llm.provider import get_llm_provider
from app.schemas.api import EvaluationSummary, HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


@router.get("/health", response_model=HealthResponse)
async def health_check(
    memory_repo: MemoryRepository = Depends(get_memory_repo),
) -> HealthResponse:
    """Health check endpoint verifying all services."""
    services = {}

    # MongoDB
    try:
        from app.database.mongo import get_database
        db = await get_database()
        await db.command("ping")
        services["mongodb"] = "ok"
    except Exception as e:
        services["mongodb"] = f"error: {str(e)[:50]}"

    # ChromaDB
    try:
        from app.database.chroma import get_collection
        col = get_collection()
        col.count()
        services["chromadb"] = "ok"
    except Exception as e:
        services["chromadb"] = f"error: {str(e)[:50]}"

    # Neo4j
    services["neo4j"] = "ok" if neo4j_available() else "disabled"

    # LLM (Ollama)
    try:
        provider = get_llm_provider()
        is_healthy = await provider.health_check()
        services["llm"] = "ok" if is_healthy else "unreachable"
    except Exception as e:
        services["llm"] = f"error: {str(e)[:50]}"

    overall = "healthy" if all(v == "ok" or v == "disabled" for v in services.values()) else "degraded"
    return HealthResponse(status=overall, services=services)


@router.get("/evaluation/summary", response_model=EvaluationSummary)
async def evaluation_summary(
    user_id: str = "demo_user",
    memory_repo: MemoryRepository = Depends(get_memory_repo),
    conv_repo: ConversationRepository = Depends(get_conv_repo),
) -> EvaluationSummary:
    """Return summary statistics for evaluation."""
    from app.models.domain import MemoryStatus

    all_memories = await memory_repo.list_active(user_id, limit=1000)
    # Count different statuses
    active = len(all_memories)

    # Superseded / archived require separate queries
    from app.database.mongo import get_database
    db = await get_database()
    superseded = await db.memories.count_documents({"user_id": user_id, "status": "superseded"})
    archived = await db.memories.count_documents({"user_id": user_id, "status": "archived"})
    total = active + superseded + archived

    avg_importance = sum(m.importance for m in all_memories) / max(len(all_memories), 1)
    avg_confidence = sum(m.confidence for m in all_memories) / max(len(all_memories), 1)

    category_counts = await memory_repo.get_category_counts(user_id)
    total_convs = await conv_repo.count_conversations(user_id)
    total_msgs = await conv_repo.count_messages(user_id)

    return EvaluationSummary(
        total_memories=total,
        active_memories=active,
        superseded_memories=superseded,
        archived_memories=archived,
        avg_importance=round(avg_importance, 3),
        avg_confidence=round(avg_confidence, 3),
        total_conversations=total_convs,
        total_messages=total_msgs,
        memory_categories=category_counts,
    )
