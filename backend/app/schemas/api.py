"""
MEMOS — API Request/Response Schemas
Pydantic schemas for FastAPI endpoints.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.models.domain import MemoryCategory, MemoryStatus, MemoryType


# ─────────────────────────────────────────────────────────────
# Chat
# ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_id: str = Field(default="demo_user", description="User identifier")
    conversation_id: Optional[str] = Field(default=None, description="Existing conversation ID")
    message: str = Field(..., min_length=1, max_length=8000)
    retrieval_mode: str = Field(default="hybrid", description="hybrid | semantic | graph | none")
    stream: bool = False


class MemorySummary(BaseModel):
    memory_id: str
    content: str
    category: str
    importance: float
    confidence: float
    rrf_score: Optional[float] = None


class RetrievalDiagnostics(BaseModel):
    semantic_candidates: int = 0
    graph_candidates: int = 0
    final_memories: int = 0
    retrieval_mode: str = "hybrid"
    timing_ms: Optional[int] = None


class ChatResponse(BaseModel):
    conversation_id: str
    message_id: str
    response: str
    memories_used: List[MemorySummary] = Field(default_factory=list)
    diagnostics: Optional[RetrievalDiagnostics] = None
    processing_ms: Optional[int] = None


# ─────────────────────────────────────────────────────────────
# Conversation
# ─────────────────────────────────────────────────────────────

class ConversationOut(BaseModel):
    conversation_id: str
    user_id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    message_count: int


class MessageOut(BaseModel):
    message_id: str
    conversation_id: str
    role: str
    content: str
    memory_ids_used: List[str]
    created_at: datetime
    processing_ms: Optional[int] = None


# ─────────────────────────────────────────────────────────────
# Memory CRUD
# ─────────────────────────────────────────────────────────────

class MemoryOut(BaseModel):
    memory_id: str
    user_id: str
    content: str
    category: str
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    importance: float
    confidence: float
    memory_type: str
    status: str
    version: int
    source_conversation_id: Optional[str] = None
    supersedes_memory_id: Optional[str] = None
    superseded_by_memory_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    last_accessed_at: Optional[datetime] = None


class MemoryUpdate(BaseModel):
    content: Optional[str] = None
    category: Optional[MemoryCategory] = None
    importance: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    status: Optional[MemoryStatus] = None


class MemoryVersionOut(BaseModel):
    version_id: str
    memory_id: str
    version: int
    content: str
    category: str
    evolution_action: str
    change_reason: Optional[str]
    created_at: datetime


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

class EvaluationSummary(BaseModel):
    total_memories: int = 0
    active_memories: int = 0
    superseded_memories: int = 0
    archived_memories: int = 0
    avg_importance: float = 0.0
    avg_confidence: float = 0.0
    total_conversations: int = 0
    total_messages: int = 0
    memory_categories: Dict[str, int] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str = "0.1.0"
    services: Dict[str, str] = Field(default_factory=dict)
