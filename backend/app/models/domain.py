"""
MEMOS — Domain Models
Pydantic models representing core domain objects stored in MongoDB.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid4())


# ─────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────

class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"
    DELETED = "deleted"
    NEEDS_REVIEW = "needs_review"


class MemoryCategory(str, Enum):
    PREFERENCE = "preference"
    GOAL = "goal"
    PROJECT = "project"
    SKILL = "skill"
    CONSTRAINT = "constraint"
    RELATIONSHIP = "relationship"
    TEMPORAL = "temporal"
    FACTUAL = "factual"
    GENERAL = "general"


class MemoryType(str, Enum):
    LONG_TERM = "long_term"
    SHORT_TERM = "short_term"
    EPISODIC = "episodic"


class EvolutionAction(str, Enum):
    CREATED = "created"
    UPDATED = "updated"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"
    MERGED = "merged"
    CONFLICT_FLAGGED = "conflict_flagged"
    RESTORED = "restored"
    DELETED = "deleted"


# ─────────────────────────────────────────────────────────────
# Memory
# ─────────────────────────────────────────────────────────────

class MemoryEntity(BaseModel):
    """An entity mentioned in a memory."""
    name: str
    entity_type: str = "entity"
    attributes: Dict[str, Any] = Field(default_factory=dict)


class MemoryRelationship(BaseModel):
    """A relationship between entities in a memory."""
    subject: str
    predicate: str
    obj: str  # object is a reserved Python word
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class Memory(BaseModel):
    """Canonical memory record stored in MongoDB."""
    memory_id: str = Field(default_factory=_uuid)
    user_id: str
    content: str
    category: MemoryCategory = MemoryCategory.GENERAL
    entities: List[MemoryEntity] = Field(default_factory=list)
    relationships: List[MemoryRelationship] = Field(default_factory=list)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    memory_type: MemoryType = MemoryType.LONG_TERM
    temporal_scope: Optional[str] = None
    source_conversation_id: Optional[str] = None
    source_message_ids: List[str] = Field(default_factory=list)
    extraction_reason: Optional[str] = None
    status: MemoryStatus = MemoryStatus.ACTIVE
    version: int = 1
    supersedes_memory_id: Optional[str] = None
    superseded_by_memory_id: Optional[str] = None
    conflict_with_memory_ids: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    last_accessed_at: Optional[datetime] = None
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    # Vector / graph sync flags
    vector_synced: bool = False
    graph_synced: bool = False

    def touch(self) -> None:
        self.last_accessed_at = _now()

    def mark_updated(self) -> None:
        self.updated_at = _now()
        self.version += 1


class MemoryVersion(BaseModel):
    """Immutable snapshot of a memory at a particular version."""
    version_id: str = Field(default_factory=_uuid)
    memory_id: str
    user_id: str
    version: int
    content: str
    category: MemoryCategory
    importance: float
    confidence: float
    status: MemoryStatus
    evolution_action: EvolutionAction
    change_reason: Optional[str] = None
    source_message_id: Optional[str] = None
    snapshot: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_now)


# ─────────────────────────────────────────────────────────────
# Conversation / Message
# ─────────────────────────────────────────────────────────────

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    message_id: str = Field(default_factory=_uuid)
    conversation_id: str
    user_id: str
    role: MessageRole
    content: str
    memory_ids_used: List[str] = Field(default_factory=list)
    retrieval_diagnostics: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=_now)
    processing_ms: Optional[int] = None


class Conversation(BaseModel):
    conversation_id: str = Field(default_factory=_uuid)
    user_id: str
    title: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    message_count: int = 0


# ─────────────────────────────────────────────────────────────
# User
# ─────────────────────────────────────────────────────────────

class User(BaseModel):
    user_id: str = Field(default_factory=_uuid)
    name: str
    email: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    settings: Dict[str, Any] = Field(default_factory=dict)
