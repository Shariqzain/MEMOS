"""
MEMOS — MongoDB Database Adapter
Handles all MongoDB operations: connection, collections, indexes, CRUD.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING, IndexModel
from pymongo.errors import DuplicateKeyError, OperationFailure

from app.core.config import get_settings
from app.models.domain import (
    Conversation,
    Memory,
    MemoryStatus,
    MemoryVersion,
    Message,
    User,
)

logger = logging.getLogger(__name__)
settings = get_settings()

# ─────────────────────────────────────────────────────────────
# Connection singleton
# ─────────────────────────────────────────────────────────────

_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None


async def get_database() -> AsyncIOMotorDatabase:
    global _client, _db
    if _db is None:
        _client = AsyncIOMotorClient(settings.mongo_uri, serverSelectionTimeoutMS=5000)
        _db = _client[settings.mongo_db_name]
        await _ensure_indexes(_db)
        logger.info("MongoDB connected: %s / %s", settings.mongo_uri, settings.mongo_db_name)
    return _db


async def close_database() -> None:
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None
        logger.info("MongoDB connection closed.")


async def _ensure_indexes(db: AsyncIOMotorDatabase) -> None:
    """Create required indexes on all collections."""
    try:
        # Memories
        await db.memories.create_indexes([
            IndexModel([("user_id", ASCENDING), ("status", ASCENDING)]),
            IndexModel([("memory_id", ASCENDING)], unique=True),
            IndexModel([("user_id", ASCENDING), ("category", ASCENDING)]),
            IndexModel([("user_id", ASCENDING), ("created_at", DESCENDING)]),
            IndexModel([("source_conversation_id", ASCENDING)]),
            IndexModel([("vector_synced", ASCENDING)]),
            IndexModel([("graph_synced", ASCENDING)]),
        ])
        # Memory versions
        await db.memory_versions.create_indexes([
            IndexModel([("memory_id", ASCENDING), ("version", ASCENDING)]),
            IndexModel([("version_id", ASCENDING)], unique=True),
            IndexModel([("user_id", ASCENDING)]),
        ])
        # Conversations
        await db.conversations.create_indexes([
            IndexModel([("conversation_id", ASCENDING)], unique=True),
            IndexModel([("user_id", ASCENDING), ("updated_at", DESCENDING)]),
        ])
        # Messages
        await db.messages.create_indexes([
            IndexModel([("message_id", ASCENDING)], unique=True),
            IndexModel([("conversation_id", ASCENDING), ("created_at", ASCENDING)]),
            IndexModel([("user_id", ASCENDING)]),
        ])
        # Users
        await db.users.create_indexes([
            IndexModel([("user_id", ASCENDING)], unique=True),
        ])
        logger.info("MongoDB indexes ensured.")
    except OperationFailure as e:
        logger.warning("Index creation issue (may already exist): %s", e)


# ─────────────────────────────────────────────────────────────
# Memory CRUD
# ─────────────────────────────────────────────────────────────

class MemoryRepository:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.col = db.memories
        self.versions_col = db.memory_versions

    async def insert(self, memory: Memory) -> Memory:
        doc = memory.model_dump()
        await self.col.insert_one(doc)
        return memory

    async def get_by_id(self, memory_id: str, user_id: str) -> Optional[Memory]:
        doc = await self.col.find_one({"memory_id": memory_id, "user_id": user_id})
        if doc:
            doc.pop("_id", None)
            return Memory(**doc)
        return None

    async def update(self, memory: Memory) -> Memory:
        memory.mark_updated()
        doc = memory.model_dump()
        await self.col.replace_one({"memory_id": memory.memory_id}, doc, upsert=True)
        return memory

    async def soft_delete(self, memory_id: str, user_id: str) -> bool:
        result = await self.col.update_one(
            {"memory_id": memory_id, "user_id": user_id},
            {"$set": {"status": MemoryStatus.DELETED, "updated_at": datetime.now(timezone.utc)}}
        )
        return result.modified_count > 0

    async def list_active(
        self,
        user_id: str,
        category: Optional[str] = None,
        limit: int = 100,
        skip: int = 0,
    ) -> List[Memory]:
        query: Dict[str, Any] = {
            "user_id": user_id,
            "status": MemoryStatus.ACTIVE,
        }
        if category:
            query["category"] = category
        cursor = self.col.find(query).sort("created_at", DESCENDING).skip(skip).limit(limit)
        docs = await cursor.to_list(length=limit)
        memories = []
        for doc in docs:
            doc.pop("_id", None)
            try:
                memories.append(Memory(**doc))
            except Exception as e:
                logger.warning("Skipping malformed memory doc: %s", e)
        return memories

    async def get_by_ids(self, memory_ids: List[str], user_id: str) -> List[Memory]:
        cursor = self.col.find({"memory_id": {"$in": memory_ids}, "user_id": user_id})
        docs = await cursor.to_list(length=len(memory_ids))
        result = []
        for doc in docs:
            doc.pop("_id", None)
            try:
                result.append(Memory(**doc))
            except Exception as e:
                logger.warning("Skipping malformed memory: %s", e)
        return result

    async def find_similar_content(
        self, user_id: str, content_keywords: List[str], limit: int = 5
    ) -> List[Memory]:
        """Rough text-based pre-filter for duplication checks."""
        if not content_keywords:
            return []
        regex_parts = "|".join(content_keywords[:5])
        cursor = self.col.find({
            "user_id": user_id,
            "status": {"$in": [MemoryStatus.ACTIVE, MemoryStatus.NEEDS_REVIEW]},
            "content": {"$regex": regex_parts, "$options": "i"},
        }).limit(limit)
        docs = await cursor.to_list(length=limit)
        result = []
        for doc in docs:
            doc.pop("_id", None)
            try:
                result.append(Memory(**doc))
            except Exception:
                pass
        return result

    async def count_active(self, user_id: str) -> int:
        return await self.col.count_documents({"user_id": user_id, "status": MemoryStatus.ACTIVE})

    async def get_category_counts(self, user_id: str) -> Dict[str, int]:
        pipeline = [
            {"$match": {"user_id": user_id, "status": MemoryStatus.ACTIVE}},
            {"$group": {"_id": "$category", "count": {"$sum": 1}}},
        ]
        cursor = self.col.aggregate(pipeline)
        docs = await cursor.to_list(length=100)
        return {doc["_id"]: doc["count"] for doc in docs}

    async def get_unsynced_vector(self, limit: int = 50) -> List[Memory]:
        cursor = self.col.find({"vector_synced": False, "status": MemoryStatus.ACTIVE}).limit(limit)
        docs = await cursor.to_list(length=limit)
        result = []
        for doc in docs:
            doc.pop("_id", None)
            try:
                result.append(Memory(**doc))
            except Exception:
                pass
        return result

    # ── Version history ───────────────────────────────────────

    async def save_version(self, version: MemoryVersion) -> None:
        doc = version.model_dump()
        try:
            await self.versions_col.insert_one(doc)
        except DuplicateKeyError:
            logger.warning("Duplicate version record ignored: %s v%d", version.memory_id, version.version)

    async def get_versions(self, memory_id: str, user_id: str) -> List[MemoryVersion]:
        cursor = self.versions_col.find(
            {"memory_id": memory_id, "user_id": user_id}
        ).sort("version", ASCENDING)
        docs = await cursor.to_list(length=100)
        result = []
        for doc in docs:
            doc.pop("_id", None)
            try:
                result.append(MemoryVersion(**doc))
            except Exception as e:
                logger.warning("Bad version doc: %s", e)
        return result

    async def get_version_by_number(
        self, memory_id: str, version: int, user_id: str
    ) -> Optional[MemoryVersion]:
        doc = await self.versions_col.find_one(
            {"memory_id": memory_id, "version": version, "user_id": user_id}
        )
        if doc:
            doc.pop("_id", None)
            return MemoryVersion(**doc)
        return None


# ─────────────────────────────────────────────────────────────
# Conversation / Message repositories
# ─────────────────────────────────────────────────────────────

class ConversationRepository:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.col = db.conversations
        self.msg_col = db.messages

    async def create(self, convo: Conversation) -> Conversation:
        await self.col.insert_one(convo.model_dump())
        return convo

    async def get(self, conversation_id: str, user_id: str) -> Optional[Conversation]:
        doc = await self.col.find_one({"conversation_id": conversation_id, "user_id": user_id})
        if doc:
            doc.pop("_id", None)
            return Conversation(**doc)
        return None

    async def update(self, convo: Conversation) -> None:
        convo.updated_at = datetime.now(timezone.utc)
        await self.col.replace_one({"conversation_id": convo.conversation_id}, convo.model_dump())

    async def list_for_user(self, user_id: str, limit: int = 50) -> List[Conversation]:
        cursor = self.col.find({"user_id": user_id}).sort("updated_at", DESCENDING).limit(limit)
        docs = await cursor.to_list(length=limit)
        result = []
        for doc in docs:
            doc.pop("_id", None)
            result.append(Conversation(**doc))
        return result

    async def save_message(self, msg: Message) -> Message:
        await self.msg_col.insert_one(msg.model_dump())
        await self.col.update_one(
            {"conversation_id": msg.conversation_id},
            {"$inc": {"message_count": 1}, "$set": {"updated_at": datetime.now(timezone.utc)}}
        )
        return msg

    async def get_messages(self, conversation_id: str, limit: int = 50) -> List[Message]:
        cursor = self.msg_col.find(
            {"conversation_id": conversation_id}
        ).sort("created_at", ASCENDING).limit(limit)
        docs = await cursor.to_list(length=limit)
        result = []
        for doc in docs:
            doc.pop("_id", None)
            result.append(Message(**doc))
        return result

    async def count_messages(self, user_id: str) -> int:
        return await self.msg_col.count_documents({"user_id": user_id})

    async def count_conversations(self, user_id: str) -> int:
        return await self.col.count_documents({"user_id": user_id})


# ─────────────────────────────────────────────────────────────
# User repository
# ─────────────────────────────────────────────────────────────

class UserRepository:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.col = db.users

    async def get_or_create(self, user_id: str, name: str = "User") -> User:
        doc = await self.col.find_one({"user_id": user_id})
        if doc:
            doc.pop("_id", None)
            return User(**doc)
        user = User(user_id=user_id, name=name)
        await self.col.insert_one(user.model_dump())
        return user
