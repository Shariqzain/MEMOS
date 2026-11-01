"""
MEMOS — API Dependencies
Dependency injection for FastAPI routes.
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.database.chroma import VectorRepository
from app.database.mongo import (
    ConversationRepository,
    MemoryRepository,
    UserRepository,
    get_database,
)
from app.database.neo4j_db import GraphRepository
from app.services.chat import ChatService


async def get_db() -> AsyncIOMotorDatabase:
    return await get_database()


async def get_memory_repo(db=Depends(get_db)) -> MemoryRepository:
    return MemoryRepository(db)


async def get_conv_repo(db=Depends(get_db)) -> ConversationRepository:
    return ConversationRepository(db)


async def get_user_repo(db=Depends(get_db)) -> UserRepository:
    return UserRepository(db)


def get_vector_repo() -> VectorRepository:
    return VectorRepository()


def get_graph_repo() -> GraphRepository:
    return GraphRepository()


async def get_chat_service(
    memory_repo=Depends(get_memory_repo),
    conv_repo=Depends(get_conv_repo),
    user_repo=Depends(get_user_repo),
    vector_repo=Depends(get_vector_repo),
    graph_repo=Depends(get_graph_repo),
) -> ChatService:
    return ChatService(
        memory_repo=memory_repo,
        conv_repo=conv_repo,
        user_repo=user_repo,
        vector_repo=vector_repo,
        graph_repo=graph_repo,
    )
