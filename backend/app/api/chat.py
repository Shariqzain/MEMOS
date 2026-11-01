"""
MEMOS — Chat API Routes
"""
from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_chat_service, get_conv_repo, get_memory_repo
from app.database.mongo import ConversationRepository, MemoryRepository
from app.schemas.api import (
    ChatRequest,
    ChatResponse,
    ConversationOut,
    MessageOut,
)
from app.services.chat import ChatService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """
    Send a message and get a memory-augmented response.

    The endpoint:
    1. Retrieves relevant memories via hybrid RRF
    2. Generates a personalized response
    3. Stores the conversation
    4. Extracts and stores new memories in the background
    """
    try:
        return await chat_service.handle_chat(request)
    except Exception as e:
        logger.error("Chat endpoint error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@router.get("/conversations", response_model=List[ConversationOut])
async def list_conversations(
    user_id: str = "demo_user",
    limit: int = 50,
    conv_repo: ConversationRepository = Depends(get_conv_repo),
) -> List[ConversationOut]:
    """List all conversations for a user."""
    conversations = await conv_repo.list_for_user(user_id, limit=limit)
    return [
        ConversationOut(
            conversation_id=c.conversation_id,
            user_id=c.user_id,
            title=c.title,
            created_at=c.created_at,
            updated_at=c.updated_at,
            message_count=c.message_count,
        )
        for c in conversations
    ]


@router.get("/conversations/{conversation_id}/messages", response_model=List[MessageOut])
async def get_messages(
    conversation_id: str,
    user_id: str = "demo_user",
    limit: int = 50,
    conv_repo: ConversationRepository = Depends(get_conv_repo),
) -> List[MessageOut]:
    """Retrieve messages for a specific conversation."""
    messages = await conv_repo.get_messages(conversation_id, limit=limit)
    # Filter to user's conversation only
    return [
        MessageOut(
            message_id=m.message_id,
            conversation_id=m.conversation_id,
            role=m.role.value,
            content=m.content,
            memory_ids_used=m.memory_ids_used,
            created_at=m.created_at,
            processing_ms=m.processing_ms,
        )
        for m in messages
        if m.user_id == user_id
    ]
