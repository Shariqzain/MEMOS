"""
MEMOS — Context Composition and Response Generation (Phase 9)
Composes prompts with retrieved memories and generates personalized responses.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.core.config import get_settings
from app.llm.provider import llm_complete
from app.models.domain import Memory, Message, MessageRole

logger = logging.getLogger(__name__)
settings = get_settings()

# ─────────────────────────────────────────────────────────────
# System Prompt Template (versioned)
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """You are MEMOS Assistant, a helpful AI with persistent long-term memory.
You remember information about the user across conversations and use it to provide personalized responses.

## Your capabilities:
- You remember user preferences, projects, goals, and past conversations
- You use this memory to give more relevant and personalized answers
- You acknowledge when you're uncertain about something
- You're honest about the limits of your knowledge and memory

## Memory context:
{memory_context}

## Guidelines:
- Use the provided memory context to personalize your response
- If memory is available about the topic, reference it naturally
- Don't pretend to remember things not in your memory context
- If memories conflict, acknowledge the uncertainty rather than guessing
- Keep responses helpful, clear, and appropriately concise
- Never claim information as certain if it comes from uncertain memory (low confidence)

Current date: {current_date}"""

NO_MEMORY_PROMPT = """You are MEMOS Assistant, a helpful AI assistant.
You currently have no relevant memories about this topic stored for this user.
Respond helpfully based on the conversation so far.

Current date: {current_date}"""

# Context budget (approximate character count for memory section)
MAX_MEMORY_CONTEXT_CHARS = 3000
MAX_HISTORY_TURNS = 10


def _format_memory_context(memories: List[Memory]) -> str:
    """Format memories into a context string for the prompt."""
    if not memories:
        return "No relevant memories found."

    lines = ["Relevant memories about the user:"]
    total_chars = 0

    for i, mem in enumerate(memories, 1):
        category = mem.category.value.replace("_", " ").title()
        confidence_note = " (uncertain)" if mem.confidence < 0.6 else ""
        line = f"{i}. [{category}]{confidence_note} {mem.content}"

        if total_chars + len(line) > MAX_MEMORY_CONTEXT_CHARS:
            lines.append(f"... and {len(memories) - i + 1} more memories.")
            break

        lines.append(line)
        total_chars += len(line)

        if mem.temporal_scope:
            lines.append(f"   (Time context: {mem.temporal_scope})")

    return "\n".join(lines)


def _format_conversation_history(messages: List[Message]) -> List[Dict[str, str]]:
    """Convert Message objects to LLM chat format."""
    result = []
    for msg in messages[-MAX_HISTORY_TURNS:]:
        if msg.role in (MessageRole.USER, MessageRole.ASSISTANT):
            result.append({
                "role": msg.role.value,
                "content": msg.content,
            })
    return result


class ResponseGenerationService:
    """
    Composes contextual prompts and generates personalized responses.
    """

    async def generate(
        self,
        user_message: str,
        conversation_history: List[Message],
        memories: List[Memory],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate a personalized response using retrieved memories as context.

        Args:
            user_message: The current user message
            conversation_history: Previous messages in this conversation
            memories: Retrieved relevant memories
            temperature: LLM sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Generated assistant response text
        """
        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Build system prompt
        if memories:
            memory_context = _format_memory_context(memories)
            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                memory_context=memory_context,
                current_date=current_date,
            )
        else:
            system_prompt = NO_MEMORY_PROMPT.format(current_date=current_date)

        # Build message list for LLM
        chat_messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history (excluding current message)
        history_msgs = _format_conversation_history(conversation_history)
        chat_messages.extend(history_msgs)

        # Add current user message
        chat_messages.append({"role": "user", "content": user_message})

        # Generate response
        try:
            response = await llm_complete(
                messages=chat_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response
        except Exception as e:
            logger.error("LLM response generation failed: %s", e)
            raise

    def compose_prompt_preview(
        self,
        user_message: str,
        memories: List[Memory],
    ) -> Dict[str, Any]:
        """Return a preview of the composed prompt for debugging."""
        memory_context = _format_memory_context(memories)
        return {
            "memory_count": len(memories),
            "memory_context_chars": len(memory_context),
            "memory_context_preview": memory_context[:500],
            "user_message_chars": len(user_message),
        }
