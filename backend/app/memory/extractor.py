"""
MEMOS — Memory Extraction Service (Phase 4)
Extracts structured memory candidates from conversation turns using LLM.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from app.core.config import get_settings
from app.llm.provider import llm_json
from app.models.domain import MemoryCategory, MemoryType

logger = logging.getLogger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────────────────────
# Extraction Schema
# ─────────────────────────────────────────────────────────────

class ExtractedMemoryCandidate(BaseModel):
    """A memory candidate extracted from a conversation turn."""
    content: str = Field(..., min_length=3, max_length=1000)
    category: MemoryCategory = MemoryCategory.GENERAL
    entities: List[Dict[str, str]] = Field(default_factory=list)
    relationships: List[Dict[str, str]] = Field(default_factory=list)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    memory_type: MemoryType = MemoryType.LONG_TERM
    temporal_scope: Optional[str] = None
    source_message_ids: List[str] = Field(default_factory=list)
    update_target: Optional[str] = None  # memory_id of existing memory to update
    extraction_reason: str = ""

    @field_validator("importance", "confidence", mode="before")
    @classmethod
    def clamp_float(cls, v) -> float:
        return max(0.0, min(1.0, float(v)))

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, v) -> MemoryCategory:
        if isinstance(v, str):
            v_lower = v.lower()
            for cat in MemoryCategory:
                if cat.value == v_lower:
                    return cat
        return MemoryCategory.GENERAL


EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction assistant for an AI system called MEMOS.
Your job is to analyze conversation turns and identify information worth remembering long-term.

## What to extract:
- User preferences, habits, and values
- Ongoing projects, goals, and objectives  
- Technical skills, tools, and workflows
- Important constraints and requirements
- Named entities and their relationships
- Changes to previously stated facts (mark these as updates)

## What NOT to extract:
- Generic conversational filler ("hi", "thanks", "sure")
- Highly time-sensitive, one-off information (e.g., "I need to buy milk today")
- Information the user has not established about themselves
- Sensitive personal data (passwords, financial details, medical records)
- Information already obvious from the current message context

## Output format:
Return a JSON object with a "memories" key containing an array of memory candidates.
Each candidate must have:
- content: Clear, self-contained statement of the fact (3-1000 chars)
- category: one of [preference, goal, project, skill, constraint, relationship, temporal, factual, general]
- entities: array of {name, entity_type} objects mentioned
- relationships: array of {subject, predicate, obj} objects
- importance: 0.0-1.0 (higher = more useful for future conversations)
- confidence: 0.0-1.0 (how certain is this fact)
- memory_type: "long_term" | "short_term" | "episodic"
- temporal_scope: optional time scope string
- extraction_reason: brief reason why this is worth remembering

Return {"memories": []} if nothing is worth remembering."""


async def extract_memories_from_turn(
    user_message: str,
    assistant_response: str,
    conversation_history: List[Dict[str, str]],
    existing_memory_summaries: Optional[List[str]] = None,
    source_message_ids: Optional[List[str]] = None,
) -> List[ExtractedMemoryCandidate]:
    """
    Extract memory candidates from a completed conversation turn.
    Returns validated ExtractedMemoryCandidate objects.
    """
    # Build context
    history_text = ""
    for msg in conversation_history[-6:]:  # Last 3 exchanges
        role = msg.get("role", "user")
        content = msg.get("content", "")[:300]
        history_text += f"{role.upper()}: {content}\n"

    existing_ctx = ""
    if existing_memory_summaries:
        existing_ctx = "\n## Existing memories (for deduplication):\n" + "\n".join(
            f"- {s}" for s in existing_memory_summaries[:10]
        )

    user_content = f"""## Recent conversation:
{history_text}
USER: {user_message}
ASSISTANT: {assistant_response[:500]}
{existing_ctx}

Extract memory candidates from the USER's messages only. Focus on what the user has expressed about themselves, their goals, preferences, and projects."""

    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        result = await llm_json(messages, temperature=0.1, max_tokens=1500)
    except Exception as e:
        logger.warning("LLM extraction call failed (%s), using rule-based heuristic extraction fallback", e)
        candidates = _heuristic_extract(user_message, source_message_ids)
        return _apply_retention_policy(candidates)

    raw_memories = result.get("memories", [])
    if not isinstance(raw_memories, list):
        logger.warning("LLM extraction returned non-list memories field.")
        return []

    candidates = []
    for raw in raw_memories:
        if not isinstance(raw, dict):
            continue
        try:
            candidate = ExtractedMemoryCandidate(**raw)
            if source_message_ids:
                candidate.source_message_ids = source_message_ids
            candidates.append(candidate)
        except Exception as e:
            logger.warning("Invalid memory candidate skipped: %s — %s", raw, e)

    # Apply retention policy
    accepted = _apply_retention_policy(candidates)
    logger.info(
        "Extraction: %d raw → %d accepted candidates",
        len(candidates), len(accepted),
    )
    return accepted


def _apply_retention_policy(
    candidates: List[ExtractedMemoryCandidate],
) -> List[ExtractedMemoryCandidate]:
    """Filter candidates based on configurable retention thresholds."""
    accepted = []
    for c in candidates:
        # Reject below threshold
        if c.importance < settings.min_importance_score:
            logger.debug("Rejected (low importance=%.2f): %s", c.importance, c.content[:50])
            continue
        if c.confidence < settings.min_confidence_score:
            logger.debug("Rejected (low confidence=%.2f): %s", c.confidence, c.content[:50])
            continue
        # Reject very short or trivial content
        if len(c.content.strip()) < 10:
            continue
        accepted.append(c)
    return accepted


def score_memory_importance(
    content: str,
    category: MemoryCategory,
    memory_type: MemoryType,
    entities: List[Dict],
    base_importance: float,
) -> float:
    """
    Refine importance score using deterministic rules as a post-LLM calibration.
    """
    score = base_importance

    # Category boosts
    category_weights = {
        MemoryCategory.GOAL: 0.10,
        MemoryCategory.PROJECT: 0.10,
        MemoryCategory.CONSTRAINT: 0.08,
        MemoryCategory.PREFERENCE: 0.06,
        MemoryCategory.SKILL: 0.05,
        MemoryCategory.RELATIONSHIP: 0.05,
    }
    score += category_weights.get(category, 0.0)

    # Long-term memories are more valuable
    if memory_type == MemoryType.LONG_TERM:
        score += 0.05
    elif memory_type == MemoryType.SHORT_TERM:
        score -= 0.05

    # Named entities increase specificity
    if len(entities) >= 2:
        score += 0.05

    return max(0.0, min(1.0, score))


def _heuristic_extract(
    user_message: str,
    source_message_ids: Optional[List[str]] = None,
) -> List[ExtractedMemoryCandidate]:
    """Heuristic rule-based memory extraction when LLM service is offline."""
    import re
    msg = user_message.strip()
    if not msg:
        return []

    patterns = [
        (r"(?:i am|i'm|i work on|working on|building)\s+(?:a\s+|an\s+)?(.+)", MemoryCategory.PROJECT, 0.85),
        (r"(?:i prefer|i like|i love|favorite|favourite)\s+(.+)", MemoryCategory.PREFERENCE, 0.85),
        (r"(?:i want|my goal is|planning to|aiming to)\s+(.+)", MemoryCategory.GOAL, 0.85),
        (r"(?:i live in|i am a|my name is|i work at)\s+(.+)", MemoryCategory.FACTUAL, 0.80),
        (r"(?:i cannot|i can't|i must not|never use)\s+(.+)", MemoryCategory.CONSTRAINT, 0.80),
    ]

    for pat, cat, base_imp in patterns:
        m = re.search(pat, msg, re.IGNORECASE)
        if m:
            content = msg
            if not content.lower().startswith("user"):
                content = "User " + content[0].lower() + content[1:] if len(content) > 1 else content

            raw_entities = re.findall(
                r'\b(?:[A-Z][a-zA-Z0-9_\-\.]+|Python|React|FastAPI|MongoDB|Neo4j|ChromaDB|Docker|TypeScript|JavaScript|Node)\b',
                msg,
            )
            entities = [{"name": e, "entity_type": "CONCEPT"} for e in set(raw_entities)]
            return [ExtractedMemoryCandidate(
                content=content,
                category=cat,
                entities=entities,
                importance=base_imp,
                confidence=0.90,
                memory_type=MemoryType.LONG_TERM,
                extraction_reason="Heuristic rule extraction",
                source_message_ids=source_message_ids or [],
            )]

    if len(msg) >= 15 and not msg.endswith("?"):
        content = "User stated: " + msg
        return [ExtractedMemoryCandidate(
            content=content,
            category=MemoryCategory.FACTUAL,
            entities=[],
            importance=0.60,
            confidence=0.75,
            memory_type=MemoryType.LONG_TERM,
            extraction_reason="Heuristic informative statement",
            source_message_ids=source_message_ids or [],
        )]

    return []

