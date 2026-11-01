"""
MEMOS — Unit Tests
Tests for: RRF, memory extraction validation, scoring, lifecycle transitions
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.retrieval.hybrid import compute_rrf, HybridCandidate
from app.memory.extractor import (
    ExtractedMemoryCandidate,
    _apply_retention_policy,
    score_memory_importance,
)
from app.models.domain import (
    Memory,
    MemoryCategory,
    MemoryStatus,
    MemoryType,
)


# ─────────────────────────────────────────────────────────────
# RRF Tests
# ─────────────────────────────────────────────────────────────

class TestRRF:
    def test_single_list(self):
        """RRF with one list gives 1/(k+rank) scores."""
        ranked = [["A", "B", "C"]]
        scores = compute_rrf(ranked, k=60)
        assert scores["A"] == pytest.approx(1 / 61)
        assert scores["B"] == pytest.approx(1 / 62)
        assert scores["C"] == pytest.approx(1 / 63)

    def test_two_lists_agreement(self):
        """Items ranked first in both lists get highest score."""
        list1 = ["A", "B", "C"]
        list2 = ["A", "C", "B"]
        scores = compute_rrf([list1, list2], k=60)
        # A is rank 1 in both
        assert scores["A"] == pytest.approx(2 / 61)
        # B is rank 2 in list1 and rank 3 in list2
        assert scores["B"] == pytest.approx(1 / 62 + 1 / 63)
        # C is rank 3 in list1 and rank 2 in list2
        assert scores["C"] == pytest.approx(1 / 63 + 1 / 62)
        # A should beat both B and C
        assert scores["A"] > scores["B"]
        assert scores["A"] > scores["C"]

    def test_disjoint_lists(self):
        """Items in only one list still receive partial RRF score."""
        scores = compute_rrf([["A"], ["B"]], k=60)
        assert "A" in scores
        assert "B" in scores
        assert scores["A"] == pytest.approx(1 / 61)
        assert scores["B"] == pytest.approx(1 / 61)

    def test_empty_lists(self):
        """Empty ranked lists produce empty scores."""
        scores = compute_rrf([], k=60)
        assert scores == {}

    def test_configurable_k(self):
        """Different k values produce different scores."""
        scores_k60 = compute_rrf([["A"]], k=60)
        scores_k10 = compute_rrf([["A"]], k=10)
        assert scores_k10["A"] > scores_k60["A"]

    def test_ordering_preserved(self):
        """Top ranked item beats lower ranked item from two lists."""
        scores = compute_rrf([["A", "B"], ["A", "B"]], k=60)
        assert scores["A"] > scores["B"]

    def test_large_rank_list(self):
        """100 items — first should beat last."""
        items = [f"item_{i}" for i in range(100)]
        scores = compute_rrf([items], k=60)
        assert scores["item_0"] > scores["item_99"]


# ─────────────────────────────────────────────────────────────
# Memory Extraction / Retention Tests
# ─────────────────────────────────────────────────────────────

class TestRetentionPolicy:
    def _make_candidate(self, importance=0.7, confidence=0.8, content="Test important fact") -> ExtractedMemoryCandidate:
        return ExtractedMemoryCandidate(
            content=content,
            importance=importance,
            confidence=confidence,
        )

    def test_high_quality_accepted(self):
        c = self._make_candidate(importance=0.8, confidence=0.9)
        result = _apply_retention_policy([c])
        assert len(result) == 1

    def test_low_importance_rejected(self):
        c = self._make_candidate(importance=0.1, confidence=0.9)
        result = _apply_retention_policy([c])
        assert len(result) == 0

    def test_low_confidence_rejected(self):
        c = self._make_candidate(importance=0.8, confidence=0.2)
        result = _apply_retention_policy([c])
        assert len(result) == 0

    def test_both_at_threshold_boundary(self):
        # At exactly the threshold, the retention policy accepts (uses strict <)
        c = self._make_candidate(importance=0.35, confidence=0.40)
        result = _apply_retention_policy([c])
        assert len(result) == 1  # Exactly at threshold passes (< check)

    def test_below_threshold_rejected(self):
        # Strictly below threshold is rejected
        c = self._make_candidate(importance=0.34, confidence=0.39)
        result = _apply_retention_policy([c])
        assert len(result) == 0

    def test_very_short_content_rejected(self):
        # Content shorter than 10 chars is rejected by retention policy
        # Note: Pydantic enforces min_length=3, so we use a 3-9 char content
        c = ExtractedMemoryCandidate(content="Short", importance=0.9, confidence=0.9)
        result = _apply_retention_policy([c])
        assert len(result) == 0

    def test_empty_list(self):
        result = _apply_retention_policy([])
        assert result == []

    def test_mixed_candidates(self):
        good = self._make_candidate(0.8, 0.8, "I prefer Python for data science")
        bad_importance = self._make_candidate(0.1, 0.9, "Thanks for the help today")
        bad_confidence = self._make_candidate(0.8, 0.1, "Probably likes coffee")
        result = _apply_retention_policy([good, bad_importance, bad_confidence])
        assert len(result) == 1
        assert result[0].content == "I prefer Python for data science"


# ─────────────────────────────────────────────────────────────
# Memory Candidate Schema Validation
# ─────────────────────────────────────────────────────────────

class TestCandidateSchema:
    def test_valid_candidate(self):
        c = ExtractedMemoryCandidate(
            content="User prefers Python for backend development",
            category="preference",
            importance=0.8,
            confidence=0.9,
        )
        assert c.category == MemoryCategory.PREFERENCE

    def test_importance_clamped(self):
        c = ExtractedMemoryCandidate(content="Test", importance=1.5)
        assert c.importance == 1.0

    def test_confidence_clamped(self):
        c = ExtractedMemoryCandidate(content="Test", confidence=-0.5)
        assert c.confidence == 0.0

    def test_invalid_category_falls_back(self):
        c = ExtractedMemoryCandidate(content="Test", category="invalid_category")
        assert c.category == MemoryCategory.GENERAL

    def test_default_memory_type(self):
        c = ExtractedMemoryCandidate(content="User likes hiking")
        assert c.memory_type == MemoryType.LONG_TERM

    def test_content_too_short_raises(self):
        with pytest.raises(Exception):
            ExtractedMemoryCandidate(content="ab")  # min_length=3


# ─────────────────────────────────────────────────────────────
# Importance Scoring Tests
# ─────────────────────────────────────────────────────────────

class TestImportanceScoring:
    def test_goal_gets_boost(self):
        score = score_memory_importance(
            content="User wants to build a startup",
            category=MemoryCategory.GOAL,
            memory_type=MemoryType.LONG_TERM,
            entities=[],
            base_importance=0.5,
        )
        assert score > 0.5

    def test_general_no_boost(self):
        score = score_memory_importance(
            content="Some general fact",
            category=MemoryCategory.GENERAL,
            memory_type=MemoryType.LONG_TERM,
            entities=[],
            base_importance=0.5,
        )
        assert score >= 0.5  # Long-term bonus still applies

    def test_entities_increase_score(self):
        score_no_entities = score_memory_importance(
            content="Test",
            category=MemoryCategory.GENERAL,
            memory_type=MemoryType.LONG_TERM,
            entities=[],
            base_importance=0.5,
        )
        score_with_entities = score_memory_importance(
            content="Test",
            category=MemoryCategory.GENERAL,
            memory_type=MemoryType.LONG_TERM,
            entities=[{"name": "Python"}, {"name": "FastAPI"}],
            base_importance=0.5,
        )
        assert score_with_entities > score_no_entities

    def test_score_clamped_to_1(self):
        score = score_memory_importance(
            content="Test",
            category=MemoryCategory.GOAL,
            memory_type=MemoryType.LONG_TERM,
            entities=[{"name": "A"}, {"name": "B"}],
            base_importance=0.99,
        )
        assert score <= 1.0

    def test_short_term_penalty(self):
        lt_score = score_memory_importance(
            "Test", MemoryCategory.GENERAL, MemoryType.LONG_TERM, [], 0.5
        )
        st_score = score_memory_importance(
            "Test", MemoryCategory.GENERAL, MemoryType.SHORT_TERM, [], 0.5
        )
        assert lt_score > st_score


# ─────────────────────────────────────────────────────────────
# Memory Domain Model Tests
# ─────────────────────────────────────────────────────────────

class TestMemoryModel:
    def test_memory_creation(self):
        m = Memory(user_id="u1", content="User prefers dark mode")
        assert m.status == MemoryStatus.ACTIVE
        assert m.version == 1
        assert m.memory_id is not None

    def test_mark_updated_increments_version(self):
        m = Memory(user_id="u1", content="Test")
        original_updated = m.updated_at
        m.mark_updated()
        assert m.version == 2

    def test_touch_sets_last_accessed(self):
        m = Memory(user_id="u1", content="Test")
        assert m.last_accessed_at is None
        m.touch()
        assert m.last_accessed_at is not None

    def test_unique_ids(self):
        m1 = Memory(user_id="u1", content="Fact 1")
        m2 = Memory(user_id="u1", content="Fact 2")
        assert m1.memory_id != m2.memory_id
