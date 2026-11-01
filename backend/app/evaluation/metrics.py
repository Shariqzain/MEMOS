"""
MEMOS — Evaluation Metrics Module (Phase 11)
Implements retrieval, extraction, and memory evolution metrics.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set


# ─────────────────────────────────────────────────────────────
# Retrieval Metrics
# ─────────────────────────────────────────────────────────────

def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Recall@K: fraction of relevant items that appear in top-k retrieved."""
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Precision@K: fraction of top-k retrieved items that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def mean_reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """MRR: reciprocal of rank of first relevant item."""
    for i, item in enumerate(retrieved):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(
    retrieved: List[str],
    relevance_map: Dict[str, float],
    k: int,
) -> float:
    """
    nDCG@K with graded relevance.
    relevance_map: {item_id: relevance_score}
    """
    if k == 0:
        return 0.0

    # DCG
    dcg = 0.0
    for i, item in enumerate(retrieved[:k]):
        rel = relevance_map.get(item, 0.0)
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1)=0

    # Ideal DCG
    ideal_rels = sorted(relevance_map.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def f1_score(precision: float, recall: float) -> float:
    """Compute F1 from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


# ─────────────────────────────────────────────────────────────
# Memory Evolution Metrics
# ─────────────────────────────────────────────────────────────

def duplicate_memory_rate(
    total_memories: int,
    duplicate_count: int,
) -> float:
    """Fraction of stored memories that are duplicates."""
    if total_memories == 0:
        return 0.0
    return duplicate_count / total_memories


def update_detection_accuracy(
    true_updates: int,
    detected_updates: int,
    false_updates: int,
) -> Dict[str, float]:
    """Precision and recall for update detection."""
    precision = detected_updates / (detected_updates + false_updates) if (detected_updates + false_updates) > 0 else 0.0
    recall = detected_updates / true_updates if true_updates > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
    }


def conflict_detection_accuracy(
    true_conflicts: int,
    detected_conflicts: int,
    false_conflicts: int,
) -> Dict[str, float]:
    """Precision and recall for conflict detection."""
    precision = detected_conflicts / (detected_conflicts + false_conflicts) if (detected_conflicts + false_conflicts) > 0 else 0.0
    recall = detected_conflicts / true_conflicts if true_conflicts > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
    }


# ─────────────────────────────────────────────────────────────
# Evaluation Report Builder
# ─────────────────────────────────────────────────────────────

class EvaluationReport:
    """Collects evaluation results across multiple test cases."""

    def __init__(self, name: str = "MEMOS Evaluation"):
        self.name = name
        self.results: List[Dict[str, Any]] = []
        self.metrics: Dict[str, List[float]] = {}

    def add_result(
        self,
        test_case_id: str,
        retrieval_mode: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = {
            "test_case_id": test_case_id,
            "retrieval_mode": retrieval_mode,
            "metrics": metrics,
            "metadata": metadata or {},
        }
        self.results.append(entry)

        for key, val in metrics.items():
            self.metrics.setdefault(key, []).append(val)

    def summary(self) -> Dict[str, Any]:
        """Return aggregate metrics."""
        aggregated = {}
        for key, values in self.metrics.items():
            aggregated[key] = {
                "mean": sum(values) / len(values) if values else 0.0,
                "min": min(values) if values else 0.0,
                "max": max(values) if values else 0.0,
                "count": len(values),
            }
        return {
            "name": self.name,
            "total_cases": len(self.results),
            "aggregate_metrics": aggregated,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "summary": self.summary(),
            "results": self.results,
        }
