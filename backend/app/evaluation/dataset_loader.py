"""
MEMOS — Dataset Loaders (Phase 11)
Supports custom MEMOS dataset, LoCoMo-style, and LongMemEval-style formats.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Dataset Record Schemas
# ─────────────────────────────────────────────────────────────

class ConversationTurn(BaseModel):
    role: str  # user / assistant
    content: str
    turn_id: Optional[int] = None


class MemoryAnnotation(BaseModel):
    """Ground-truth memory to be extracted or retrieved."""
    content: str
    category: str = "general"
    entities: List[str] = Field(default_factory=list)
    is_update: bool = False
    is_contradiction: bool = False
    expected_action: str = "create"  # create | update | supersede | ignore


class EvalQuestion(BaseModel):
    """A question used to evaluate retrieval / response quality."""
    question: str
    expected_answer: str
    relevant_memory_ids: List[str] = Field(default_factory=list)
    question_type: str = "factual"  # factual | preference | temporal | relationship | unanswerable


class DatasetRecord(BaseModel):
    """One evaluation scenario."""
    record_id: str
    description: str = ""
    conversation: List[ConversationTurn] = Field(default_factory=list)
    memory_annotations: List[MemoryAnnotation] = Field(default_factory=list)
    eval_questions: List[EvalQuestion] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# Dataset Loader
# ─────────────────────────────────────────────────────────────

class DatasetLoader:
    """Load and validate evaluation datasets from JSON/JSONL files."""

    def __init__(self, base_path: str = "datasets"):
        self.base_path = Path(base_path)

    def load_json(self, filename: str) -> List[DatasetRecord]:
        """Load a JSON file containing a list of dataset records."""
        path = self.base_path / filename
        if not path.exists():
            logger.warning("Dataset file not found: %s", path)
            return []

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                records = []
                for i, item in enumerate(data):
                    try:
                        records.append(DatasetRecord(**item))
                    except Exception as e:
                        logger.warning("Malformed record at index %d: %s", i, e)
                return records
            else:
                logger.warning("Expected list in %s, got %s", filename, type(data).__name__)
                return []
        except json.JSONDecodeError as e:
            logger.error("JSON parse error in %s: %s", filename, e)
            return []

    def load_jsonl(self, filename: str) -> List[DatasetRecord]:
        """Load a JSONL file (one record per line)."""
        path = self.base_path / filename
        if not path.exists():
            logger.warning("Dataset file not found: %s", path)
            return []

        records = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    records.append(DatasetRecord(**item))
                except Exception as e:
                    logger.warning("Malformed line %d: %s", line_num, e)

        logger.info("Loaded %d records from %s", len(records), filename)
        return records

    def load_smoke_test(self) -> List[DatasetRecord]:
        """Load the built-in smoke test dataset."""
        return self.load_json("samples/memos_eval_dataset.json")

    def get_available_datasets(self) -> List[str]:
        """List available dataset files."""
        if not self.base_path.exists():
            return []
        files = []
        for ext in ("*.json", "*.jsonl"):
            files.extend(str(f.relative_to(self.base_path)) for f in self.base_path.rglob(ext))
        return files
