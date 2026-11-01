# MEMOS Evaluation Datasets

## Directory Structure

- `samples/memos_eval_dataset.json` — Custom MEMOS benchmark (12 scenarios)

## Dataset Format

All datasets use a common schema. Each record contains:

```json
{
  "record_id": "unique_id",
  "description": "Scenario description",
  "conversation": [
    {"role": "user", "content": "...", "turn_id": 1},
    {"role": "assistant", "content": "...", "turn_id": 2}
  ],
  "memory_annotations": [
    {
      "content": "expected extracted memory",
      "category": "preference",
      "entities": ["Entity1"],
      "is_update": false,
      "is_contradiction": false,
      "expected_action": "create"
    }
  ],
  "eval_questions": [
    {
      "question": "Retrieval question",
      "expected_answer": "Expected answer",
      "question_type": "factual"
    }
  ]
}
```

## External Datasets

### LoCoMo
- Paper: Long-Context Conversational Memory
- Use for: temporal facts, entity recall, multi-turn memory
- Format: Convert to MEMOS DatasetRecord schema using dataset_loader.py

### LongMemEval
- Paper: Long-Term Memory Evaluation
- Use for: multi-session recall, temporal reasoning
- Format: Convert to MEMOS DatasetRecord schema using dataset_loader.py

Place external dataset files in this directory and use the DatasetLoader to load them.

## Scenario Types

1. **Stable preferences** — Facts that should persist
2. **Explicit updates** — User corrects or changes a fact
3. **Contradictions** — Conflicting statements requiring resolution
4. **Scoped preferences** — Different facts for different contexts
5. **Temporal facts** — Time-sensitive information
6. **Relationships** — Multi-hop entity relationships
7. **Irrelevant content** — Should NOT create memories
8. **Duplicates** — Repeated facts that should be deduplicated
9. **Unanswerable questions** — Should acknowledge missing information
10. **Obsolete memories** — Facts that become outdated
