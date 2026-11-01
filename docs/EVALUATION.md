# MEMOS — Evaluation Methodology & Benchmark Results

---

## 1. Overview

Evaluation of memory in conversational AI requires measuring three interrelated capabilities:
1. **Retrieval Efficacy**: Are the relevant historical memories retrieved when answering user queries?
2. **Evolution Integrity**: Does the system update facts correctly when preferences change, without creating contradictory or duplicate entries?
3. **Response Quality**: Does the injected memory produce accurate, personalized responses while handling unanswerable questions gracefully?

---

## 2. Evaluation Scenarios & Dataset

MEMOS ships with a curated benchmark (`datasets/samples/memos_eval_dataset.json`) consisting of **12 distinct scenarios** modeled after research challenges from LongMemEval and LoCoMo:

| Scenario ID | Category | Challenge Tested |
|---|---|---|
| `stable_preference_01` | Preference | Persisting long-term user preference across sessions |
| `project_info_01` | Project | Retaining multi-faceted project status & architecture |
| `preference_update_01` | Update | Explicit user correction (superseding old preference) |
| `temporal_fact_01` | Temporal | Time-bound fact evaluation |
| `contradiction_01` | Conflict | Resolving contradictory statements |
| `scoped_preference_01` | Context Scope | Differentiating global vs. domain-specific preference |
| `irrelevant_01` | Noise Rejection | Refusing to memorize small-talk or transient greetings |
| `relationship_01` | Multi-hop | Graph traversal over entity relationships |
| `unanswerable_01` | Groundedness | Acknowledging missing information instead of hallucinating |
| `duplicate_01` | Deduplication | Re-affirming existing knowledge without duplicate records |
| `goal_01` | Goal | High-priority goal retention & recall |
| `obsolete_01` | Deprecation | Archiving expired or superseded information |

---

## 3. Metrics Formulation

### 3.1 Retrieval Metrics
- **Recall@K**: Fraction of relevant ground-truth memory IDs present within the top-$K$ retrieved candidates:
  $$\text{Recall}@K = \frac{|\text{Retrieved}_{1..K} \cap \text{Relevant}|}{|\text{Relevant}|}$$
- **Precision@K**: Fraction of top-$K$ candidates that are relevant:
  $$\text{Precision}@K = \frac{|\text{Retrieved}_{1..K} \cap \text{Relevant}|}{K}$$
- **Mean Reciprocal Rank (MRR)**: Evaluates where the first relevant item appears in ranking:
  $$\text{MRR} = \frac{1}{\text{rank}_1}$$
- **Normalized Discounted Cumulative Gain (nDCG@K)**: Measures graded ranking relevance:
  $$\text{DCG}@K = \sum_{i=1}^K \frac{\text{rel}_i}{\log_2(i+1)}, \quad \text{nDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$

### 3.2 Evolution Metrics
- **Duplicate Memory Rate**:
  $$\text{DMR} = \frac{\text{Duplicate Memory Count}}{\text{Total Stored Memories}}$$
- **Update Detection F1**: Harmonic mean of Precision and Recall for identifying when a new utterance supersedes an existing memory.
- **Conflict Resolution Accuracy**: Accuracy of flagging contradictory statements versus updating existing entries.

---

## 4. Benchmark Execution & Results

Run the automated evaluation suite via:
```bash
python backend/scripts/run_evaluation.py
```

### Empirical Results Summary

```
=================================================================
           MEMOS EVALUATION BENCHMARK RESULTS
=================================================================
Total Scenarios Evaluated: 12
Total Queries Evaluated:   13
Ground Truth Memories:    12
Ground Truth Updates:     3
Ground Truth Conflicts:   1
-----------------------------------------------------------------
Metric               Mean         Min        Max       
-----------------------------------------------------------------
recall@1             1.0000       1.0000     1.0000    
recall@3             1.0000       1.0000     1.0000    
recall@5             1.0000       1.0000     1.0000    
precision@1          1.0000       1.0000     1.0000    
precision@3          0.3333       0.3333     0.3333    
mrr                  1.0000       1.0000     1.0000    
ndcg@3               1.0000       1.0000     1.0000    
=================================================================
```

---

## 5. External Benchmark Integration

MEMOS provides native loaders for two standardized benchmarks in `app/evaluation/dataset_loader.py`:

1. **LoCoMo** (Long-Context Conversational Memory):
   - Multi-session chat conversations with distant entity recall.
   - Run via: `python scripts/run_evaluation.py --dataset path/to/locomo.json`
2. **LongMemEval**:
   - Long-term memory evaluation spanning cross-domain updates.
   - Run via: `python scripts/run_evaluation.py --dataset path/to/longmemeval.jsonl`
