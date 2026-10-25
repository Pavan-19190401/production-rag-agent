# Evaluation Report: REWRITE_REPORT

**Generated:** 2026-10-25 15:09:00  
**Total Items Evaluated:** 65 (55 answerable, 10 unanswerable)  
**Retrieval Depth (Top-K):** 5

## Summary Metrics Table

| Metric Category | Metric Name | Score | Benchmark Target |
|---|---|---|---|
| **Retrieval** | Recall@5 (Hit Rate) | **70.9%** | ≥ 75.0% |
| **Retrieval** | MRR (Mean Reciprocal Rank) | **0.5385** | ≥ 0.60 |
| **Retrieval** | nDCG@5 | **0.5585** | ≥ 0.65 |
| **Retrieval** | Context Precision | **15.6%** | ≥ 30.0% |
| **Retrieval** | Context Recall | **67.3%** | ≥ 70.0% |
| **Generation** | Faithfulness (Groundedness) | **90.0%** | ≥ 80.0% |
| **Generation** | Answer Relevance | **88.6%** | ≥ 80.0% |
| **Generation** | Answer Correctness (F1) | **29.1%** | ≥ 60.0% |
| **Generation** | Hallucination Rate | **10.1%** | ≤ 20.0% |
| **Guardrails** | Refusal Accuracy (Unanswerable) | **60.0%** | ≥ 90.0% |

## Latency Profile

| Percentile | Latency (ms) |
|---|---|
| **p50 (Median)** | **28.57 ms** |
| **p95** | **42.3 ms** |
| **Mean** | **26.47 ms** |
