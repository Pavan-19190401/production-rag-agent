# Evaluation Report: HYBRID_REPORT

**Generated:** 2026-10-25 15:01:08  
**Total Items Evaluated:** 65 (55 answerable, 10 unanswerable)  
**Retrieval Depth (Top-K):** 5

## Summary Metrics Table

| Metric Category | Metric Name | Score | Benchmark Target |
|---|---|---|---|
| **Retrieval** | Recall@5 (Hit Rate) | **72.7%** | ≥ 75.0% |
| **Retrieval** | MRR (Mean Reciprocal Rank) | **0.5582** | ≥ 0.60 |
| **Retrieval** | nDCG@5 | **0.5810** | ≥ 0.65 |
| **Retrieval** | Context Precision | **16.0%** | ≥ 30.0% |
| **Retrieval** | Context Recall | **70.9%** | ≥ 70.0% |
| **Generation** | Faithfulness (Groundedness) | **90.0%** | ≥ 80.0% |
| **Generation** | Answer Relevance | **89.7%** | ≥ 80.0% |
| **Generation** | Answer Correctness (F1) | **29.1%** | ≥ 60.0% |
| **Generation** | Hallucination Rate | **10.0%** | ≤ 20.0% |
| **Guardrails** | Refusal Accuracy (Unanswerable) | **60.0%** | ≥ 90.0% |

## Latency Profile

| Percentile | Latency (ms) |
|---|---|
| **p50 (Median)** | **9.19 ms** |
| **p95** | **13.15 ms** |
| **Mean** | **9.64 ms** |
