# Evaluation Report: BASELINE_REPORT

**Generated:** 2026-10-25 14:56:53  
**Total Items Evaluated:** 65 (55 answerable, 10 unanswerable)  
**Retrieval Depth (Top-K):** 5

## Summary Metrics Table

| Metric Category | Metric Name | Score | Benchmark Target |
|---|---|---|---|
| **Retrieval** | Recall@5 (Hit Rate) | **47.3%** | ≥ 75.0% |
| **Retrieval** | MRR (Mean Reciprocal Rank) | **0.3027** | ≥ 0.60 |
| **Retrieval** | nDCG@5 | **0.3328** | ≥ 0.65 |
| **Retrieval** | Context Precision | **10.2%** | ≥ 30.0% |
| **Retrieval** | Context Recall | **44.5%** | ≥ 70.0% |
| **Generation** | Faithfulness (Groundedness) | **88.3%** | ≥ 80.0% |
| **Generation** | Answer Relevance | **83.3%** | ≥ 80.0% |
| **Generation** | Answer Correctness (F1) | **28.3%** | ≥ 60.0% |
| **Generation** | Hallucination Rate | **11.7%** | ≤ 20.0% |
| **Guardrails** | Refusal Accuracy (Unanswerable) | **60.0%** | ≥ 90.0% |

## Latency Profile

| Percentile | Latency (ms) |
|---|---|
| **p50 (Median)** | **8.28 ms** |
| **p95** | **10.11 ms** |
| **Mean** | **8.26 ms** |
