# Evaluation Report: RERANK_REPORT

**Generated:** 2026-10-25 15:08:36  
**Total Items Evaluated:** 65 (55 answerable, 10 unanswerable)  
**Retrieval Depth (Top-K):** 5

## Summary Metrics Table

| Metric Category | Metric Name | Score | Benchmark Target |
|---|---|---|---|
| **Retrieval** | Recall@5 (Hit Rate) | **61.8%** | ≥ 75.0% |
| **Retrieval** | MRR (Mean Reciprocal Rank) | **0.4915** | ≥ 0.60 |
| **Retrieval** | nDCG@5 | **0.4950** | ≥ 0.65 |
| **Retrieval** | Context Precision | **13.1%** | ≥ 30.0% |
| **Retrieval** | Context Recall | **57.3%** | ≥ 70.0% |
| **Generation** | Faithfulness (Groundedness) | **90.0%** | ≥ 80.0% |
| **Generation** | Answer Relevance | **88.1%** | ≥ 80.0% |
| **Generation** | Answer Correctness (F1) | **28.7%** | ≥ 60.0% |
| **Generation** | Hallucination Rate | **10.0%** | ≤ 20.0% |
| **Guardrails** | Refusal Accuracy (Unanswerable) | **60.0%** | ≥ 90.0% |

## Latency Profile

| Percentile | Latency (ms) |
|---|---|
| **p50 (Median)** | **2920.59 ms** |
| **p95** | **3191.96 ms** |
| **Mean** | **2920.17 ms** |
