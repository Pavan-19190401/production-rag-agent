# Evaluation Report: BASELINE_REPORT

**Generated:** 2026-10-25 15:40:36  
**Total Items Evaluated:** 20 (20 answerable, 0 unanswerable)  
**Retrieval Depth (Top-K):** 5

## Summary Metrics Table

| Metric Category | Metric Name | Score | Benchmark Target |
|---|---|---|---|
| **Retrieval** | Recall@5 (Hit Rate) | **100.0%** | ≥ 75.0% |
| **Retrieval** | MRR (Mean Reciprocal Rank) | **0.8667** | ≥ 0.60 |
| **Retrieval** | nDCG@5 | **0.8701** | ≥ 0.65 |
| **Retrieval** | Context Precision | **22.0%** | ≥ 30.0% |
| **Retrieval** | Context Recall | **97.5%** | ≥ 70.0% |
| **Generation** | Faithfulness (Groundedness) | **91.5%** | ≥ 80.0% |
| **Generation** | Answer Relevance | **93.4%** | ≥ 80.0% |
| **Generation** | Answer Correctness (F1) | **40.6%** | ≥ 60.0% |
| **Generation** | Hallucination Rate | **8.5%** | ≤ 20.0% |
| **Guardrails** | Refusal Accuracy (Unanswerable) | **0.0%** | ≥ 90.0% |

## Latency Profile

| Percentile | Latency (ms) |
|---|---|
| **p50 (Median)** | **7.32 ms** |
| **p95** | **10.45 ms** |
| **Mean** | **7.65 ms** |
