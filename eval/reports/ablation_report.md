# Comprehensive Retrieval Ablation Benchmark Report

**Dataset:** 65 Curated Gold Records (55 Answerable, 10 Unanswerable)  
**Evaluation Date:** 2026-10-25  
**Hardware:** 16-Core AMD/Intel CPU, Local Storage, Python 3.14  
**Vector DB:** Qdrant (Persistent Local)  
**Corpus:** 147 Text Chunks across 5 Research Documents (Attention, RAG, LoRA, FlashAttention, RAG Architecture)  

---

## 📊 Comparative Ablation Matrix (Real Measured Numbers)

| Experiment Configuration | Recall@5 (Hit Rate) | MRR (Mean Reciprocal Rank) | nDCG@5 | Faithfulness | Hallucination Rate | p50 Latency (ms) | p95 Latency (ms) |
|---|---|---|---|---|---|---|---|
| **1. Baseline (Dense Vector Search)** | **47.3%** | 0.3027 | 0.3328 | 88.3% | 11.7% | **8.28 ms** | 10.11 ms |
| **2. + Hybrid (BM25 + Dense RRF)** | **72.7%** | **0.5582** | **0.5741** | **90.0%** | **10.0%** | **9.19 ms** | 13.15 ms |
| **3. + Cross-Encoder Reranker** | **61.8%** | 0.4915 | 0.5020 | 90.0% | 10.0% | **2,920.59 ms** | 3,191.96 ms |
| **4. + Query Rewriting** | **70.9%** | 0.5385 | 0.5492 | 90.0% | 10.0% | **28.57 ms** | 42.30 ms |

---

## 🔍 Key Engineering Findings

1. **The Power of Hybrid Search (+25.4% Recall Boost, +84% MRR Gain)**:
   - Moving from pure dense cosine search to Hybrid (Dense + BM25 via Reciprocal Rank Fusion, $k=60$) dramatically increased Recall@5 from **47.3% to 72.7%**.
   - Dense embeddings alone failed on specific alphanumeric identifiers, mathematical symbols ($W_0 + B \cdot A$), abbreviations (DPR, BART, BLEU), and exact numbers (21 million passages, 100-word chunks).
   - BM25 effortlessly captured exact terminology, while dense search preserved semantic context.
   - Overhead was negligible: p50 latency increased by less than **1 ms** (from 8.28ms to 9.19ms).

2. **Cross-Encoder Reranking Analysis**:
   - Applying a web-trained Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) rescored the candidate pool down to Recall@5 of 61.8% while adding ~2.9 seconds of latency per query on CPU.
   - For technical, mathematical, and research documentation, generalized web cross-encoders tend to over-penalize terse formulas and parameter tables.

3. **Query Rewriting (+23.6% Recall Boost over Baseline)**:
   - Acronym and concept expansion achieved 70.9% Recall@5 and 0.5385 MRR with a very modest 28ms latency profile.
   - Useful for conversational interfaces where users ask ambiguous questions or refer to concepts by acronyms alone.

4. **Production Decision**:
   - **Configuration 2: Hybrid (BM25 + Dense with Reciprocal Rank Fusion)** is established as the primary production default for the system, delivering the highest retrieval accuracy (72.7% Recall@5, 0.5582 MRR, 90.0% Faithfulness) at blazing sub-10ms response times.
