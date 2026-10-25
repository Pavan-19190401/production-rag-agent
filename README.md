# Production-Grade RAG / LLM Agent System

[![CI/CD Pipeline](https://github.com/placeholder/rag-system/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/placeholder/rag-system/actions)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-dc2626.svg?logo=qdrant)](https://qdrant.tech)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end, enterprise-ready Retrieval-Augmented Generation (RAG) system with hybrid retrieval (Dense + BM25 via Reciprocal Rank Fusion), cross-encoder reranking, query rewriting, real-time Server-Sent Events (SSE) streaming, hallucination guardrails, and automated evaluation regression testing.

---

## 🏛️ Architecture Overview

```mermaid
flowchart TD
    subgraph Ingestion["Offline Ingestion Pipeline"]
        RawDocs["Documents (PDF, MD, HTML)"] --> Clean["Deduplicate (SHA-256) & Clean"]
        Clean --> Chunk["Structure-Aware / Recursive Chunking"]
        Chunk --> Embed["BGE Embeddings (BAAI/bge-small-en-v1.5)"]
        Chunk --> BM25["BM25 Inverted Index"]
        Embed --> Qdrant[("Qdrant Vector Database")]
    end

    subgraph Serving["Online Serving Pipeline (FastAPI)"]
        User["Client / Next.js Web UI"] --> Gateway["FastAPI (Auth + Rate Limiter)"]
        Gateway --> Cache{"Redis / Memory Cache"}
        Cache -- Hit --> SSE["Stream Answer Directly"]
        Cache -- Miss --> Rewrite["Query Expansion & Rewriter"]
        Rewrite --> DenseRet["Dense Search (Qdrant)"]
        Rewrite --> SparseRet["Sparse Search (BM25)"]
        DenseRet & SparseRet --> RRF["Reciprocal Rank Fusion (RRF)"]
        RRF --> Rerank["Cross-Encoder Reranker (BGE-Reranker)"]
        Rerank --> Context["Top-5 Context + Citations"]
        Context --> LLM["LLM Grounded Generation"]
        LLM --> Guardrail{"Faithfulness Check"}
        Guardrail -- Valid --> SSE
        Guardrail -- Hallucinated --> Refusal["'I don't know based on provided docs'"]
    end

    subgraph Observability["Observability & Tracing"]
        Gateway --> Langfuse["Langfuse Tracing (Latency, Tokens, Cost)"]
        SSE --> Feedback["/feedback Endpoint"]
        Feedback --> Langfuse
    end
```

---

## 🚀 Quickstart

### Prerequisites
- Python 3.11+
- Git
- (Optional) Docker & Docker Compose

### 1. Clone & Set Up Virtual Environment
```bash
git clone <repo-url>
cd RAG
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
```bash
cp .env.example .env
# Edit .env with your OPENAI_API_KEY or GEMINI_API_KEY
```

### 3. Run Ingestion CLI
```bash
python -m ingest.ingest --chunk-size 512 --overlap 50
```

### 4. Query from CLI
```bash
python cli.py --question "What is the primary formula for Scaled Dot-Product Attention?"
```

---

## 📊 Evaluation & Ablation Results

*Evaluated on a 65-item curated golden benchmark (55 answerable, 10 unanswerable) across 5 technical research documents.*

| Configuration | Recall@5 | MRR | Faithfulness | Hallucination Rate | p50 Latency (ms) |
|---|---|---|---|---|---|
| **1. Dense Vector Search (Baseline)** | **47.3%** | 0.3027 | 88.3% | 11.7% | **8.28 ms** |
| **2. + Hybrid (BM25 + Dense RRF)** *(Default)* | **72.7%** | **0.5582** | **90.0%** | **10.0%** | **9.19 ms** |
| **3. + Cross-Encoder Reranker** | **61.8%** | 0.4915 | 90.0% | 10.0% | **2,920.59 ms** |
| **4. + Query Rewriting** | **70.9%** | 0.5385 | 90.0% | 10.0% | **28.57 ms** |

> **Production Takeaway**: Adding BM25 sparse search and Reciprocal Rank Fusion ($k=60$) yielded a **+25.4% jump in Recall@5** and an **84% improvement in MRR** over pure vector search, with sub-10ms median latency.
