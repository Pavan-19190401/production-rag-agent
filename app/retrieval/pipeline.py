from typing import List, Dict, Any, Optional
from app.config import settings
from app.retrieval.dense import DenseRetriever, RetrievedChunk
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.rerank import RerankRetriever
from app.retrieval.rewrite import RewriteRetriever


def get_default_retriever(mode: str = "hybrid") -> Any:
    """
    Factory function returning the configured production retriever.
    Based on empirical Phase 3 ablation benchmarks:
    - 'hybrid' (BM25 + Dense RRF) is the best overall performer:
      Recall@5: 72.7%, MRR: 0.5582, p50 Latency: 9.2ms.
    """
    mode = mode.lower()
    if mode == "dense":
        return DenseRetriever()
    elif mode == "rerank":
        return RerankRetriever()
    elif mode == "rewrite":
        return RewriteRetriever()
    else:  # Default to best empirical configuration: Hybrid RRF
        return HybridRetriever(candidate_pool=30, rrf_k=60)
