from typing import List, Dict, Any, Optional
from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.config import settings
from app.retrieval.dense import RetrievedChunk
from app.retrieval.hybrid import HybridRetriever


class RerankRetriever:
    """
    Two-stage retrieval pipeline:
    Stage 1: Retrieve top-N candidate chunks via Hybrid (BM25 + Dense RRF)
    Stage 2: Rescore top-N candidates down to top-K using a Cross-Encoder (BAAI/bge-reranker-base)
    """

    def __init__(
        self,
        base_retriever: Optional[Any] = None,
        reranker_model: Optional[str] = None,
        candidate_pool: int = 30
    ):
        self.base_retriever = base_retriever or HybridRetriever(candidate_pool=candidate_pool)
        self.candidate_pool = candidate_pool
        model_name = reranker_model or settings.RERANKER_MODEL
        self.reranker = TextCrossEncoder(model_name=model_name)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """Retrieves candidates and rescores them using full cross-attention."""
        if not query.strip():
            return []

        # Stage 1: Fast hybrid candidate retrieval (e.g. top 30)
        candidates = self.base_retriever.retrieve(query, top_k=self.candidate_pool)
        if not candidates:
            return []

        # If candidates are fewer than or equal to top_k, no reranking needed
        if len(candidates) <= 1:
            return candidates[:top_k]

        # Stage 2: Cross-encoder scoring
        # BAAI/bge-reranker processes (query, text) pairs
        doc_texts = [c.text for c in candidates]
        scores = list(self.reranker.rerank(query, doc_texts))

        scored_candidates = []
        for chunk, score in zip(candidates, scores):
            rescore_chunk = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=float(score),
                source=chunk.source,
                page=chunk.page,
                metadata=chunk.metadata
            )
            scored_candidates.append(rescore_chunk)

        # Sort descending by cross-encoder score
        scored_candidates.sort(key=lambda c: c.score, reverse=True)
        return scored_candidates[:top_k]
