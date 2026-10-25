import concurrent.futures
from typing import List, Dict, Any, Optional
from app.retrieval.dense import DenseRetriever, RetrievedChunk
from app.retrieval.bm25 import BM25Retriever


def reciprocal_rank_fusion(
    ranked_lists: List[List[RetrievedChunk]],
    k: int = 60,
    top_k: int = 5
) -> List[RetrievedChunk]:
    """
    Combines multiple ranked lists using Reciprocal Rank Fusion (RRF):
    RRF_score(d) = sum(1 / (k + rank_m(d)))
    """
    scores: Dict[str, float] = {}
    chunk_store: Dict[str, RetrievedChunk] = {}

    for rank_list in ranked_lists:
        for rank, chunk in enumerate(rank_list, start=1):
            c_id = chunk.chunk_id
            chunk_store[c_id] = chunk
            scores[c_id] = scores.get(c_id, 0.0) + (1.0 / (k + rank))

    sorted_chunk_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)[:top_k]

    fused_results: List[RetrievedChunk] = []
    for cid in sorted_chunk_ids:
        chunk = chunk_store[cid]
        # Store normalized RRF score
        fused_chunk = RetrievedChunk(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            score=scores[cid],
            source=chunk.source,
            page=chunk.page,
            metadata=chunk.metadata
        )
        fused_results.append(fused_chunk)

    return fused_results


class HybridRetriever:
    """
    Hybrid retriever combining Dense vector search with BM25 sparse search
    fused via Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        candidate_pool: int = 30,
        rrf_k: int = 60
    ):
        self.dense = dense_retriever or DenseRetriever()
        self.bm25 = bm25_retriever or BM25Retriever()
        self.candidate_pool = candidate_pool
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """Runs both dense and BM25 retrievers concurrently and fuses them with RRF."""
        if not query.strip():
            return []

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_dense = executor.submit(self.dense.retrieve, query, self.candidate_pool)
            future_bm25 = executor.submit(self.bm25.retrieve, query, self.candidate_pool)
            dense_hits = future_dense.result()
            bm25_hits = future_bm25.result()

        return reciprocal_rank_fusion(
            [dense_hits, bm25_hits],
            k=self.rrf_k,
            top_k=top_k
        )
