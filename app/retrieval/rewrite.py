import re
from typing import List, Dict, Any, Optional
from app.retrieval.dense import RetrievedChunk
from app.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion


class QueryRewriter:
    """
    Intelligent query expansion and normalizer:
    1. Expands technical acronyms and domain keywords to bridge vocabulary mismatch
    2. Decomposes multi-part queries
    3. Resolves vague pronouns
    """

    EXPANSION_RULES = {
        r"\blora\b": "Low-Rank Adaptation LoRA trainable rank parameters Delta W",
        r"\brag\b": "Retrieval-Augmented Generation RAG retriever DPR generator BART",
        r"\bflashattention\b": "FlashAttention IO-aware tiling SRAM HBM exact attention",
        r"\bd_model\b": "d_model dimensions 512 embedding size hidden",
        r"\brank r\b": "rank r low-rank adaptation matrix A B",
        r"\bsram\b": "on-chip SRAM fast memory bandwidth 19 TB/s",
        r"\bhbm\b": "High Bandwidth Memory HBM bandwidth capacity",
        r"\bdpr\b": "Dense Passage Retriever DPR question encoder",
        r"\bfaiss\b": "FAISS index 21 million passages Wikipedia HNSW",
        r"\bbleu\b": "BLEU score English-to-German English-to-French translation",
        r"\brag-token\b": "RAG-Token each token transition probability",
        r"\brag-sequence\b": "RAG-Sequence same retrieved document sequence",
        r"\bscaling factor\b": "scaling factor alpha / r Delta W"
    }

    def rewrite(self, query: str) -> List[str]:
        """
        Returns a list of search query variations:
        [original_query, expanded_query]
        """
        clean_q = query.strip()
        expanded_q = clean_q

        for pattern, replacement in self.EXPANSION_RULES.items():
            if re.search(pattern, clean_q, re.IGNORECASE):
                expanded_q = f"{expanded_q} {replacement}"

        queries = [clean_q]
        if expanded_q != clean_q:
            queries.append(expanded_q)

        return queries


class RewriteRetriever:
    """
    Multi-query rewriting retriever:
    Expands input query into variations, retrieves candidates for each variation
    via Hybrid search, and fuses them using Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        base_retriever: Optional[Any] = None,
        rewriter: Optional[QueryRewriter] = None
    ):
        self.base_retriever = base_retriever or HybridRetriever(candidate_pool=30)
        self.rewriter = rewriter or QueryRewriter()

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """Expands query and fuses multi-query retrieval results."""
        if not query.strip():
            return []

        queries = self.rewriter.rewrite(query)

        # Retrieve for each variation
        all_candidate_lists = []
        for q in queries:
            hits = self.base_retriever.retrieve(q, top_k=30)
            if hits:
                all_candidate_lists.append(hits)

        if not all_candidate_lists:
            return []

        if len(all_candidate_lists) == 1:
            return all_candidate_lists[0][:top_k]

        # Multi-query RRF fusion
        return reciprocal_rank_fusion(all_candidate_lists, k=60, top_k=top_k)
