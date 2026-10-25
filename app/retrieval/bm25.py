import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

from app.config import settings, BASE_DIR
from app.retrieval.dense import RetrievedChunk

CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.json"


def tokenize_text(text: str) -> List[str]:
    """Tokenizes text for BM25 indexing (alphanumeric lowercase tokens)."""
    return [token.lower() for token in re.findall(r"\w+", text) if len(token) > 1]


class BM25Retriever:
    """Sparse keyword retriever powered by BM25Okapi."""

    def __init__(self, chunks_path: Path = CHUNKS_PATH):
        self.chunks_path = chunks_path
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._load_and_index()

    def _load_and_index(self) -> None:
        if not self.chunks_path.exists():
            print(f"Warning: Chunks path {self.chunks_path} does not exist.")
            return

        with open(self.chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        if not self.chunks:
            return

        tokenized_corpus = [tokenize_text(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 30) -> List[RetrievedChunk]:
        """Performs BM25 keyword search and returns top-k ranked chunks."""
        if not query.strip() or not self.bm25 or not self.chunks:
            return []

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        scored_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results: List[RetrievedChunk] = []
        for idx in scored_indices:
            score = float(scores[idx])
            if score <= 0.0:
                continue

            chunk_data = self.chunks[idx]
            results.append(RetrievedChunk(
                chunk_id=chunk_data["chunk_id"],
                text=chunk_data["text"],
                score=score,
                source=chunk_data.get("source", "doc"),
                page=int(chunk_data.get("page", 1)),
                metadata=chunk_data
            ))

        return results
