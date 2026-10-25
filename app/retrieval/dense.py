from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from app.config import settings
from ingest.ingest import get_qdrant_client


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    source: str
    page: int
    metadata: Dict[str, Any]


class DenseRetriever:
    """Performs cosine vector search against Qdrant using BGE embeddings."""

    def __init__(self, client: Optional[QdrantClient] = None):
        self.client = client or get_qdrant_client()
        self.embed_model = TextEmbedding(model_name=settings.EMBEDDING_MODEL)

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[RetrievedChunk]:
        """Embeds query and returns top-k nearest neighbor chunks."""
        if not query.strip():
            return []

        # Embed query vector
        query_vectors = list(self.embed_model.embed([query]))
        query_vector = query_vectors[0].tolist()

        try:
            # Check if collection exists
            collections = [c.name for c in self.client.get_collections().collections]
            if settings.QDRANT_COLLECTION not in collections:
                return []

            # In modern qdrant-client: search or query_points
            if hasattr(self.client, "search"):
                search_results = self.client.search(
                    collection_name=settings.QDRANT_COLLECTION,
                    query_vector=query_vector,
                    limit=top_k,
                    score_threshold=score_threshold if score_threshold > 0 else None
                )
            else:
                res = self.client.query_points(
                    collection_name=settings.QDRANT_COLLECTION,
                    query=query_vector,
                    limit=top_k
                )
                search_results = res.points

            results: List[RetrievedChunk] = []
            for hit in search_results:
                payload = hit.payload or {}
                chunk = RetrievedChunk(
                    chunk_id=payload.get("chunk_id", str(hit.id)),
                    text=payload.get("text", ""),
                    score=float(hit.score),
                    source=payload.get("source", "unknown"),
                    page=int(payload.get("page", 1)),
                    metadata=payload
                )
                results.append(chunk)

            return results
        except Exception as e:
            print(f"Dense retrieval error: {e}")
            return []
