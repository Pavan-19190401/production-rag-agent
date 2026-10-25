from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """User query request schema."""
    question: str = Field(..., min_length=3, max_length=2000, description="User question to answer")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve")
    mode: str = Field(default="hybrid", description="Retrieval mode: dense, hybrid, rerank, or rewrite")
    session_id: Optional[str] = Field(default=None, description="Optional conversational session ID")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata filter parameters")


class Citation(BaseModel):
    """Represents a citation reference to an indexed source document."""
    source: str = Field(..., description="Filename of source document")
    page: int = Field(..., description="1-indexed page number")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    score: float = Field(..., description="Retrieval relevance score")
    snippet: str = Field(default="", description="Text preview of the cited chunk")


class QueryResponse(BaseModel):
    """Synchronous query response schema."""
    request_id: str
    answer: str
    citations: List[Citation]
    latency_ms: float
    faithfulness_score: float
    is_refusal: bool
    cached: bool


class FeedbackRequest(BaseModel):
    """User feedback schema."""
    request_id: str = Field(..., description="Target request ID")
    rating: int = Field(..., ge=-1, le=1, description="1 for thumbs up, -1 for thumbs down")
    comment: Optional[str] = Field(default=None, max_length=1000, description="Optional user comment")


class FeedbackResponse(BaseModel):
    """Feedback acknowledgment schema."""
    status: str
    message: str


class IngestResponse(BaseModel):
    """Ingestion job summary schema."""
    status: str
    processed_files: int
    chunks_indexed: int
    vector_dim: int


class HealthResponse(BaseModel):
    """Service health and liveness probe schema."""
    status: str
    uptime_seconds: float
    qdrant_status: str
    indexed_chunks: int
    cache_type: str
