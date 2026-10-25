import os
import sys
import json
import time
import uuid
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncIterator

# Ensure UTF-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from fastapi import FastAPI, Request, HTTPException, Security, Depends, status, UploadFile, File
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import settings, BASE_DIR
from app.schemas import (
    QueryRequest, QueryResponse, Citation,
    FeedbackRequest, FeedbackResponse,
    IngestResponse, HealthResponse
)
from app.retrieval.pipeline import get_default_retriever
from app.generation.llm import LLMClient
from app.generation.guardrails import enforce_faithfulness_guardrail
from app.cache import cache
from app.budget import budget_manager, BudgetExceededError
from app.observability import logger, telemetry
from ingest.ingest import run_ingestion, get_qdrant_client

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Production-Grade RAG & LLM Agent System",
    description="Enterprise RAG service featuring Hybrid Retrieval (BM25 + Dense RRF), Cross-Encoder Reranking, Strict Citations, and SSE Streaming.",
    version="1.0.0"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key Authentication
API_KEY_HEADER = APIKeyHeader(name="x-api-key", auto_error=False)


async def verify_api_key(
    request: Request,
    api_key_header: Optional[str] = Security(API_KEY_HEADER)
) -> str:
    """Validates the client x-api-key header unless exempted."""
    exempt_paths = {"/health", "/metrics", "/docs", "/openapi.json", "/redoc"}
    if request.url.path in exempt_paths:
        return ""

    expected_key = settings.API_KEY
    # If no expected key is configured or default secret, allow for local dev
    if not expected_key:
        return "dev-key"

    if not api_key_header or api_key_header != expected_key:
        logger.warning(
            f"Unauthorized access attempt from {request.client.host if request.client else 'unknown'}",
            extra={"extra_data": {"path": request.url.path}}
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing 'x-api-key' header."
        )
    return api_key_header


# Request ID & Logging Middleware
@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000

    response.headers["x-request-id"] = request_id
    response.headers["x-latency-ms"] = f"{duration_ms:.2f}"

    logger.info(
        f"{request.method} {request.url.path} status={response.status_code} latency={duration_ms:.2f}ms",
        extra={"extra_data": {"request_id": request_id, "status": response.status_code, "latency_ms": duration_ms}}
    )
    return response


# ------------------------------------------------------------------------------
# Core Endpoints
# ------------------------------------------------------------------------------

@app.post(
    "/query",
    summary="Query RAG System with Real-Time SSE Streaming",
    response_description="Server-Sent Events text stream or JSON response"
)
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def query_endpoint(
    req: QueryRequest,
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    Submits a user query to the RAG pipeline.
    Streams Server-Sent Events (SSE):
    1. 'event: sources' -> yields JSON list of retrieved chunk citations
    2. 'data: {"token": "..."}' -> yields generated tokens sequentially
    3. 'event: done' -> signals completion
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    t_start = time.perf_counter()

    # 1. Budget Verification
    try:
        budget_manager.check_budget()
    except BudgetExceededError as be:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(be))

    # 2. Check Cache
    cached_result = cache.get(req.question, mode=req.mode)
    if cached_result:
        latency_ms = (time.perf_counter() - t_start) * 1000
        telemetry.record_query(latency_ms, cached=True, is_refusal=cached_result.get("is_refusal", False))

        async def cached_stream():
            yield f"event: sources\ndata: {json.dumps(cached_result['citations'], ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'token': cached_result['answer'], 'cached': True}, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: [DONE]\n\n"

        return StreamingResponse(cached_stream(), media_type="text/event-stream")

    # 3. Retrieve Chunks
    retriever = get_default_retriever(mode=req.mode)
    chunks = retriever.retrieve(req.question, top_k=req.top_k)

    citations = [
        Citation(
            source=c.source,
            page=c.page,
            chunk_id=c.chunk_id,
            score=round(c.score, 4),
            snippet=c.text[:180].replace("\n", " ") + "..."
        ).model_dump()
        for c in chunks
    ]

    llm = LLMClient()
    context_str = " ".join([c.text for c in chunks])

    # 4. SSE Stream Generator
    async def sse_event_stream() -> AsyncIterator[str]:
        # Step A: Stream sources event immediately upon retrieval
        yield f"event: sources\ndata: {json.dumps(citations, ensure_ascii=False)}\n\n"

        accumulated_answer = ""
        # Step B: Stream LLM tokens asynchronously
        async for token in llm.astream(req.question, chunks):
            accumulated_answer += token
            yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"

        # Step C: Post-Generation Faithfulness Guardrail Check
        final_answer, faith_score, is_refusal = enforce_faithfulness_guardrail(
            accumulated_answer, context_str
        )

        latency_ms = (time.perf_counter() - t_start) * 1000
        telemetry.record_query(latency_ms, cached=False, is_refusal=is_refusal)
        budget_manager.record_usage(prompt_tokens=len(context_str) // 4, completion_tokens=len(final_answer) // 4)

        # Cache result for future requests
        cache.set(
            req.question,
            {
                "answer": final_answer,
                "citations": citations,
                "faithfulness": faith_score,
                "is_refusal": is_refusal
            },
            mode=req.mode,
            ttl=3600
        )

        # Step D: Completion signal
        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(sse_event_stream(), media_type="text/event-stream")


@app.post(
    "/ingest",
    summary="Upload & Index New Documents",
    response_model=IngestResponse
)
async def ingest_endpoint(
    file: Optional[UploadFile] = File(None),
    api_key: str = Depends(verify_api_key)
):
    """Uploads and ingests a document or triggers an idempotent directory rescan."""
    raw_dir = Path(settings.DATA_RAW_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if file:
        file_path = raw_dir / file.filename
        content = await file.read()
        file_path.write_bytes(content)
        logger.info(f"Saved uploaded document: {file.filename}")

    # Run ingestion job
    result = run_ingestion()
    return IngestResponse(
        status=result.get("status", "unknown"),
        processed_files=result.get("processed_files", 0),
        chunks_indexed=result.get("chunks_indexed", 0),
        vector_dim=result.get("vector_dim", 384)
    )


@app.post(
    "/feedback",
    summary="Submit User Feedback on an Answer",
    response_model=FeedbackResponse
)
async def feedback_endpoint(
    req: FeedbackRequest,
    api_key: str = Depends(verify_api_key)
):
    """Records thumbs-up (+1) or thumbs-down (-1) rating tied to a request_id."""
    telemetry.record_feedback(req.rating)
    logger.info(
        f"User feedback for request_id={req.request_id}: rating={req.rating} comment={req.comment or ''}",
        extra={"extra_data": {"request_id": req.request_id, "rating": req.rating, "comment": req.comment}}
    )
    return FeedbackResponse(status="success", message="Feedback recorded. Thank you!")


@app.get(
    "/health",
    summary="Health & Readiness Check",
    response_model=HealthResponse
)
async def health_endpoint():
    """Liveness probe verifying vector store connection, indexed chunk count, and cache."""
    qdrant_ok = "healthy"
    indexed_count = 0
    try:
        client = get_qdrant_client()
        coll_info = client.get_collection(settings.QDRANT_COLLECTION)
        indexed_count = coll_info.points_count or 0
    except Exception as e:
        qdrant_ok = f"degraded: {e}"

    return HealthResponse(
        status="ok" if "healthy" in qdrant_ok else "degraded",
        uptime_seconds=telemetry.get_metrics()["uptime_seconds"],
        qdrant_status=qdrant_ok,
        indexed_chunks=indexed_count,
        cache_type=cache.cache_type
    )


@app.get(
    "/metrics",
    summary="Telemetry and Cost Metrics"
)
async def metrics_endpoint():
    """Returns query telemetry, cache performance, and budget utilization."""
    return {
        "telemetry": telemetry.get_metrics(),
        "budget": budget_manager.get_stats(),
        "settings": {
            "embedding_model": settings.EMBEDDING_MODEL,
            "reranker_model": settings.RERANKER_MODEL,
            "llm_model": settings.LLM_MODEL,
            "rate_limit_per_minute": settings.RATE_LIMIT_PER_MINUTE
        }
    }
