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
from app.tracer import tracer
from app.agents.router import QueryRouter
from app.agents.react_agent import ReActAgent
from app.agents.tools import CalculatorTool
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

    # 3. Initialize Distributed Trace
    trace = tracer.create_trace(request_id, req.question, mode=req.mode)

    # 4. Mode Determination & Dynamic Agent Routing
    mode = req.mode.lower()
    effective_mode = mode
    if mode == "auto":
        router = QueryRouter()
        effective_mode = router.route(req.question)

    if effective_mode == "direct":
        async def direct_stream():
            yield "event: sources\ndata: []\n\n"
            greeting = (
                "Hello! I am your production research assistant grounded strictly on arXiv papers "
                "and technical documentation (Attention, RAG, LoRA, FlashAttention). "
                "Ask me a technical question, calculate a parameter ratio, or select an example below!"
            )
            words = greeting.split(" ")
            for i, w in enumerate(words):
                yield f"data: {json.dumps({'token': w + (' ' if i < len(words) - 1 else '')}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)
            yield "event: done\ndata: [DONE]\n\n"
            latency_ms = (time.perf_counter() - t_start) * 1000
            tracer.finalize_trace(trace, greeting, 1.0, False, latency_ms)
            telemetry.record_query(latency_ms, cached=False, is_refusal=False)

        return StreamingResponse(direct_stream(), media_type="text/event-stream")

    elif effective_mode == "calculator":
        calc = CalculatorTool()
        res_calc = calc.execute(req.question)

        async def calc_stream():
            yield f"event: thought\ndata: {json.dumps({'iteration': 1, 'thought': 'Evaluating arithmetic expression directly using AST calculator.', 'tool': 'calculator', 'tool_input': req.question}, ensure_ascii=False)}\n\n"
            yield f"event: observation\ndata: {json.dumps({'iteration': 1, 'tool': 'calculator', 'content': str(res_calc)}, ensure_ascii=False)}\n\n"
            yield "event: sources\ndata: []\n\n"
            calc_answer = f"Calculated Result: {res_calc}"
            words = calc_answer.split(" ")
            for i, w in enumerate(words):
                yield f"data: {json.dumps({'token': w + (' ' if i < len(words) - 1 else '')}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)
            yield "event: done\ndata: [DONE]\n\n"
            latency_ms = (time.perf_counter() - t_start) * 1000
            tracer.finalize_trace(trace, calc_answer, 1.0, False, latency_ms)
            telemetry.record_query(latency_ms, cached=False, is_refusal=False)

        return StreamingResponse(calc_stream(), media_type="text/event-stream")

    elif effective_mode == "agent":
        agent = ReActAgent()

        async def agent_stream():
            accumulated_answer = ""
            agent_citations = []
            async for item in agent.astream_run(req.question):
                itype = item.get("type")
                if itype == "thought":
                    yield f"event: thought\ndata: {json.dumps(item, ensure_ascii=False)}\n\n"
                elif itype == "observation":
                    yield f"event: observation\ndata: {json.dumps(item, ensure_ascii=False)}\n\n"
                elif itype == "sources":
                    agent_citations = item.get("citations", [])
                    yield f"event: sources\ndata: {json.dumps(agent_citations, ensure_ascii=False)}\n\n"
                elif itype == "token":
                    accumulated_answer += item.get("token", "")
                    yield f"data: {json.dumps({'token': item.get('token', '')}, ensure_ascii=False)}\n\n"
                elif itype == "done":
                    latency_ms = (time.perf_counter() - t_start) * 1000
                    tracer.finalize_trace(trace, accumulated_answer, 1.0, False, latency_ms)
                    telemetry.record_query(latency_ms, cached=False, is_refusal=False)
                    cache.set(
                        req.question,
                        {
                            "answer": accumulated_answer,
                            "citations": agent_citations,
                            "faithfulness": 1.0,
                            "is_refusal": False
                        },
                        mode=req.mode,
                        ttl=3600
                    )
                    yield "event: done\ndata: [DONE]\n\n"

        return StreamingResponse(agent_stream(), media_type="text/event-stream")

    # 5. Standard RAG Retrieval Pipeline (hybrid, dense, rerank, rewrite)
    retrieval_mode = "hybrid" if effective_mode not in {"dense", "rerank", "rewrite"} else effective_mode
    retrieval_start = time.perf_counter()
    retriever = get_default_retriever(mode=retrieval_mode)
    chunks = retriever.retrieve(req.question, top_k=req.top_k)
    if req.filters and "sources" in req.filters and req.filters["sources"]:
        allowed_sources = set(req.filters["sources"])
        chunks = [c for c in chunks if c.source in allowed_sources]
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
    tracer.record_span_retrieval(trace, req.question, chunks, retrieval_ms)

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

    # 5. SSE Stream Generator
    async def sse_event_stream() -> AsyncIterator[str]:
        # Step A: Stream sources event immediately upon retrieval
        yield f"event: sources\ndata: {json.dumps(citations, ensure_ascii=False)}\n\n"

        gen_start = time.perf_counter()
        accumulated_answer = ""
        # Step B: Stream LLM tokens asynchronously
        async for token in llm.astream(req.question, chunks):
            accumulated_answer += token
            yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"

        gen_ms = (time.perf_counter() - gen_start) * 1000

        # Step C: Post-Generation Faithfulness Guardrail Check
        final_answer, faith_score, is_refusal = enforce_faithfulness_guardrail(
            accumulated_answer, context_str
        )

        prompt_tokens = len(context_str) // 4
        comp_tokens = len(final_answer) // 4
        cost_usd = (prompt_tokens * 0.00000015) + (comp_tokens * 0.00000060)

        # Record LLM Span and Finalize Trace
        tracer.record_span_llm(
            trace,
            prompt=context_str[:500],
            answer=final_answer,
            model=settings.LLM_MODEL,
            duration_ms=gen_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=comp_tokens,
            cost_usd=cost_usd
        )

        latency_ms = (time.perf_counter() - t_start) * 1000
        tracer.finalize_trace(trace, final_answer, faith_score, is_refusal, latency_ms)
        telemetry.record_query(latency_ms, cached=False, is_refusal=is_refusal)
        budget_manager.record_usage(prompt_tokens=prompt_tokens, completion_tokens=comp_tokens, cost_usd=cost_usd)

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
    tracer.record_feedback(req.request_id, req.rating, req.comment)
    logger.info(
        f"User feedback for request_id={req.request_id}: rating={req.rating} comment={req.comment or ''}",
        extra={"extra_data": {"request_id": req.request_id, "rating": req.rating, "comment": req.comment}}
    )
    return FeedbackResponse(status="success", message="Feedback recorded. Thank you!")


@app.get(
    "/documents",
    summary="List All Indexed Documents & Metadata"
)
async def documents_endpoint():
    """Returns the catalog of indexed documents with chunk and page metrics."""
    chunks_path = BASE_DIR / "data" / "processed" / "chunks.json"
    if not chunks_path.exists():
        return {"documents": []}
    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        doc_stats = {}
        for c in chunks:
            src = c.get("source", "unknown")
            if src not in doc_stats:
                doc_stats[src] = {
                    "name": src,
                    "chunks": 0,
                    "pages": set(),
                    "type": src.split(".")[-1].lower() if "." in src else "txt",
                    "title": src.replace("_", " ").replace(".pdf", "").replace(".md", "").replace(".html", "").title()
                }
            doc_stats[src]["chunks"] += 1
            doc_stats[src]["pages"].add(c.get("page", 1))

        docs = [
            {
                "name": v["name"],
                "chunks": v["chunks"],
                "pages": len(v["pages"]),
                "type": v["type"],
                "title": v["title"]
            }
            for v in doc_stats.values()
        ]
        return {"documents": docs}
    except Exception as e:
        return {"documents": [], "error": str(e)}


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
