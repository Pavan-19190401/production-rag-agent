import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from app.config import settings, BASE_DIR
from app.retrieval.dense import RetrievedChunk

TRACES_DIR = BASE_DIR / "eval" / "traces"
TRACES_FILE = TRACES_DIR / "traces.jsonl"

try:
    from langfuse import Langfuse
    _has_langfuse = True
except ImportError:
    _has_langfuse = False


class RAGTracer:
    """
    Production telemetry and distributed tracing manager.
    Integrates with Langfuse Cloud when credentials are set,
    and maintains an audit-grade local JSONL trace repository for offline analysis.
    """

    def __init__(self):
        self.langfuse = None
        TRACES_DIR.mkdir(parents=True, exist_ok=True)

        if _has_langfuse and settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY:
            try:
                self.langfuse = Langfuse(
                    public_key=settings.LANGFUSE_PUBLIC_KEY,
                    secret_key=settings.LANGFUSE_SECRET_KEY,
                    host=settings.LANGFUSE_HOST
                )
            except Exception as e:
                print(f"Failed to connect to Langfuse: {e}")
                self.langfuse = None

    def create_trace(self, request_id: str, question: str, mode: str = "hybrid") -> Dict[str, Any]:
        """Initializes a trace context."""
        trace_data = {
            "trace_id": request_id,
            "question": question,
            "mode": mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "spans": {},
            "metrics": {},
            "feedback": None
        }

        if self.langfuse:
            try:
                self.langfuse.trace(
                    id=request_id,
                    name="rag_query",
                    input={"question": question, "mode": mode},
                    metadata={"timestamp": trace_data["timestamp"]}
                )
            except Exception:
                pass

        return trace_data

    def record_span_rewrite(self, trace_data: Dict[str, Any], original_q: str, rewritten_qs: List[str], duration_ms: float):
        span_info = {
            "name": "rewrite",
            "input": original_q,
            "output": rewritten_qs,
            "duration_ms": round(duration_ms, 2)
        }
        trace_data["spans"]["rewrite"] = span_info

        if self.langfuse:
            try:
                self.langfuse.span(
                    trace_id=trace_data["trace_id"],
                    name="rewrite",
                    input={"original": original_q},
                    output={"rewritten": rewritten_qs},
                    metadata={"duration_ms": duration_ms}
                )
            except Exception:
                pass

    def record_span_retrieval(self, trace_data: Dict[str, Any], query: str, chunks: List[RetrievedChunk], duration_ms: float):
        chunks_summary = [
            {"chunk_id": c.chunk_id, "source": c.source, "page": c.page, "score": round(c.score, 4)}
            for c in chunks
        ]
        span_info = {
            "name": "retrieval",
            "query": query,
            "chunks_count": len(chunks),
            "retrieved_chunks": chunks_summary,
            "duration_ms": round(duration_ms, 2)
        }
        trace_data["spans"]["retrieval"] = span_info

        if self.langfuse:
            try:
                self.langfuse.span(
                    trace_id=trace_data["trace_id"],
                    name="retrieval",
                    input={"query": query},
                    output={"chunks": chunks_summary},
                    metadata={"duration_ms": duration_ms, "count": len(chunks)}
                )
            except Exception:
                pass

    def record_span_rerank(self, trace_data: Dict[str, Any], candidates_count: int, rescored_chunks: List[RetrievedChunk], duration_ms: float):
        rescored_summary = [
            {"chunk_id": c.chunk_id, "score": round(c.score, 4)}
            for c in rescored_chunks
        ]
        span_info = {
            "name": "rerank",
            "candidates_count": candidates_count,
            "final_count": len(rescored_chunks),
            "rescored_chunks": rescored_summary,
            "duration_ms": round(duration_ms, 2)
        }
        trace_data["spans"]["rerank"] = span_info

        if self.langfuse:
            try:
                self.langfuse.span(
                    trace_id=trace_data["trace_id"],
                    name="rerank",
                    input={"candidates_in": candidates_count},
                    output={"top_chunks": rescored_summary},
                    metadata={"duration_ms": duration_ms}
                )
            except Exception:
                pass

    def record_span_llm(
        self,
        trace_data: Dict[str, Any],
        prompt: str,
        answer: str,
        model: str,
        duration_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float
    ):
        span_info = {
            "name": "generation",
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": round(cost_usd, 6),
            "duration_ms": round(duration_ms, 2)
        }
        trace_data["spans"]["llm"] = span_info

        if self.langfuse:
            try:
                self.langfuse.generation(
                    trace_id=trace_data["trace_id"],
                    name="generation",
                    model=model,
                    input=prompt,
                    output=answer,
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    },
                    metadata={"cost_usd": cost_usd, "duration_ms": duration_ms}
                )
            except Exception:
                pass

    def finalize_trace(
        self,
        trace_data: Dict[str, Any],
        answer: str,
        faithfulness_score: float,
        is_refusal: bool,
        total_latency_ms: float
    ):
        trace_data["answer"] = answer
        trace_data["metrics"]["faithfulness"] = round(faithfulness_score, 4)
        trace_data["metrics"]["is_refusal"] = is_refusal
        trace_data["metrics"]["total_latency_ms"] = round(total_latency_ms, 2)

        # Append to persistent trace store
        try:
            with open(TRACES_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_data, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Error persisting trace: {e}")

        if self.langfuse:
            try:
                self.langfuse.score(
                    trace_id=trace_data["trace_id"],
                    name="faithfulness",
                    value=faithfulness_score
                )
            except Exception:
                pass

    def record_feedback(self, request_id: str, rating: int, comment: Optional[str] = None):
        """Attaches human feedback to the trace in Langfuse and local file."""
        if self.langfuse:
            try:
                self.langfuse.score(
                    trace_id=request_id,
                    name="user_feedback",
                    value=1.0 if rating > 0 else 0.0,
                    comment=comment
                )
            except Exception:
                pass

        # Update local file record if present
        if TRACES_FILE.exists():
            try:
                lines = []
                with open(TRACES_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            if record.get("trace_id") == request_id:
                                record["feedback"] = {
                                    "rating": rating,
                                    "comment": comment,
                                    "timestamp": datetime.now(timezone.utc).isoformat()
                                }
                            lines.append(json.dumps(record, ensure_ascii=False))

                with open(TRACES_FILE, "w", encoding="utf-8") as f:
                    for l in lines:
                        f.write(l + "\n")
            except Exception as e:
                print(f"Error updating feedback in local trace file: {e}")

    def load_recent_traces(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Loads the most recent N traces from storage."""
        if not TRACES_FILE.exists():
            return []
        traces = []
        with open(TRACES_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    traces.append(json.loads(line))
        return traces[-limit:]


# Global Tracer instance
tracer = RAGTracer()
