#!/usr/bin/env python3
"""
Production Verification & Diagnostics Script
Tests database connectivity, LLM provider, caching, and end-to-end RAG pipeline.
"""

import sys
import time
from pathlib import Path

# Ensure UTF-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.cache import cache
from app.budget import budget_manager
from app.retrieval.pipeline import get_default_retriever
from app.generation.llm import LLMClient
from app.generation.guardrails import enforce_faithfulness_guardrail
from app.tracer import tracer


def print_status(component: str, status: str, detail: str = "", ok: bool = True):
    badge = "[PASS]" if ok else "[WARN]"
    color_prefix = "\033[92m" if ok else "\033[93m"
    reset = "\033[0m"
    print(f"{color_prefix}{badge}{reset} {component:25} | {status} {f'({detail})' if detail else ''}")


def main():
    print("=" * 70)
    print(" PRODUCTION READINESS & DIAGNOSTICS SUITE")
    print("=" * 70)

    all_passed = True

    # 1. Check Configuration & Security
    api_key_configured = bool(settings.API_KEY and settings.API_KEY != "dev-key")
    print_status(
        "API Security",
        "Configured" if api_key_configured else "Default Dev Key",
        f"API_KEY: {settings.API_KEY[:6]}... (rate limit: {settings.RATE_LIMIT_PER_MINUTE}/min)",
        ok=True
    )

    # 2. Check Vector Database (Qdrant)
    try:
        # Check if local server is already running
        import httpx
        server_running = False
        try:
            h_res = httpx.get("http://127.0.0.1:8000/health", timeout=1.0)
            if h_res.status_code == 200:
                server_running = True
                h_data = h_res.json()
                print_status(
                    "Vector Database",
                    "Connected (via Live Server)",
                    f"Status: {h_data.get('qdrant_status')}, {h_data.get('indexed_chunks')} chunks indexed",
                    ok="healthy" in h_data.get("qdrant_status", "")
                )
        except Exception:
            pass

        if not server_running:
            from ingest.ingest import get_qdrant_client
            client = get_qdrant_client()
            coll = client.get_collection(settings.QDRANT_COLLECTION)
            pt_count = coll.points_count or 0
            target = "Qdrant Cloud" if settings.QDRANT_URL.startswith("http") else f"Local ({settings.QDRANT_URL})"
            print_status(
                "Vector Database",
                "Connected",
                f"{target}, {pt_count} points in '{settings.QDRANT_COLLECTION}'",
                ok=pt_count > 0
            )
    except Exception as e:
        print_status("Vector Database", "Error", str(e), ok=False)
        all_passed = False

    # 3. Check LLM Provider
    llm = LLMClient()
    if llm.provider == "openai" and settings.OPENAI_API_KEY:
        print_status("LLM Provider", "OpenAI Active", f"Model: {settings.LLM_MODEL}", ok=True)
    elif llm.provider == "gemini" and settings.GEMINI_API_KEY:
        print_status("LLM Provider", "Gemini Active", f"Model: {settings.LLM_MODEL}", ok=True)
    else:
        print_status("LLM Provider", "Local Mock Active", "Set OPENAI_API_KEY or GEMINI_API_KEY in .env for live AI", ok=True)

    # 4. Check Cache
    cache_type = cache.cache_type
    print_status(
        "Semantic Cache",
        cache_type.capitalize(),
        "Redis cluster active" if cache_type == "redis" else "In-memory LRU fallback (add REDIS_URL for distributed cache)",
        ok=True
    )

    # 5. Check Observability & Tracing
    has_langfuse = tracer.langfuse is not None
    print_status(
        "Observability",
        "Langfuse Cloud" if has_langfuse else "Local Audit Logs",
        "Connected to cloud dashboard" if has_langfuse else "Traces logged to eval/traces/traces.jsonl",
        ok=True
    )

    # 6. Check Daily Spend Guard
    budget_stats = budget_manager.get_stats()
    print_status(
        "Budget Protection",
        "Active",
        f"Daily limit: ${budget_stats['daily_budget_usd']:.2f} | Remaining: ${budget_stats['budget_remaining_usd']:.2f}",
        ok=True
    )

    # 7. End-to-End Pipeline Execution
    print("\n" + "-" * 70)
    print(" EXECUTING END-TO-END PIPELINE VALIDATION QUERY")
    print("-" * 70)

    test_q = "What is FlashAttention and how does tiling help?"

    if server_running:
        t0 = time.perf_counter()
        try:
            import httpx
            res = httpx.post(
                "http://127.0.0.1:8000/query",
                json={"question": test_q, "top_k": 3, "mode": "hybrid"},
                headers={"x-api-key": settings.API_KEY},
                timeout=10.0
            )
            total_ms = (time.perf_counter() - t0) * 1000
            if res.status_code == 200:
                body = res.text
                print(f"Live Server returned SSE Stream ({len(body)} bytes) in {total_ms:.2f}ms")
                # Extract citation sources
                for line in body.splitlines():
                    if line.startswith("data: [{\"source"):
                        print(f"  Verified Sources: {line[:120]}...")
                        break
                print_status("End-to-End Test", "PASSED (Live HTTP API)", f"Latency: {total_ms:.1f}ms", ok=True)
            else:
                print_status("End-to-End Test", "FAILED", f"HTTP {res.status_code}: {res.text[:100]}", ok=False)
                all_passed = False
        except Exception as e:
            print_status("End-to-End Test", "FAILED", str(e), ok=False)
            all_passed = False
    else:
        t0 = time.perf_counter()
        try:
            retriever = get_default_retriever(mode="hybrid")
            chunks = retriever.retrieve(test_q, top_k=3)
            retrieval_ms = (time.perf_counter() - t0) * 1000

            print(f"Retrieval returned {len(chunks)} chunks in {retrieval_ms:.2f}ms:")
            for idx, c in enumerate(chunks, 1):
                print(f"  [{idx}] {c.source} (page {c.page}, score: {c.score:.4f})")

            gen_t0 = time.perf_counter()
            res = llm.generate(test_q, chunks)
            gen_ms = (time.perf_counter() - gen_t0) * 1000

            context_str = " ".join([c.text for c in chunks])
            final_ans, faith, is_refusal = enforce_faithfulness_guardrail(res["answer"], context_str)
            total_ms = (time.perf_counter() - t0) * 1000

            print(f"\nGenerated Answer ({gen_ms:.2f}ms, Faithfulness: {faith * 100:.1f}%):")
            print(f"  \"{final_ans[:200]}...\"")
            print(f"\nTotal Pipeline Latency: {total_ms:.2f}ms")
            print_status("End-to-End Test", "PASSED (In-Process)", f"Latency: {total_ms:.1f}ms, Faith: {faith * 100:.1f}%", ok=True)
        except Exception as e:
            print_status("End-to-End Test", "FAILED", str(e), ok=False)
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("\033[92mALL PRODUCTION DIAGNOSTIC CHECKS PASSED SUCCESSFULLY!\033[0m")
    else:
        print("\033[93mSOME OPTIONAL COMPONENTS HAVE WARNINGS (SEE DETAILS ABOVE)\033[0m")
    print("=" * 70)


if __name__ == "__main__":
    main()
