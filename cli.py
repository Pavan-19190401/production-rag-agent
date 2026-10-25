import argparse
import sys
import time
from typing import Optional

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from app.retrieval.pipeline import get_default_retriever
from app.generation.llm import LLMClient
from app.config import settings


def query_rag(question: str, top_k: int = 5, mode: str = "hybrid") -> None:
    """Executes retrieve-then-generate pipeline and prints response with citations."""
    print("=" * 70)
    print(f"QUERY: {question}")
    print(f"PIPELINE: {mode.upper()} RETRIEVAL")
    print("=" * 70)

    start_time = time.perf_counter()

    # 1. Retrieval
    retriever = get_default_retriever(mode=mode)
    retrieval_start = time.perf_counter()
    chunks = retriever.retrieve(question, top_k=top_k)
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

    print(f"\n[Retrieval] Found {len(chunks)} chunks in {retrieval_ms:.1f}ms")
    for idx, c in enumerate(chunks, start=1):
        preview = c.text[:100].replace("\n", " ") + "..."
        print(f"  {idx}. [{c.source}, page {c.page}] (score: {c.score:.4f}) -> {preview}")

    # 2. Generation
    llm = LLMClient()
    gen_start = time.perf_counter()
    result = llm.generate(question, chunks)
    gen_ms = (time.perf_counter() - gen_start) * 1000
    total_ms = (time.perf_counter() - start_time) * 1000

    # 3. Output
    print("\n" + "-" * 70)
    print("ANSWER:")
    print("-" * 70)
    print(result["answer"])
    print("-" * 70)

    print("\nCITATIONS:")
    if result["citations"]:
        for cit in result["citations"]:
            print(f"  * {cit['source']} (Page {cit['page']}, Chunk: {cit['chunk_id']})")
    else:
        print("  None (Answer refused or no relevant context)")

    print(f"\n[Telemetry] Provider: {result['provider']} | Model: {result['model']} | Total Latency: {total_ms:.1f}ms")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Production RAG CLI Interface")
    parser.add_argument("--question", "-q", type=str, default=None, help="User query to ask the RAG system")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--mode", "-m", type=str, default="hybrid", choices=["dense", "hybrid", "rerank", "rewrite"], help="Retrieval mode")
    args = parser.parse_args()

    if args.question:
        query_rag(args.question, top_k=args.top_k, mode=args.mode)
    else:
        print(f"Production RAG System CLI (mode: {args.mode}). Type 'exit' or 'quit' to stop.")
        while True:
            try:
                q = input("\nEnter Question > ").strip()
                if not q or q.lower() in ("exit", "quit"):
                    break
                query_rag(q, top_k=args.top_k, mode=args.mode)
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
