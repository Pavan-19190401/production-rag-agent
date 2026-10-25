import os
import sys
import json
import time
import math
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from app.config import settings, BASE_DIR
from app.retrieval.dense import DenseRetriever
from app.generation.llm import LLMClient

DEFAULT_DATASET_PATH = BASE_DIR / "eval" / "golden_dataset.jsonl"
REPORTS_DIR = BASE_DIR / "eval" / "reports"


def compute_token_f1(pred: str, target: str) -> float:
    """Computes token-level precision, recall, and F1 between two strings."""
    pred_tokens = re.findall(r"\w+", pred.lower())
    target_tokens = re.findall(r"\w+", target.lower())

    if not pred_tokens or not target_tokens:
        return 1.0 if pred_tokens == target_tokens else 0.0

    common = set(pred_tokens) & set(target_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(set(pred_tokens))
    recall = len(common) / len(set(target_tokens))
    return 2 * (precision * recall) / (precision + recall)


def compute_faithfulness(answer: str, context: str) -> float:
    """
    Computes groundedness score of the generated answer against the retrieved context.
    Checks what proportion of non-stopword tokens in the answer appear in the context.
    """
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for",
        "of", "and", "or", "that", "this", "it", "by", "as", "from", "with", "be",
        "i", "don't", "know", "based", "provided", "documents", "page", "source"
    }
    ans_tokens = [w for w in re.findall(r"\w+", answer.lower()) if w not in stopwords and len(w) > 2]
    if not ans_tokens:
        return 1.0

    ctx_tokens = set(re.findall(r"\w+", context.lower()))
    supported = sum(1 for t in ans_tokens if t in ctx_tokens)
    return supported / len(ans_tokens)


def compute_answer_relevance(answer: str, question: str) -> float:
    """Computes relevance of answer to the original question."""
    q_words = set(re.findall(r"\w+", question.lower())) - {"what", "is", "how", "why", "the", "in", "of", "to"}
    ans_words = set(re.findall(r"\w+", answer.lower()))
    if not q_words or not ans_words:
        return 1.0
    overlap = len(q_words & ans_words)
    return min(1.0, overlap / max(1, len(q_words) * 0.5))


def run_evaluation(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    top_k: int = 5,
    subset: Optional[int] = None,
    report_name: str = "baseline_report",
    min_recall: Optional[float] = None,
    min_faithfulness: Optional[float] = None,
    retriever_override: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Runs comprehensive RAG evaluation over golden dataset.
    Calculates:
      - Retrieval: Recall@k, MRR, nDCG@k, Context Precision, Context Recall
      - Generation: Faithfulness, Answer Relevance, Answer Correctness (F1), Hallucination Rate
      - Guardrails: Refusal Accuracy on unanswerable questions
      - Latency: p50, p95, mean latency in ms
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Golden dataset not found at {dataset_path}")

    records: List[Dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if subset and subset < len(records):
        records = records[:subset]

    print(f"\n{'='*75}")
    print(f"RUNNING RAG EVALUATION BENCHMARK: {report_name.upper()}")
    print(f"Dataset: {len(records)} items | Top-K: {top_k}")
    print(f"{'='*75}\n")

    retriever = retriever_override or DenseRetriever()
    llm = LLMClient()

    results_detail: List[Dict[str, Any]] = []
    latencies: List[float] = []

    hits_at_k = 0
    reciprocal_ranks = []
    ndcg_scores = []
    context_precisions = []
    context_recalls = []

    faithfulness_scores = []
    relevance_scores = []
    correctness_scores = []

    unanswerable_total = 0
    unanswerable_correct_refusals = 0
    answerable_total = 0

    for idx, item in enumerate(records, start=1):
        q_id = item["id"]
        question = item["question"]
        ref_answer = item["reference_answer"]
        gold_chunks = set(item.get("source_chunk_ids", []))
        is_answerable = item.get("answerable", True)

        t_start = time.perf_counter()

        # 1. Retrieval
        retrieved_chunks = retriever.retrieve(question, top_k=top_k)
        retrieved_ids = [c.chunk_id for c in retrieved_chunks]
        context_str = " ".join([c.text for c in retrieved_chunks])

        # 2. Generation
        gen_output = llm.generate(question, retrieved_chunks)
        answer = gen_output["answer"]

        latency_ms = (time.perf_counter() - t_start) * 1000
        latencies.append(latency_ms)

        # 3. Retrieval Metrics (for answerable questions)
        is_hit = False
        first_rank = 0
        dcg = 0.0

        if is_answerable and gold_chunks:
            answerable_total += 1
            for rank_idx, r_id in enumerate(retrieved_ids, start=1):
                if r_id in gold_chunks:
                    if not is_hit:
                        is_hit = True
                        first_rank = rank_idx
                    dcg += 1.0 / math.log2(rank_idx + 1)

            if is_hit:
                hits_at_k += 1
                reciprocal_ranks.append(1.0 / first_rank)
            else:
                reciprocal_ranks.append(0.0)

            # Ideal DCG
            idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(gold_chunks), top_k) + 1))
            ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

            # Precision & Recall
            relevant_retrieved = len(set(retrieved_ids) & gold_chunks)
            context_precisions.append(relevant_retrieved / max(1, len(retrieved_ids)))
            context_recalls.append(relevant_retrieved / len(gold_chunks))

        # 4. Guardrail Refusal Check (for unanswerable questions)
        if not is_answerable:
            unanswerable_total += 1
            is_refused = "I don't know based on the provided documents" in answer
            if is_refused:
                unanswerable_correct_refusals += 1

        # 5. Generation Metrics
        faith = compute_faithfulness(answer, context_str)
        rel = compute_answer_relevance(answer, question)
        corr = compute_token_f1(answer, ref_answer)

        if is_answerable:
            faithfulness_scores.append(faith)
            relevance_scores.append(rel)
            correctness_scores.append(corr)

        status_flag = "[HIT]" if is_hit else ("[REFUSED]" if not is_answerable and "I don't know" in answer else "[MISS]")
        print(f"[{idx:02d}/{len(records):02d}] {status_flag:9s} {q_id}: {question[:50]}... ({latency_ms:.1f}ms)")

        results_detail.append({
            "id": q_id,
            "question": question,
            "answerable": is_answerable,
            "retrieved_chunk_ids": retrieved_ids,
            "gold_chunk_ids": list(gold_chunks),
            "hit_at_k": is_hit,
            "first_rank": first_rank,
            "answer": answer,
            "faithfulness": faith,
            "relevance": rel,
            "correctness": corr,
            "latency_ms": latency_ms
        })

    # Summary Statistics
    latencies.sort()
    p50_latency = latencies[len(latencies) // 2] if latencies else 0.0
    p95_index = int(len(latencies) * 0.95)
    p95_latency = latencies[p95_index] if latencies else 0.0
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0

    recall_at_k = hits_at_k / max(1, answerable_total)
    mean_mrr = sum(reciprocal_ranks) / max(1, len(reciprocal_ranks))
    mean_ndcg = sum(ndcg_scores) / max(1, len(ndcg_scores))
    mean_ctx_precision = sum(context_precisions) / max(1, len(context_precisions))
    mean_ctx_recall = sum(context_recalls) / max(1, len(context_recalls))

    mean_faithfulness = sum(faithfulness_scores) / max(1, len(faithfulness_scores))
    mean_relevance = sum(relevance_scores) / max(1, len(relevance_scores))
    mean_correctness = sum(correctness_scores) / max(1, len(correctness_scores))
    hallucination_rate = max(0.0, 1.0 - mean_faithfulness)
    refusal_acc = unanswerable_correct_refusals / max(1, unanswerable_total)

    summary = {
        "benchmark_name": report_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_evaluated": len(records),
        "answerable_count": answerable_total,
        "unanswerable_count": unanswerable_total,
        "top_k": top_k,
        "metrics": {
            "recall_at_k": round(recall_at_k, 4),
            "mrr": round(mean_mrr, 4),
            "ndcg_at_k": round(mean_ndcg, 4),
            "context_precision": round(mean_ctx_precision, 4),
            "context_recall": round(mean_ctx_recall, 4),
            "faithfulness": round(mean_faithfulness, 4),
            "answer_relevance": round(mean_relevance, 4),
            "answer_correctness": round(mean_correctness, 4),
            "hallucination_rate": round(hallucination_rate, 4),
            "refusal_accuracy": round(refusal_acc, 4)
        },
        "latency_ms": {
            "p50": round(p50_latency, 2),
            "p95": round(p95_latency, 2),
            "mean": round(mean_latency, 2)
        }
    }

    # Save Reports
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / f"{report_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results_detail}, f, indent=2, ensure_ascii=False)

    md_path = REPORTS_DIR / f"{report_name}.md"
    markdown_content = f"""# Evaluation Report: {report_name.upper()}

**Generated:** {summary['timestamp']}  
**Total Items Evaluated:** {summary['total_evaluated']} ({summary['answerable_count']} answerable, {summary['unanswerable_count']} unanswerable)  
**Retrieval Depth (Top-K):** {top_k}

## Summary Metrics Table

| Metric Category | Metric Name | Score | Benchmark Target |
|---|---|---|---|
| **Retrieval** | Recall@{top_k} (Hit Rate) | **{summary['metrics']['recall_at_k'] * 100:.1f}%** | ≥ 75.0% |
| **Retrieval** | MRR (Mean Reciprocal Rank) | **{summary['metrics']['mrr']:.4f}** | ≥ 0.60 |
| **Retrieval** | nDCG@{top_k} | **{summary['metrics']['ndcg_at_k']:.4f}** | ≥ 0.65 |
| **Retrieval** | Context Precision | **{summary['metrics']['context_precision'] * 100:.1f}%** | ≥ 30.0% |
| **Retrieval** | Context Recall | **{summary['metrics']['context_recall'] * 100:.1f}%** | ≥ 70.0% |
| **Generation** | Faithfulness (Groundedness) | **{summary['metrics']['faithfulness'] * 100:.1f}%** | ≥ 80.0% |
| **Generation** | Answer Relevance | **{summary['metrics']['answer_relevance'] * 100:.1f}%** | ≥ 80.0% |
| **Generation** | Answer Correctness (F1) | **{summary['metrics']['answer_correctness'] * 100:.1f}%** | ≥ 60.0% |
| **Generation** | Hallucination Rate | **{summary['metrics']['hallucination_rate'] * 100:.1f}%** | ≤ 20.0% |
| **Guardrails** | Refusal Accuracy (Unanswerable) | **{summary['metrics']['refusal_accuracy'] * 100:.1f}%** | ≥ 90.0% |

## Latency Profile

| Percentile | Latency (ms) |
|---|---|
| **p50 (Median)** | **{summary['latency_ms']['p50']} ms** |
| **p95** | **{summary['latency_ms']['p95']} ms** |
| **Mean** | **{summary['latency_ms']['mean']} ms** |
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    print(f"\n{'='*75}")
    print(f"EVALUATION COMPLETE")
    print(f"Recall@{top_k}: {summary['metrics']['recall_at_k']*100:.1f}% | MRR: {summary['metrics']['mrr']:.4f}")
    print(f"Faithfulness: {summary['metrics']['faithfulness']*100:.1f}% | Refusal Accuracy: {summary['metrics']['refusal_accuracy']*100:.1f}%")
    print(f"p50 Latency: {summary['latency_ms']['p50']}ms | p95 Latency: {summary['latency_ms']['p95']}ms")
    print(f"Report saved to: {md_path}")
    print(f"{'='*75}\n")

    # CI/CD Regression Gate Enforcement
    if min_recall and summary['metrics']['recall_at_k'] < min_recall:
        print(f"[FAIL] Recall@{top_k} ({summary['metrics']['recall_at_k']}) below threshold {min_recall}")
        sys.exit(1)
    if min_faithfulness and summary['metrics']['faithfulness'] < min_faithfulness:
        print(f"[FAIL] Faithfulness ({summary['metrics']['faithfulness']}) below threshold {min_faithfulness}")
        sys.exit(1)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Production RAG Evaluation Runner")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET_PATH), help="Path to golden dataset")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K retrieval depth")
    parser.add_argument("--subset", type=int, default=None, help="Subset of dataset to evaluate")
    parser.add_argument("--report-name", type=str, default="baseline_report", help="Report filename prefix")
    parser.add_argument("--retriever", type=str, default="dense", choices=["dense", "hybrid", "rerank", "rewrite"], help="Retriever configuration to evaluate")
    parser.add_argument("--min-recall", type=float, default=None, help="CI threshold for recall@k")
    parser.add_argument("--min-faithfulness", type=float, default=None, help="CI threshold for faithfulness")
    args = parser.parse_args()

    # Dynamic retriever selection
    retriever_instance = None
    if args.retriever == "hybrid":
        from app.retrieval.hybrid import HybridRetriever
        retriever_instance = HybridRetriever()
    elif args.retriever == "rerank":
        from app.retrieval.rerank import RerankRetriever
        retriever_instance = RerankRetriever()
    elif args.retriever == "rewrite":
        from app.retrieval.rewrite import RewriteRetriever
        retriever_instance = RewriteRetriever()
    else:
        from app.retrieval.dense import DenseRetriever
        retriever_instance = DenseRetriever()

    run_evaluation(
        dataset_path=Path(args.dataset),
        top_k=args.top_k,
        subset=args.subset,
        report_name=args.report_name,
        min_recall=args.min_recall,
        min_faithfulness=args.min_faithfulness,
        retriever_override=retriever_instance
    )


if __name__ == "__main__":
    main()
