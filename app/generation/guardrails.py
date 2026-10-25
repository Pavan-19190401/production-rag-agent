import re
from typing import Tuple

FAITHFULNESS_THRESHOLD = 0.40

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for",
    "of", "and", "or", "that", "this", "it", "by", "as", "from", "with", "be",
    "i", "don't", "know", "based", "provided", "documents", "page", "source"
}


def evaluate_faithfulness(answer: str, context: str) -> float:
    """
    Computes claim-level groundedness score of the generated answer against the retrieved context.
    Returns a score between 0.0 and 1.0.
    """
    ans_tokens = [w for w in re.findall(r"\w+", answer.lower()) if w not in STOPWORDS and len(w) > 2]
    if not ans_tokens:
        return 1.0

    ctx_tokens = set(re.findall(r"\w+", context.lower()))
    supported = sum(1 for t in ans_tokens if t in ctx_tokens)
    return supported / len(ans_tokens)


def enforce_faithfulness_guardrail(
    answer: str,
    context: str,
    threshold: float = FAITHFULNESS_THRESHOLD
) -> Tuple[str, float, bool]:
    """
    Evaluates faithfulness. If answer asserts ungrounded claims below threshold,
    it strictly overrides the answer with the refusal directive.
    Returns: (final_answer, faithfulness_score, is_refusal)
    """
    # If answer is already a direct refusal, pass it through
    if "I don't know based on the provided documents" in answer:
        return answer, 1.0, True

    score = evaluate_faithfulness(answer, context)

    if score < threshold:
        # Refuse hallucinated / unsupported answer
        guarded_answer = "I don't know based on the provided documents."
        return guarded_answer, score, True

    return answer, score, False
