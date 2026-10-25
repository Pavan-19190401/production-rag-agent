import re
from typing import Dict, Any, Literal
from app.config import settings

RouteType = Literal["direct", "calculator", "agent", "rag"]


class QueryRouter:
    """
    Classifies incoming user intent to optimize latency and compute:
    1. 'direct': Greetings, capabilities, or meta-questions -> 0ms direct reply
    2. 'calculator': Pure arithmetic / numerical computation -> calculator tool
    3. 'agent': Multi-step questions requiring retrieval + math computation
    4. 'rag': Domain question answering over documents -> standard Hybrid RAG
    """

    GREETINGS = {
        "hi", "hello", "hey", "who are you", "what are you", "what can you do",
        "help", "thanks", "thank you", "good morning", "good evening"
    }

    def route(self, query: str) -> RouteType:
        clean_q = query.strip().lower()
        clean_q_no_punct = re.sub(r"[^\w\s]", "", clean_q)
        words = clean_q_no_punct.split()

        # 1. Direct conversational check
        if clean_q_no_punct in self.GREETINGS or any(w in {"hi", "hello", "hey", "thanks", "thank"} for w in words):
            return "direct"
        if "what can you do" in clean_q_no_punct or "who are you" in clean_q_no_punct:
            return "direct"

        # 2. Pure arithmetic check (e.g. "512 / 8", "sqrt(64) * 2")
        if re.match(r"^[\d\s\+\-\*\/\^\(\)\.\%eE]+$", clean_q.replace("what is", "").replace("calculate", "").strip()):
            return "calculator"

        # 3. Multi-step Agent check (e.g. combining retrieval with calculation/ratios)
        has_math_intent = any(w in clean_q for w in ["calculate", "ratio", "how many times", "percentage", "difference between bandwidth", "speedup ratio"])
        has_domain_intent = any(w in clean_q for w in ["flashattention", "sram", "hbm", "lora", "gpt-3", "d_model", "bleu", "parameters"])

        if has_math_intent and has_domain_intent:
            return "agent"

        # 4. Default: Standard Hybrid RAG
        return "rag"
