import os
import re
from typing import List, Dict, Any, Optional, AsyncIterator
import httpx
from app.config import settings
from app.generation.prompts import GROUNDING_SYSTEM_PROMPT, build_grounded_user_prompt, format_context
from app.retrieval.dense import RetrievedChunk

try:
    from openai import OpenAI, AsyncOpenAI
    _has_openai = True
except ImportError:
    _has_openai = False


class LLMClient:
    """
    Unified LLM Client with grounding enforcement, fallback model support,
    and multi-provider dispatch (OpenAI, Gemini, Local Mock).
    """

    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        self.openai_client = None
        self.async_openai_client = None

        if self.provider == "auto":
            if settings.OPENAI_API_KEY:
                self.provider = "openai"
            elif settings.GEMINI_API_KEY:
                self.provider = "gemini"
            else:
                self.provider = "mock"

        if self.provider == "openai":
            if _has_openai and settings.OPENAI_API_KEY:
                self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
                self.async_openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            else:
                self.provider = "mock"
        elif self.provider == "gemini" and not settings.GEMINI_API_KEY:
            self.provider = "mock"

    def generate(self, question: str, chunks: List[RetrievedChunk]) -> Dict[str, Any]:
        """
        Synchronously generates a grounded answer given question and retrieved chunks.
        Returns: { 'answer': str, 'citations': List[Dict], 'provider': str, 'model': str }
        """
        # If no context was retrieved, immediately refuse without calling LLM
        if not chunks:
            return {
                "answer": "I don't know based on the provided documents.",
                "citations": [],
                "provider": self.provider,
                "model": "none",
                "refusal": True
            }

        context = format_context(chunks)
        user_prompt = build_grounded_user_prompt(question, context)

        citations = [
            {"source": c.source, "page": c.page, "chunk_id": c.chunk_id, "score": c.score}
            for c in chunks
        ]

        if self.provider == "openai" and self.openai_client:
            answer = self._generate_openai(user_prompt)
        elif self.provider == "gemini" and settings.GEMINI_API_KEY:
            answer = self._generate_gemini(user_prompt)
        else:
            answer = self._generate_mock(question, chunks)

        return {
            "answer": answer,
            "citations": citations,
            "provider": self.provider,
            "model": settings.LLM_MODEL if self.provider != "mock" else "deterministic-extractive",
            "refusal": "I don't know based on the provided documents" in answer
        }

    async def astream(self, question: str, chunks: List[RetrievedChunk]) -> AsyncIterator[str]:
        """Streams response tokens asynchronously."""
        if not chunks:
            yield "I don't know based on the provided documents."
            return

        context = format_context(chunks)
        user_prompt = build_grounded_user_prompt(question, context)

        if self.provider == "openai" and self.async_openai_client:
            try:
                response = await self.async_openai_client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=settings.MAX_TOKENS_PER_QUERY,
                    stream=True
                )
                async for chunk in response:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        yield delta
                return
            except Exception as e:
                # Fallback to secondary model if primary fails
                print(f"Primary model stream failed: {e}. Trying fallback...")
                try:
                    fb_response = await self.async_openai_client.chat.completions.create(
                        model=settings.FALLBACK_LLM_MODEL,
                        messages=[
                            {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=settings.LLM_TEMPERATURE,
                        max_tokens=settings.MAX_TOKENS_PER_QUERY,
                        stream=True
                    )
                    async for chunk in fb_response:
                        delta = chunk.choices[0].delta.content or ""
                        if delta:
                            yield delta
                    return
                except Exception as fb_err:
                    yield f"Error generating response: {fb_err}"
                    return

        elif self.provider == "gemini" and settings.GEMINI_API_KEY:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.LLM_MODEL}:streamGenerateContent?alt=sse&key={settings.GEMINI_API_KEY}"
                payload = {
                    "contents": [
                        {
                            "parts": [
                                {"text": f"{GROUNDING_SYSTEM_PROMPT}\n\n{user_prompt}".strip()}
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": settings.LLM_TEMPERATURE,
                        "maxOutputTokens": settings.MAX_TOKENS_PER_QUERY
                    }
                }
                import json
                async with httpx.AsyncClient(timeout=30.0) as client:
                    async with client.stream("POST", url, json=payload) as stream_res:
                        stream_res.raise_for_status()
                        async for line in stream_res.aiter_lines():
                            if line.startswith("data:"):
                                data_str = line[5:].strip()
                                if not data_str:
                                    continue
                                try:
                                    parsed = json.loads(data_str)
                                    candidates = parsed.get("candidates", [])
                                    if candidates:
                                        parts = candidates[0].get("content", {}).get("parts", [])
                                        if parts and "text" in parts[0]:
                                            yield parts[0]["text"]
                                except Exception:
                                    pass
                return
            except Exception as e:
                pass

        # Fallback synchronous generator yielding chunks
        res = self.generate(question, chunks)
        # Yield words with small pause
        words = res["answer"].split(" ")
        for i, w in enumerate(words):
            yield w + (" " if i < len(words) - 1 else "")

    def _generate_openai(self, user_prompt: str) -> str:
        """Invokes OpenAI Chat Completion API with fallback."""
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.MAX_TOKENS_PER_QUERY
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI error with {settings.LLM_MODEL}: {e}. Retrying fallback {settings.FALLBACK_LLM_MODEL}...")
            try:
                fb = self.openai_client.chat.completions.create(
                    model=settings.FALLBACK_LLM_MODEL,
                    messages=[
                        {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=settings.MAX_TOKENS_PER_QUERY
                )
                return fb.choices[0].message.content.strip()
            except Exception as e2:
                return f"LLM Generation Error: {e2}"

    def _generate_gemini(self, user_prompt: str) -> str:
        """Invokes Google Gemini REST API."""
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.LLM_MODEL}:generateContent?key={settings.GEMINI_API_KEY}"
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": f"{GROUNDING_SYSTEM_PROMPT}\n\n{user_prompt}"}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": settings.LLM_TEMPERATURE,
                    "maxOutputTokens": settings.MAX_TOKENS_PER_QUERY
                }
            }
            with httpx.Client(timeout=30.0) as client:
                res = client.post(url, json=payload)
                res.raise_for_status()
                data = res.json()
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            return f"Gemini Generation Error: {e}"

    def _generate_mock(self, question: str, chunks: List[RetrievedChunk]) -> str:
        """
        Deterministic extractive answer for offline verification or when API keys are absent.
        Finds sentences matching keywords in question, formats citations, or refuses if unanswerable.
        """
        q_words = set(re.findall(r"\w+", question.lower())) - {
            "what", "is", "the", "are", "how", "and", "in", "to", "of", "a", "for", "does", "by", "with", "an"
        }
        if not q_words:
            return "I don't know based on the provided documents."

        matched_sentences = []
        for c in chunks:
            sentences = re.split(r"(?<=[.!?\n])\s+", c.text)
            for s in sentences:
                s_words = set(re.findall(r"\w+", s.lower()))
                overlap = len(q_words.intersection(s_words))
                if overlap >= 2 or (len(q_words) <= 2 and overlap >= 1):
                    matched_sentences.append((overlap, s.strip(), c.source, c.page))

        if not matched_sentences:
            return "I don't know based on the provided documents."

        matched_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentence = matched_sentences[0][1]
        source = matched_sentences[0][2]
        page = matched_sentences[0][3]

        return f"{top_sentence} [{source}, page {page}]"

    def generate_raw(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generates raw completion without requiring RetrievedChunk list (e.g. for ReAct loop, agents, intent reasoning).
        """
        if self.provider == "openai" and self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt or "You are an intelligent AI assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=settings.MAX_TOKENS_PER_QUERY
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                try:
                    fb = self.openai_client.chat.completions.create(
                        model=settings.FALLBACK_LLM_MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt or "You are an intelligent AI assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=settings.LLM_TEMPERATURE,
                        max_tokens=settings.MAX_TOKENS_PER_QUERY
                    )
                    return fb.choices[0].message.content.strip()
                except Exception as fb_err:
                    return f"LLM Generation Error: {fb_err}"

        elif self.provider == "gemini" and settings.GEMINI_API_KEY:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.LLM_MODEL}:generateContent?key={settings.GEMINI_API_KEY}"
                payload = {
                    "contents": [
                        {
                            "parts": [
                                {"text": f"{system_prompt or ''}\n\n{prompt}".strip()}
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": settings.LLM_TEMPERATURE,
                        "maxOutputTokens": settings.MAX_TOKENS_PER_QUERY
                    }
                }
                with httpx.Client(timeout=30.0) as client:
                    res = client.post(url, json=payload)
                    res.raise_for_status()
                    data = res.json()
                    return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            except Exception as e:
                return f"Gemini Generation Error: {e}"

        return self._generate_mock_raw(prompt)

    def _generate_mock_raw(self, prompt: str) -> str:
        """
        Deterministic mock generator for ReAct agent loops and raw text prompts when running offline.
        """
        lower = prompt.lower()

        # Handle ReAct agent loop progression
        if "observation:" in lower:
            if "error" in lower:
                return "Thought: An error occurred. I will conclude with a fallback.\nFinal Answer: I was unable to complete the query due to a tool error."
            if "calculator" in lower or any(char.isdigit() for char in lower.split("observation:")[-1][:50]):
                obs_part = lower.split("observation:")[-1].strip()
                return f"Thought: I now have enough information to answer the question.\nFinal Answer: Based on the calculation and retrieved evidence, the computed value is {obs_part.splitlines()[0][:100]}."
            return "Thought: I now have enough information to answer the question.\nFinal Answer: Based on the retrieved research papers, FlashAttention and LoRA significantly optimize memory hierarchy and compute efficiency. [flash_attention.md, page 1]"

        # Initial ReAct step
        if "action:" not in lower:
            if "ratio" in lower or ("calculate" in lower and "parameters" in lower):
                return "Thought: I need to search the indexed research papers for parameter figures before calculating.\nAction: document_search\nAction Input: LoRA parameters GPT-3"
            elif any(w in lower for w in ["calculate", "512", "+", "-", "*", "/"]):
                math_match = re.search(r"[\d\.\s\+\-\*\/\(\)]+", prompt)
                expr = math_match.group(0).strip() if math_match else "512 / 8"
                return f"Thought: I need to calculate this arithmetic expression.\nAction: calculator\nAction Input: {expr}"
            elif "catalog" in lower or "what documents" in lower or "metadata" in lower:
                return "Thought: I should inspect the indexed document metadata.\nAction: document_metadata\nAction Input: "
            else:
                q_match = re.search(r"Question:\s*(.+)", prompt)
                q_text = q_match.group(1).strip() if q_match else "research architecture"
                return f"Thought: I need to search the knowledge base for this question.\nAction: document_search\nAction Input: {q_text}"

        return "Final Answer: I don't know based on the provided documents."
