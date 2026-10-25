import re
import time
import asyncio
from typing import List, Dict, Any, Optional, AsyncIterator
from app.config import settings
from app.agents.tools import TOOL_REGISTRY, Tool
from app.generation.llm import LLMClient

REACT_SYSTEM_PROMPT = """You are a helpful ReAct AI Agent with access to tools.
Solve the user's question step by step using the following format:

Question: the input question
Thought: your reasoning about what step to take next
Action: the tool to use, must be exactly one of: [document_search, calculator, document_metadata]
Action Input: the exact input argument for the tool
Observation: the result of the tool

When you have gathered enough information to answer the question, conclude with:
Thought: I now have enough information to answer the question.
Final Answer: your final, comprehensive answer with citations [source, page].

Important Rules:
- If a calculation or ratio is needed, use the 'calculator' tool.
- If information from research papers is needed, use the 'document_search' tool.
- If the documents do not contain the answer, conclude with: "Final Answer: I don't know based on the provided documents."
"""


class ReActAgent:
    """
    Hand-rolled, resilient ReAct agent loop with execution safeguards:
    - Step limit enforcement (prevents infinite loops)
    - Error-resilient tool dispatch
    - Multi-step reasoning trace
    - Real-time Server-Sent Events (SSE) streaming support
    """

    def __init__(self, max_iterations: int = 4):
        self.max_iterations = max_iterations
        self.tools: Dict[str, Tool] = TOOL_REGISTRY
        self.llm = LLMClient()

    def _get_accumulated_citations(self) -> List[Dict[str, Any]]:
        search_tool = self.tools.get("document_search")
        if not search_tool or not hasattr(search_tool, "last_retrieved_chunks"):
            return []
        citations = []
        seen = set()
        for c in search_tool.last_retrieved_chunks:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                citations.append({
                    "source": c.source,
                    "page": c.page,
                    "chunk_id": c.chunk_id,
                    "score": round(c.score, 4),
                    "snippet": c.text[:180].replace("\n", " ") + "..."
                })
        return citations

    def run(self, question: str) -> Dict[str, Any]:
        """Synchronously executes the ReAct loop and returns steps + final answer."""
        t_start = time.perf_counter()
        steps: List[Dict[str, Any]] = []
        scratchpad = f"Question: {question}\n"

        search_tool = self.tools.get("document_search")
        if search_tool and hasattr(search_tool, "clear"):
            search_tool.clear()

        for iteration in range(1, self.max_iterations + 1):
            prompt = f"{scratchpad}Thought:"

            output_text = self.llm.generate_raw(prompt, system_prompt=REACT_SYSTEM_PROMPT)
            if not output_text.startswith("Thought:"):
                output_text = "Thought: " + output_text

            # Check for Final Answer
            if "Final Answer:" in output_text:
                final_answer = output_text.split("Final Answer:", 1)[1].strip()
                steps.append({"iteration": iteration, "type": "final_answer", "content": final_answer})
                return {
                    "answer": final_answer,
                    "steps": steps,
                    "citations": self._get_accumulated_citations(),
                    "iterations": iteration,
                    "duration_ms": round((time.perf_counter() - t_start) * 1000, 2)
                }

            # Parse Action and Action Input
            action_match = re.search(r"Action:\s*(\w+)", output_text)
            action_input_match = re.search(r"Action Input:\s*(.+)", output_text)

            if not action_match:
                thought_text = output_text.replace("Thought:", "").strip()
                steps.append({"iteration": iteration, "type": "thought", "content": thought_text})
                final_answer = thought_text
                return {
                    "answer": final_answer,
                    "steps": steps,
                    "citations": self._get_accumulated_citations(),
                    "iterations": iteration,
                    "duration_ms": round((time.perf_counter() - t_start) * 1000, 2)
                }

            tool_name = action_match.group(1).strip()
            tool_input = action_input_match.group(1).strip() if action_input_match else ""

            thought_piece = output_text[:action_match.start()].replace("Thought:", "").strip()
            steps.append({
                "iteration": iteration,
                "type": "thought",
                "content": thought_piece,
                "tool": tool_name,
                "tool_input": tool_input
            })

            # Execute tool
            if tool_name in self.tools:
                try:
                    observation = self.tools[tool_name].execute(tool_input)
                except Exception as e:
                    observation = f"Error executing tool '{tool_name}': {e}"
            else:
                observation = f"Unknown tool '{tool_name}'. Available tools: {list(self.tools.keys())}"

            steps.append({
                "iteration": iteration,
                "type": "observation",
                "tool": tool_name,
                "content": observation
            })

            # Append to scratchpad for next iteration
            scratchpad += f"Thought: {thought_piece}\nAction: {tool_name}\nAction Input: {tool_input}\nObservation: {observation}\n"

        # Max iterations reached
        fallback_answer = "I was unable to reach a conclusive answer within the maximum reasoning steps."
        steps.append({"iteration": self.max_iterations, "type": "final_answer", "content": fallback_answer})
        return {
            "answer": fallback_answer,
            "steps": steps,
            "citations": self._get_accumulated_citations(),
            "iterations": self.max_iterations,
            "duration_ms": round((time.perf_counter() - t_start) * 1000, 2)
        }

    async def astream_run(self, question: str) -> AsyncIterator[Dict[str, Any]]:
        """
        Asynchronously streams ReAct thoughts, actions, observations, citations,
        and final answer tokens.
        """
        scratchpad = f"Question: {question}\n"
        search_tool = self.tools.get("document_search")
        if search_tool and hasattr(search_tool, "clear"):
            search_tool.clear()

        for iteration in range(1, self.max_iterations + 1):
            prompt = f"{scratchpad}Thought:"

            output_text = self.llm.generate_raw(prompt, system_prompt=REACT_SYSTEM_PROMPT)
            if not output_text.startswith("Thought:"):
                output_text = "Thought: " + output_text

            # Check for Final Answer
            if "Final Answer:" in output_text:
                final_answer = output_text.split("Final Answer:", 1)[1].strip()
                thought_piece = output_text.split("Final Answer:", 1)[0].replace("Thought:", "").strip()
                if thought_piece:
                    yield {
                        "type": "thought",
                        "iteration": iteration,
                        "thought": thought_piece,
                        "tool": None,
                        "tool_input": None
                    }
                    await asyncio.sleep(0.02)

                citations = self._get_accumulated_citations()
                yield {"type": "sources", "citations": citations}

                words = final_answer.split(" ")
                for i, w in enumerate(words):
                    yield {"type": "token", "token": w + (" " if i < len(words) - 1 else "")}
                    await asyncio.sleep(0.01)

                yield {"type": "done", "answer": final_answer, "citations": citations}
                return

            # Parse Action and Action Input
            action_match = re.search(r"Action:\s*(\w+)", output_text)
            action_input_match = re.search(r"Action Input:\s*(.+)", output_text)

            if not action_match:
                thought_text = output_text.replace("Thought:", "").strip()
                yield {
                    "type": "thought",
                    "iteration": iteration,
                    "thought": thought_text,
                    "tool": None,
                    "tool_input": None
                }
                citations = self._get_accumulated_citations()
                yield {"type": "sources", "citations": citations}
                words = thought_text.split(" ")
                for i, w in enumerate(words):
                    yield {"type": "token", "token": w + (" " if i < len(words) - 1 else "")}
                    await asyncio.sleep(0.01)
                yield {"type": "done", "answer": thought_text, "citations": citations}
                return

            tool_name = action_match.group(1).strip()
            tool_input = action_input_match.group(1).strip() if action_input_match else ""
            thought_piece = output_text[:action_match.start()].replace("Thought:", "").strip()

            yield {
                "type": "thought",
                "iteration": iteration,
                "thought": thought_piece,
                "tool": tool_name,
                "tool_input": tool_input
            }
            await asyncio.sleep(0.02)

            # Execute tool
            if tool_name in self.tools:
                try:
                    observation = self.tools[tool_name].execute(tool_input)
                except Exception as e:
                    observation = f"Error executing tool '{tool_name}': {e}"
            else:
                observation = f"Unknown tool '{tool_name}'. Available tools: {list(self.tools.keys())}"

            yield {
                "type": "observation",
                "iteration": iteration,
                "tool": tool_name,
                "content": observation
            }
            await asyncio.sleep(0.02)

            # Append to scratchpad
            scratchpad += f"Thought: {thought_piece}\nAction: {tool_name}\nAction Input: {tool_input}\nObservation: {observation}\n"

        # Max iterations reached
        fallback_answer = "I was unable to reach a conclusive answer within the maximum reasoning steps."
        citations = self._get_accumulated_citations()
        yield {"type": "sources", "citations": citations}
        words = fallback_answer.split(" ")
        for i, w in enumerate(words):
            yield {"type": "token", "token": w + (" " if i < len(words) - 1 else "")}
            await asyncio.sleep(0.01)
        yield {"type": "done", "answer": fallback_answer, "citations": citations}
