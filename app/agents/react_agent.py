import re
import time
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
    """

    def __init__(self, max_iterations: int = 4):
        self.max_iterations = max_iterations
        self.tools: Dict[str, Tool] = TOOL_REGISTRY
        self.llm = LLMClient()

    def run(self, question: str) -> Dict[str, Any]:
        """Synchronously executes the ReAct loop and returns steps + final answer."""
        t_start = time.perf_counter()
        steps: List[Dict[str, Any]] = []
        scratchpad = f"Question: {question}\n"

        for iteration in range(1, self.max_iterations + 1):
            prompt = f"{REACT_SYSTEM_PROMPT}\n\n{scratchpad}Thought:"

            # Query LLM (or mock dispatch for offline/tests)
            llm_res = self.llm.generate(prompt, chunks=[])
            output_text = "Thought: " + llm_res["answer"]

            # Check for Final Answer
            if "Final Answer:" in output_text:
                final_answer = output_text.split("Final Answer:", 1)[1].strip()
                steps.append({"iteration": iteration, "type": "final_answer", "content": final_answer})
                return {
                    "answer": final_answer,
                    "steps": steps,
                    "iterations": iteration,
                    "duration_ms": round((time.perf_counter() - t_start) * 1000, 2)
                }

            # Parse Action and Action Input
            action_match = re.search(r"Action:\s*(\w+)", output_text)
            action_input_match = re.search(r"Action Input:\s*(.+)", output_text)

            if not action_match:
                # Agent provided a direct thought without tool call
                thought_text = output_text.replace("Thought:", "").strip()
                steps.append({"iteration": iteration, "type": "thought", "content": thought_text})
                # Attempt to extract answer or force completion
                final_answer = thought_text
                return {
                    "answer": final_answer,
                    "steps": steps,
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

        # Max iterations reached: Synthesize answer from scratchpad
        fallback_answer = "I was unable to reach a conclusive answer within the maximum reasoning steps."
        steps.append({"iteration": self.max_iterations, "type": "final_answer", "content": fallback_answer})
        return {
            "answer": fallback_answer,
            "steps": steps,
            "iterations": self.max_iterations,
            "duration_ms": round((time.perf_counter() - t_start) * 1000, 2)
        }
