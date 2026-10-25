import ast
import json
import math
import operator
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from app.config import settings, BASE_DIR
from app.retrieval.pipeline import get_default_retriever

CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.json"


@dataclass
class Tool:
    name: str
    description: str

    def execute(self, argument: str) -> str:
        raise NotImplementedError


class DocumentSearchTool(Tool):
    def __init__(self):
        super().__init__(
            name="document_search",
            description="Searches the indexed research documents using hybrid retrieval. Input: search query string."
        )
        self.retriever = get_default_retriever(mode="hybrid")

    def execute(self, argument: str) -> str:
        query = argument.strip().strip('"\'')
        chunks = self.retriever.retrieve(query, top_k=3)
        if not chunks:
            return "No relevant passages found in the indexed documents."

        results = []
        for idx, c in enumerate(chunks, start=1):
            results.append(
                f"[Doc {idx}] {c.source} (Page {c.page}, Score: {c.score:.4f}):\n{c.text[:250]}..."
            )
        return "\n\n".join(results)


class CalculatorTool(Tool):
    """Safe mathematical evaluator using Abstract Syntax Trees (AST)."""

    ALLOWED_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def __init__(self):
        super().__init__(
            name="calculator",
            description="Evaluates arithmetic expressions and ratios safely. Input: a mathematical string like '(175e9 - 37.7e6) / 175e9' or '512 / 8'."
        )

    def _eval_node(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Unsupported constant: {node.value}")
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type in self.ALLOWED_OPERATORS:
                left = self._eval_node(node.left)
                right = self._eval_node(node.right)
                return self.ALLOWED_OPERATORS[op_type](left, right)
            raise ValueError(f"Operator not allowed: {op_type}")
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type in self.ALLOWED_OPERATORS:
                operand = self._eval_node(node.operand)
                return self.ALLOWED_OPERATORS[op_type](operand)
            raise ValueError(f"Unary operator not allowed: {op_type}")
        else:
            raise ValueError(f"Unsupported syntax expression: {type(node).__name__}")

    def execute(self, argument: str) -> str:
        expr = argument.strip().strip('"\'')
        try:
            tree = ast.parse(expr, mode="eval")
            res = self._eval_node(tree.body)
            # Format nicely
            if abs(res) < 0.0001 or abs(res) > 1000000:
                return f"{res:.6e}"
            return f"{res:.4f}".rstrip("0").rstrip(".")
        except Exception as e:
            return f"Error evaluating expression '{expr}': {e}"


class DocumentMetadataTool(Tool):
    def __init__(self):
        super().__init__(
            name="document_metadata",
            description="Inspects indexed corpus catalog, listing filenames, chunk counts, and total indexed pages. Input: empty or document filename."
        )

    def execute(self, argument: str) -> str:
        if not CHUNKS_PATH.exists():
            return "No indexed chunk catalog found."

        try:
            with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            doc_stats: Dict[str, Dict[str, Any]] = {}
            for c in chunks:
                src = c.get("source", "unknown")
                if src not in doc_stats:
                    doc_stats[src] = {"chunks": 0, "pages": set()}
                doc_stats[src]["chunks"] += 1
                doc_stats[src]["pages"].add(c.get("page", 1))

            lines = ["Indexed Knowledge Base Catalog:"]
            for doc, s in doc_stats.items():
                lines.append(f"- {doc}: {s['chunks']} chunks across {len(s['pages'])} page(s)")

            return "\n".join(lines)
        except Exception as e:
            return f"Error reading document metadata: {e}"


# Tool Registry
TOOL_REGISTRY: Dict[str, Tool] = {
    "document_search": DocumentSearchTool(),
    "calculator": CalculatorTool(),
    "document_metadata": DocumentMetadataTool(),
}
