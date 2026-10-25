import pytest
from app.agents.tools import CalculatorTool, DocumentSearchTool, DocumentMetadataTool
from app.agents.router import QueryRouter
from app.agents.react_agent import ReActAgent


def test_query_router():
    router = QueryRouter()

    assert router.route("Hello there") == "direct"
    assert router.route("what can you do") == "direct"
    assert router.route("512 / 8") == "calculator"
    assert router.route("calculate 100000 / 12") == "calculator"
    assert router.route("calculate the ratio of parameters in LoRA vs GPT-3") == "agent"
    assert router.route("What is FlashAttention and how does tiling help?") == "rag"


def test_calculator_tool():
    calc = CalculatorTool()

    assert calc.execute("512 / 8") == "64"
    assert calc.execute("2 ** 10") == "1024"
    assert float(calc.execute("(100 - 20) / 100")) == 0.8

    # Safe error handling on syntax error
    err_res = calc.execute("invalid ++ syntax")
    assert "Error" in err_res


def test_document_metadata_tool():
    tool = DocumentMetadataTool()
    res = tool.execute("")
    assert "Indexed Knowledge Base Catalog" in res
    assert "chunks" in res


def test_document_search_tool():
    tool = DocumentSearchTool()
    res = tool.execute("FlashAttention tiling SRAM")
    assert "flash_attention.md" in res or "Page" in res


def test_react_agent_safeguard():
    # Verify that ReAct agent halts within max_iterations
    agent = ReActAgent(max_iterations=2)
    res = agent.run("What is the ratio of attention heads?")
    assert "answer" in res
    assert res["iterations"] <= 2
    assert "steps" in res
