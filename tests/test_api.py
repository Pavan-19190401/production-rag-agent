import json
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import settings
from app.cache import cache
from app.budget import budget_manager

client = TestClient(app)
VALID_API_KEY = settings.API_KEY


def parse_sse_answer(sse_text: str) -> str:
    """Helper to reconstruct streamed tokens from SSE body."""
    tokens = []
    for line in sse_text.splitlines():
        if line.startswith("data: "):
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                continue
            try:
                payload = json.loads(data_str)
                if "token" in payload:
                    tokens.append(payload["token"])
            except Exception:
                pass
    return "".join(tokens)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ok", "degraded")
    assert "uptime_seconds" in data
    assert "indexed_chunks" in data


def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "telemetry" in data
    assert "budget" in data


def test_auth_unauthorized():
    # Calling /query without valid x-api-key should fail with 401
    response = client.post(
        "/query",
        json={"question": "What is FlashAttention?"},
        headers={"x-api-key": "invalid-wrong-key"}
    )
    assert response.status_code == 401
    assert "Invalid or missing 'x-api-key'" in response.json()["detail"]


def test_query_sse_streaming():
    response = client.post(
        "/query",
        json={"question": "What is FlashAttention and how does tiling help?", "top_k": 3},
        headers={"x-api-key": VALID_API_KEY}
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    body = response.text
    # Verify SSE events format: event: sources and event: done
    assert "event: sources" in body
    assert "event: done" in body
    assert "[DONE]" in body
    assert "flash_attention.md" in body


def test_unanswerable_refusal_path():
    # Tests the strict "I don't know based on the provided documents" path
    response = client.post(
        "/query",
        json={"question": "What is the return and refund policy for annual enterprise subscription plans?"},
        headers={"x-api-key": VALID_API_KEY}
    )
    assert response.status_code == 200
    answer = parse_sse_answer(response.text)
    assert "I don't know based on the provided documents" in answer


def test_caching_behavior():
    cache.clear()
    q = "What is the primary memory bottleneck FlashAttention addresses?"

    # First request: Populates cache
    res1 = client.post(
        "/query",
        json={"question": q},
        headers={"x-api-key": VALID_API_KEY}
    )
    assert res1.status_code == 200

    # Second request: Served from cache
    res2 = client.post(
        "/query",
        json={"question": q},
        headers={"x-api-key": VALID_API_KEY}
    )
    assert res2.status_code == 200
    assert '"cached": true' in res2.text.lower() or "cached" in res2.text


def test_feedback_endpoint():
    res = client.post(
        "/feedback",
        json={"request_id": "test-req-001", "rating": 1, "comment": "Excellent answer!"},
        headers={"x-api-key": VALID_API_KEY}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"


def test_budget_cap_throttling():
    # Temporarily set budget to 0 to simulate limit breach
    original_budget = budget_manager.daily_budget_usd
    original_spend = budget_manager.daily_spend_usd
    try:
        budget_manager.daily_budget_usd = 0.001
        budget_manager.daily_spend_usd = 0.005  # exceed limit

        response = client.post(
            "/query",
            json={"question": "What is Scaled Dot Product Attention?"},
            headers={"x-api-key": VALID_API_KEY}
        )
        assert response.status_code == 429
        assert "budget cap" in response.json()["detail"].lower()
    finally:
        budget_manager.daily_budget_usd = original_budget
        budget_manager.daily_spend_usd = original_spend
