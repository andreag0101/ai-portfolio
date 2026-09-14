"""Smoke tests: the app imports, starts, and every router's cheap endpoints
respond. Deliberately skips the endpoints that train a classifier on first
request (face/texture/car-detection's classify routes, calibration's
run-sample) -- those are exercised by gen_precomputed.py locally, not CI.

This is exactly the kind of test that would have caught today's "the whole
app fails to import" class of bug -- e.g. if a new router's module-level
code (like the RAG index load) throws on startup, main.py's import fails
and the app never boots, which is a very different failure mode than a
single endpoint returning a bad response.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


@pytest.mark.parametrize(
    "path",
    [
        "/api/segmentation/samples",
        "/api/panorama/sample",
        "/api/disparity/sample",
        "/api/rectify/sample",
        "/api/rectify/insert-sample",
        "/api/corners/sample",
        "/api/texture/samples",
        "/api/face/samples",
        "/api/calibration/samples",
        "/api/calibration/pattern",
        "/api/car-detection/samples",
        "/api/rag/documents",
        "/api/rag/suggested",
    ],
)
def test_sample_endpoints_respond(path):
    res = client.get(path)
    assert res.status_code == 200, res.text


def test_rag_ask_end_to_end_without_an_api_key():
    # No ANTHROPIC_API_KEY in the test environment -- this exercises the
    # extractive fallback path, not live generation.
    res = client.post("/api/rag/ask", json={"question": "What optimizer converged fastest?"})
    assert res.status_code == 200
    body = res.json()
    assert body["mode"] in ("extractive", "generated", "none")
    assert isinstance(body["answer"], str) and body["answer"]


def test_rag_ask_rejects_too_short_a_question():
    res = client.post("/api/rag/ask", json={"question": "hi"})
    assert res.status_code == 422
