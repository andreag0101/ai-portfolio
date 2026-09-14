from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..rag import retrieval
from ..rag.generation import generate_answer
from ..rag.logging_utils import log_ask

router = APIRouter(prefix="/api/rag", tags=["rag"])

SUGGESTED_PATH = Path(__file__).resolve().parents[2] / "data" / "rag" / "suggested_qa.json"

# Single-process, in-memory rate limit: fine for one Render instance serving
# a portfolio demo, not for a multi-instance deployment (which would need a
# shared store like Redis). Caps how much a public, unauthenticated endpoint
# that may call a paid LLM API can cost per visitor.
_RATE_LIMIT = 12  # requests
_RATE_WINDOW = 60  # seconds
_requests: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(client_id: str):
    now = time.time()
    recent = [t for t in _requests[client_id] if now - t < _RATE_WINDOW]
    if len(recent) >= _RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many questions -- wait a minute and try again.")
    recent.append(now)
    _requests[client_id] = recent


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=300)


@router.get("/documents")
async def documents():
    return {"documents": retrieval.document_list()}


@router.get("/suggested")
async def suggested():
    if not SUGGESTED_PATH.exists():
        return {"suggested": []}
    return json.loads(SUGGESTED_PATH.read_text())


@router.post("/ask")
async def ask(body: AskRequest, request: Request):
    _check_rate_limit(request.client.host if request.client else "unknown")

    with log_ask(body.question) as log:
        chunks = retrieval.retrieve(body.question, k=5)
        result = generate_answer(body.question, chunks)
        citations = [retrieval.to_citation(c) for c in chunks[:3]] if result["mode"] != "none" else []

        log["mode"] = result["mode"]
        log["top_score"] = round(chunks[0].score, 4) if chunks else None
        log["num_citations"] = len(citations)

    return {"answer": result["answer"], "mode": result["mode"], "citations": citations}
