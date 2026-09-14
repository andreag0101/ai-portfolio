"""Structured per-request logging for the RAG ask endpoint.

Just formatted lines to stdout/stderr (readable in Render's log viewer or
`docker logs`), not persisted anywhere durable or aggregated into a
dashboard -- see the model card's "Future work" for what a real version of
this would add. Still enough to answer "what are people actually asking,
how often does retrieval find nothing, how slow is this" without touching
a debugger.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager

logger = logging.getLogger("app.rag")


@contextmanager
def log_ask(question: str):
    """Usage: `with log_ask(question) as log: ...; log["mode"] = ...`

    Logs one line per request on exit (including on exception), with
    whatever fields were set on `log` plus the elapsed wall-clock time.
    """
    start = time.perf_counter()
    fields: dict = {}
    try:
        yield fields
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "rag_ask question=%r mode=%s top_score=%s num_citations=%s elapsed_ms=%.1f",
            question,
            fields.get("mode"),
            fields.get("top_score"),
            fields.get("num_citations"),
            elapsed_ms,
        )
