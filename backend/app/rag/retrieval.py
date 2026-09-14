"""Loads the precomputed TF-IDF index (built by gen_rag_index.py) once at
import time, then answers each query with a hand-rolled cosine-similarity
ranking -- no vector database, no LangChain: the vectorizer does term
weighting, everything after that (scoring, ranking, top-k selection) is a
plain numpy dot product against the (already L2-normalized) TF-IDF matrix.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse

from . import corpus

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "rag"


@dataclass
class Chunk:
    id: int
    text: str
    file: str
    title: str
    project_slug: str
    page: int


def _load():
    chunks_raw = json.loads((DATA_DIR / "chunks.json").read_text())
    chunks = [Chunk(**c) for c in chunks_raw]
    vectorizer = joblib.load(DATA_DIR / "vectorizer.pkl")
    doc_vectors = sparse.load_npz(DATA_DIR / "doc_vectors.npz")
    return chunks, vectorizer, doc_vectors


_CHUNKS, _VECTORIZER, _DOC_VECTORS = _load()


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


def retrieve(query: str, k: int = 5) -> list[RetrievedChunk]:
    query_vec = _VECTORIZER.transform([query])  # sparse, 1 x vocab, L2-normalized
    # TF-IDF vectors are unit-normalized, so the dot product against every
    # chunk *is* the cosine similarity -- no separate norm division needed.
    scores = (_DOC_VECTORS @ query_vec.T).toarray().ravel()

    if not np.any(scores > 0):
        return []

    top_idx = np.argsort(-scores)[:k]
    return [RetrievedChunk(chunk=_CHUNKS[i], score=float(scores[i])) for i in top_idx if scores[i] > 0]


def to_citation(rc: RetrievedChunk) -> dict:
    return {
        "title": rc.chunk.title,
        "page": rc.chunk.page,
        "snippet": rc.chunk.text[:280],
        "pdfHref": corpus.pdf_href(rc.chunk.file, rc.chunk.page),
        "projectHref": corpus.project_href(rc.chunk.project_slug),
        "score": round(rc.score, 4),
    }


def document_list() -> list[dict]:
    return [
        {
            "file": d["file"],
            "title": d["title"],
            "pdfHref": f"{corpus.REPORTS_URL_PREFIX}/{d['file']}",
            "projectHref": corpus.project_href(d["project_slug"]),
        }
        for d in corpus.DOCS
    ]
