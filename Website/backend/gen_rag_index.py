"""Builds the RAG retrieval index from the bundled deep-learning write-up
PDFs (frontend/public/deep-learning/reports/*.pdf) so the deployed backend
never has to parse PDFs at runtime -- it just loads the three files this
script produces under data/rag/:

  - chunks.json      chunk text + {doc file, title, page, project_slug}
  - vectorizer.pkl    the fitted TfidfVectorizer (joblib)
  - doc_vectors.npz   the chunk x vocabulary TF-IDF matrix (scipy sparse)

Not imported by the app; safe to leave in the repo. Re-run whenever a
write-up PDF is added or changed:

    pip install pypdf   # one-time, this script's only extra dependency
    python gen_rag_index.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
from pypdf import PdfReader
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from app.rag.chunking import chunk_page
from app.rag.corpus import DOCS

REPORTS_DIR = Path(__file__).resolve().parents[1] / "frontend" / "public" / "deep-learning" / "reports"
OUT_DIR = Path(__file__).resolve().parent / "data" / "rag"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_chunks() -> list[dict]:
    chunks: list[dict] = []
    for doc in DOCS:
        path = REPORTS_DIR / doc["file"]
        reader = PdfReader(str(path))
        print(f"  {doc['file']}: {len(reader.pages)} pages")
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for chunk_text_ in chunk_page(text):
                chunks.append(
                    {
                        "id": len(chunks),
                        "text": chunk_text_,
                        "file": doc["file"],
                        "title": doc["title"],
                        "project_slug": doc["project_slug"],
                        "page": page_num,
                    }
                )
    return chunks


def main():
    print("Extracting + chunking PDFs...")
    chunks = extract_chunks()
    print(f"Total chunks: {len(chunks)}")

    # Prepend the write-up's title to each chunk before vectorizing (not for
    # display -- chunks.json keeps the raw text). A query that names the
    # topic ("...for semantic segmentation?") often uses words that appear
    # in the write-up's title far more than in any single paragraph of its
    # body, so without this a chunk that happens to repeat a generic term
    # (e.g. "loss function") densely can outrank the actually-relevant
    # write-up. This is a standard metadata-boost, not query-specific
    # tuning: it's computed once at index time from data every chunk has.
    texts = [f"{c['title']}. {c['text']}" for c in chunks]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=8000,
        sublinear_tf=True,
    )
    doc_vectors = vectorizer.fit_transform(texts)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    (OUT_DIR / "chunks.json").write_text(json.dumps(chunks))
    joblib.dump(vectorizer, OUT_DIR / "vectorizer.pkl")
    sparse.save_npz(OUT_DIR / "doc_vectors.npz", doc_vectors)
    print(f"Wrote index to {OUT_DIR}")


if __name__ == "__main__":
    main()
