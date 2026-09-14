# Model Card: Research Assistant (RAG)

A retrieval-augmented Q&A system over 10 of Andrea Goh's own deep-learning
technical write-ups. Served at `/research-assistant` on this portfolio site.

## Intended use

Demonstrating a from-scratch RAG pipeline for portfolio/interview purposes,
and letting a visitor ask questions about the write-ups behind the Deep
Learning project pages instead of reading the PDFs directly. Not intended
for any use beyond this: it is not a general-purpose assistant, has no
access to anything outside the 10 indexed PDFs, and should not be relied on
for factual claims about anything other than the content of those PDFs.

## Data

- **Corpus**: 10 PDF write-ups (`frontend/public/deep-learning/reports/`),
  all authored by Andrea Goh for Purdue's ECE 60146 (Deep Learning)
  coursework. No third-party, licensed, or personal data of any kind.
- **Chunking**: each PDF page is split into ~140-word, 25-word-overlap
  chunks on paragraph boundaries (`app/rag/chunking.py`), 555 chunks total.
  See `backend/gen_rag_index.py`.

## Method

- **Retrieval**: `TfidfVectorizer` (unigrams + bigrams, English stop words,
  8000 max features) fit once over all chunks with each chunk's document
  title prepended (a metadata boost -- see the comment in
  `gen_rag_index.py` for why). At query time, a hand-rolled cosine
  similarity (sparse dot product against the pre-normalized TF-IDF matrix)
  ranks all 555 chunks; the top-k are returned. No vector database, no
  embedding model, no LangChain.
- **Generation**: the top-k chunks are handed to Claude Haiku
  (`app/rag/generation.py`) with a system prompt instructing it to answer
  only from the provided excerpts and cite them inline. If
  `ANTHROPIC_API_KEY` is not set for a deployment, generation is skipped
  entirely and the top retrieved passage is returned verbatim instead
  (`mode: "extractive"` in the API response) -- retrieval never depends on
  the API key.
- **Observability**: every `/api/rag/ask` request logs one structured line
  (question, mode, top retrieval score, citation count, latency) via
  `app/rag/logging_utils.py` -- visible in Render's log viewer or `docker
  logs`. Not persisted or aggregated anywhere beyond that (see
  "Future work").

## Evaluation

No formal eval set (see "Future work"). The closest thing to one is
`backend/tests/test_rag_retrieval.py`: five hand-picked queries with a
known-correct top-ranked document, run against the real bundled index on
every CI run. This is a regression test, not a quality benchmark -- it
would not have caught the original ranking bug on its own; that bug (below)
was found by manually trying an unscripted question.

## Known limitations

- **Keyword retrieval, not semantic retrieval.** TF-IDF matches on term
  overlap, not meaning. A query using different words than the source text
  (paraphrases, synonyms) will retrieve worse results than a query that
  echoes the PDFs' own vocabulary. A dense embedding model (e.g. a small
  sentence-transformer) would generalize better across phrasings; it was
  deliberately not used here to keep retrieval free of any model download
  or inference cost, matching the site's "bundled, standalone, no external
  dependencies" philosophy (see the root `README.md`).
- **Found and fixed once already, could regress:** a chunk that densely
  repeats a generic term (e.g. "loss function" in a backprop write-up) can
  outrank a chunk that's actually about the query's topic but uses the term
  more sparingly. Folding each chunk's document title into its indexed text
  mitigated this for the current corpus and query set, but is a heuristic,
  not a fix for the underlying keyword-matching limitation.
- **Small, narrow, single-author corpus.** 10 PDFs, all the same author,
  all the same course. Retrieval quality here says little about how this
  approach would generalize to a larger or more heterogeneous corpus.
- **No answer quality evaluation of the generated (non-extractive) path.**
  The retrieval regression tests check *ranking*, not whether Claude's
  synthesized answer is accurate or well-cited given good retrieval.
- **Single-process rate limiting.** `/api/rag/ask` rate-limits per IP with
  an in-memory dict (`app/routers/rag.py`), which resets on every deploy
  and does not hold up across multiple instances -- fine for a one-instance
  portfolio demo, not for a production deployment behind a load balancer.

## Future work

- A small labeled eval set (question -> expected chunk id) to measure
  retrieval precision/recall directly, instead of five spot-checked
  queries.
- A dense-embedding retrieval path (e.g. `fastembed`, ONNX-based, no torch)
  as an opt-in alternative, to quantify the actual quality gap against
  TF-IDF on this corpus rather than asserting it.
- Persisting and aggregating the per-request logs (currently stdout-only,
  see "Observability" above) somewhere queryable, to find real retrieval
  failures from actual free-text questions beyond the ones caught by
  manual testing.
