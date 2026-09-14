"""Runs against the real bundled index (backend/data/rag/), not a fixture --
these are effectively regression tests for retrieval *quality*, not just
plumbing. They pin down a bug that shipped once already: TF-IDF over raw
chunk text let a generic, densely-repeated term ("loss function") in one
write-up outrank the write-up the question was actually about, fixed by
folding each chunk's document title into the text before vectorizing (see
gen_rag_index.py). If this regresses, these are the tests that should catch
it before a user does.
"""
import pytest

from app.rag.retrieval import retrieve, to_citation, document_list


# (query, expected top-ranked document title)
KNOWN_GOOD_QUERIES = [
    ("What FID score did the from-scratch DCGAN get compared to the pretrained diffusion model?", "GANs vs. Diffusion: Face Generation"),
    ("Which optimizer converged fastest for the multi-neuron model?", "Backpropagation & Optimizers, From Scratch"),
    ("What loss function was used for semantic segmentation and did ASPP help?", "Semantic Segmentation: mUNet + ASPP"),
    ("Post-LayerNorm vs pre-LayerNorm transformer translation Levenshtein edit distance", "Transformer Machine Translation: Post-LN vs. Pre-LN"),
    ("What anchor boxes and grid size did the object detector use?", "Object Detection From Scratch on COCO"),
]


@pytest.mark.parametrize("query,expected_title", KNOWN_GOOD_QUERIES)
def test_top_result_matches_expected_document(query, expected_title):
    results = retrieve(query, k=3)
    assert results, f"no results for: {query}"
    assert results[0].chunk.title == expected_title


def test_results_are_sorted_by_descending_score():
    results = retrieve("CNN depth accuracy CIFAR-10", k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_nonsense_query_returns_no_results():
    assert retrieve("zzz qqq xyzzy nonexistent gibberish term", k=5) == []


def test_to_citation_shape():
    results = retrieve("Adam optimizer momentum", k=1)
    citation = to_citation(results[0])
    assert citation["title"] == results[0].chunk.title
    assert citation["page"] == results[0].chunk.page
    assert citation["pdfHref"].startswith("/deep-learning/reports/")
    assert f"#page={results[0].chunk.page}" in citation["pdfHref"]
    assert citation["projectHref"].startswith("/deep-learning/")
    assert len(citation["snippet"]) <= 280


def test_document_list_covers_every_corpus_doc():
    from app.rag.corpus import DOCS

    docs = document_list()
    assert len(docs) == len(DOCS)
    assert {d["file"] for d in docs} == {d["file"] for d in DOCS}
