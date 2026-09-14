from app.rag import corpus


def test_every_doc_maps_to_a_known_project_slug():
    # Loose guard against a typo silently orphaning a citation: every
    # project_slug should look like a real kebab-case route segment.
    for doc in corpus.DOCS:
        assert doc["file"].endswith(".pdf")
        assert doc["project_slug"]
        assert " " not in doc["project_slug"]


def test_docs_by_file_is_consistent_with_docs():
    assert len(corpus.DOCS_BY_FILE) == len(corpus.DOCS)
    for doc in corpus.DOCS:
        assert corpus.DOCS_BY_FILE[doc["file"]] == doc


def test_pdf_href_points_at_the_reports_dir_with_a_page_fragment():
    href = corpus.pdf_href("hw09-gan-diffusion.pdf", 6)
    assert href == "/deep-learning/reports/hw09-gan-diffusion.pdf#page=6"


def test_project_href_points_at_the_deep_learning_route():
    assert corpus.project_href("gan-diffusion") == "/deep-learning/gan-diffusion"
