from app.rag.chunking import chunk_page


def test_empty_text_returns_no_chunks():
    assert chunk_page("") == []
    assert chunk_page("   \n\n  ") == []


def test_short_paragraph_becomes_one_chunk():
    text = "This is a short paragraph with plenty of words in it to pass the minimum length filter."
    chunks = chunk_page(text, target_words=140, overlap_words=25)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_fragments_shorter_than_eight_words_are_dropped():
    assert chunk_page("too short") == []


def test_long_text_is_split_into_multiple_overlapping_chunks():
    # Three fake paragraphs, each well past the target word count on its own.
    paragraph = " ".join(f"word{i}" for i in range(120))
    text = "\n\n".join([paragraph] * 3)

    chunks = chunk_page(text, target_words=140, overlap_words=25)

    assert len(chunks) > 1
    # Every word in the source text shows up in at least one chunk -- the
    # chunker packs paragraphs greedily but should never silently drop one.
    combined = " ".join(chunks)
    for i in range(120):
        assert f"word{i}" in combined


def test_overlap_repeats_tail_words_across_chunk_boundary():
    paragraphs = [" ".join(f"p{p}w{i}" for i in range(80)) for p in range(3)]
    text = "\n\n".join(paragraphs)

    chunks = chunk_page(text, target_words=90, overlap_words=20)

    assert len(chunks) >= 2
    tail_of_first = chunks[0].split()[-5:]
    assert any(w in chunks[1] for w in tail_of_first)


def test_oversized_single_paragraph_is_windowed_on_its_own():
    huge_paragraph = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_page(huge_paragraph, target_words=140, overlap_words=25)

    assert len(chunks) > 1
    for c in chunks:
        assert len(c.split()) <= 140
