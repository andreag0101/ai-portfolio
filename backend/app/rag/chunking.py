"""Splits per-page PDF text into overlapping, word-bounded chunks.

Deliberately simple (no sentence tokenizer, no external NLP library): split
on paragraph breaks first, then greedily pack paragraphs into ~target-sized
chunks, falling back to a plain word-count window for any single paragraph
longer than the target. A `overlap` tail of words is repeated at the start
of the next chunk so a fact split across a chunk boundary isn't orphaned
from its context in either chunk.
"""
from __future__ import annotations

import re


def _split_paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text)]
    return [p for p in paras if p]


def chunk_page(text: str, target_words: int = 140, overlap_words: int = 25) -> list[str]:
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush():
        if current:
            chunks.append(" ".join(current).strip())

    for para in paragraphs:
        words = para.split()
        if current_len + len(words) > target_words and current:
            flush()
            tail = current[-overlap_words:] if overlap_words else []
            current = list(tail)
            current_len = len(current)

        if len(words) > target_words * 1.5:
            # A single oversized paragraph: window it on its own.
            flush()
            current, current_len = [], 0
            start = 0
            while start < len(words):
                window = words[start : start + target_words]
                chunks.append(" ".join(window).strip())
                start += target_words - overlap_words
            continue

        current.extend(words)
        current_len += len(words)

    flush()
    return [c for c in chunks if len(c.split()) >= 8]  # drop stray fragments
