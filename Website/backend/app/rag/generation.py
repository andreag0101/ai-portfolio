"""Turns retrieved chunks into an answer.

If ANTHROPIC_API_KEY is set, calls Claude to synthesize a grounded, cited
answer from the retrieved passages. If it isn't (e.g. a fresh deploy with no
key configured yet), falls back to an extractive answer -- the top-ranked
passage itself -- so the endpoint is always honest about what it's showing
rather than erroring out.

Haiku 4.5 is used deliberately, not by default: this endpoint is reachable
by anyone who finds the site, so a cheap, fast model bounds the cost of that
exposure. Override with the ANTHROPIC_MODEL env var if you want a stronger
model instead.
"""
from __future__ import annotations

import os

from .retrieval import RetrievedChunk

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")

SYSTEM_PROMPT = """You are a research assistant answering questions about Andrea Goh's \
technical write-ups (deep learning coursework projects: from-scratch backprop and \
optimizers, CNN design, object detection, semantic segmentation, GANs/diffusion, and \
transformer machine translation).

Answer ONLY using the numbered excerpts provided in the user message. Rules:
- Cite the excerpt number(s) you drew on inline, like [1] or [1][3].
- If the excerpts don't contain enough to answer, say so plainly instead of guessing \
or using outside knowledge.
- Be concrete: prefer the specific numbers, architectures, and results in the excerpts \
over vague summary.
- Keep the answer to 2-4 sentences unless the question needs a list.
"""


def _client():
    import anthropic  # imported lazily so the module still loads without the package installed

    return anthropic.Anthropic()


def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, rc in enumerate(chunks, start=1):
        parts.append(f"[{i}] ({rc.chunk.title}, p.{rc.chunk.page})\n{rc.chunk.text}")
    return "\n\n".join(parts)


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> dict:
    if not chunks:
        return {
            "answer": "Nothing in the indexed write-ups looks relevant to that question -- try rephrasing, or ask about one of the suggested questions.",
            "mode": "none",
        }

    if not os.environ.get("ANTHROPIC_API_KEY"):
        top = chunks[0].chunk
        return {
            "answer": f'No live model is configured for this deployment, so here\'s the most relevant passage found instead (from "{top.title}", p.{top.page}): "{top.text}"',
            "mode": "extractive",
        }

    try:
        response = _client().messages.create(
            model=MODEL,
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Excerpts:\n\n{_build_context(chunks)}\n\nQuestion: {question}",
                }
            ],
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        return {"answer": text.strip(), "mode": "generated"}
    except Exception as e:
        top = chunks[0].chunk
        return {
            "answer": f'The live model call failed ({e}), so here\'s the most relevant passage instead (from "{top.title}", p.{top.page}): "{top.text}"',
            "mode": "extractive",
        }
