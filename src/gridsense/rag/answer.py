"""Day 5 RAG — answer a question strictly from retrieved chunks, with citations."""

from __future__ import annotations

import re
import sys
from typing import Any

from google.genai import types

from gridsense.rag.aliases import alias_for
from gridsense.rag.ingest import get_client
from gridsense.rag.search import search

ANSWER_MODEL = "gemini-2.5-flash"
REFUSAL = "I cannot answer this from the available documents."
_CITATION_RE = re.compile(r"\[[^\]]+ p\.\d+\]")

PROMPT_TEMPLATE = """You are GridSense, an energy-document analyst. Answer the QUESTION using ONLY \
the CONTEXT chunks below.

Rules:
- Use only facts found in the chunks. Do not use any outside knowledge.
- After every factual sentence, cite the chunk(s) it came from using the exact bracketed tag shown \
above each chunk, e.g. [Silver Star p.4].
- If the chunks do not contain the answer, reply with exactly this and nothing else: {refusal}

CONTEXT:
{context}

QUESTION: {query}
"""


def _format_context(hits: list[dict[str, Any]]) -> str:
    blocks = []
    for hit in hits:
        tag = f"[{alias_for(hit['source_pdf'])} p.{hit['page_number']}]"
        blocks.append(f"{tag}\n{hit['chunk_text']}")
    return "\n\n---\n\n".join(blocks)


def answer(query: str, k: int = 5) -> str:
    """Retrieve k chunks and have Gemini answer using only those, with citations."""
    hits = search(query, k)
    if not hits:
        return REFUSAL

    prompt = PROMPT_TEMPLATE.format(refusal=REFUSAL, context=_format_context(hits), query=query)
    client = get_client()
    response = client.models.generate_content(
        model=ANSWER_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0),
    )
    text = (response.text or "").strip()

    # Guardrail: if it's not the refusal and has no citation tag, surface a warning.
    if text and text != REFUSAL and not _CITATION_RE.search(text):
        print("[warn] answer has no citation tag — treat with suspicion", file=sys.stderr)
    return text
