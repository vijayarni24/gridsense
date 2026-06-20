"""RAG unit tests: alias fallback + page-scoped chunking with cross-page carry.

These avoid network/Chroma — they exercise pure logic only.
"""

from __future__ import annotations

from gridsense.rag.aliases import alias_for
from gridsense.rag.ingest import CHUNK_TOKENS, chunk_pages


def test_alias_known_and_fallback() -> None:
    assert alias_for("may2026.pdf") == "EIA EPM May 2026"
    # Unknown file falls back to the filename stem.
    assert alias_for("some_other_file.pdf") == "some_other_file"


def test_short_pages_one_chunk_each_with_correct_page_numbers() -> None:
    chunks = chunk_pages(["alpha content one", "bravo content two"])
    assert [(page, idx) for page, idx, _ in chunks] == [(1, 0), (2, 0)]


def test_cross_page_carry_into_next_page_first_chunk() -> None:
    chunks = chunk_pages(["unique_alpha_token", "unique_bravo_token"])
    page2_text = next(text for page, _, text in chunks if page == 2)
    # Page 2's first chunk should carry page 1's tail as lead-in context.
    assert "unique_bravo_token" in page2_text
    assert "unique_alpha_token" in page2_text


def test_empty_pages_are_skipped() -> None:
    assert chunk_pages(["", "   ", ""]) == []


def test_long_page_splits_into_multiple_overlapping_chunks() -> None:
    long_text = " ".join(f"token{i}" for i in range(CHUNK_TOKENS * 2))
    chunks = chunk_pages([long_text])
    assert len(chunks) > 1
    # All on page 1, chunk_index increments from 0.
    assert all(page == 1 for page, _, _ in chunks)
    assert [idx for _, idx, _ in chunks] == list(range(len(chunks)))
