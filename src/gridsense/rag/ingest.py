"""Day 5 RAG — ingest PDFs into a local Chroma vector store.

Pipeline per PDF:
    pypdf (layout mode) -> per-page text (whitespace-normalized)
    -> ~800-token chunks (100-token overlap, page-scoped + cross-page carry)
    -> local fastembed embeddings (BAAI/bge-small-en-v1.5) -> Chroma upsert.

Embeddings run locally (ONNX, no API quota) so the whole corpus ingests offline;
only the answer step uses Gemini. Chunks are page-scoped so each carries an exact
page_number for citation; the first chunk of each page is prefixed with the prior
page's trailing ~100 tokens, so a paragraph split across a page break stays
retrievable whole — but its citation stays the page where its bulk lives.

Idempotent: chunk id = sha256(source|page|chunk_index|text), then upsert; a re-run
skips chunks already present.

NOTE: pypdf 'layout' mode keeps row labels near their numbers for the EIA tables,
but pure-text extraction of dense numeric tables is the known weak spot — a
specific cell may extract imperfectly, so a numeric EIA answer can legitimately
come back as the refusal sentinel rather than a wrong number.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Any, cast

import chromadb
import tiktoken
from chromadb import Collection
from dotenv import load_dotenv
from fastembed import TextEmbedding
from google import genai
from pypdf import PdfReader

CHUNK_TOKENS = 800
OVERLAP_TOKENS = 100
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # local ONNX model, 384-dim, no API quota
CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "gridsense_docs"

_ENCODER = tiktoken.get_encoding("cl100k_base")
_WHITESPACE_RUN = re.compile(r"[ \t]{2,}")
_EMBEDDER: TextEmbedding | None = None


def _normalize(text: str) -> str:
    """Collapse intra-line whitespace runs and drop blank lines.

    pypdf 'layout' mode pads columns with spaces to preserve alignment; those
    spaces are tokens that inflate chunk counts without adding meaning. Collapsing
    runs (while keeping newlines) keeps each row's label next to its numbers.
    """
    lines = [_WHITESPACE_RUN.sub(" ", line).rstrip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line.strip())


def get_client() -> genai.Client:
    """Gemini client for the answer step; reads GOOGLE_API_KEY from .env/env."""
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY not set (add it to .env).")
    return genai.Client(api_key=api_key)


def get_collection() -> Collection:
    """Persistent Chroma collection using cosine distance."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def _embedder() -> TextEmbedding:
    """Lazily construct (and cache) the local embedding model."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = TextEmbedding(model_name=EMBED_MODEL_NAME)
    return _EMBEDDER


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed passages locally; one vector per input text."""
    return [cast("list[float]", vector.tolist()) for vector in _embedder().passage_embed(texts)]


def embed_query(text: str) -> list[float]:
    """Embed a search query locally (bge query prefix applied by fastembed)."""
    vector = next(iter(_embedder().query_embed([text])))
    return cast("list[float]", vector.tolist())


def extract_pages(pdf_path: Path) -> list[str]:
    """Per-page text via pypdf layout mode, whitespace-normalized."""
    reader = PdfReader(str(pdf_path))
    return [_normalize(page.extract_text(extraction_mode="layout") or "") for page in reader.pages]


def chunk_pages(pages: list[str]) -> list[tuple[int, int, str]]:
    """Token-chunk pages into (page_number, chunk_index, text), with cross-page carry.

    Page numbers are 1-based. The first chunk of each page (after page 1) is
    prefixed with the prior page's trailing OVERLAP_TOKENS tokens.
    """
    step = CHUNK_TOKENS - OVERLAP_TOKENS
    chunks: list[tuple[int, int, str]] = []
    prev_tail: list[int] = []

    for page_number, page_text in enumerate(pages, start=1):
        tokens = _ENCODER.encode(page_text)
        if not tokens:
            continue

        chunk_index = 0
        for win_start in range(0, len(tokens), step):
            window = tokens[win_start : win_start + CHUNK_TOKENS]
            if chunk_index == 0 and prev_tail:
                window = prev_tail + window  # heal a page-break split into this page's first chunk
            text = _ENCODER.decode(window).strip()
            if text:
                chunks.append((page_number, chunk_index, text))
                chunk_index += 1
            if win_start + CHUNK_TOKENS >= len(tokens):
                break

        prev_tail = tokens[-OVERLAP_TOKENS:]

    return chunks


def _chunk_id(source_pdf: str, page_number: int, chunk_index: int, text: str) -> str:
    raw = f"{source_pdf}|{page_number}|{chunk_index}|{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def ingest_directory(directory: Path) -> tuple[int, int]:
    """Ingest every PDF in ``directory``. Returns (n_chunks, n_pdfs).

    Resumable: chunks already in the collection are skipped (not re-embedded).
    """
    pdf_paths = sorted(directory.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDFs found in {directory}")

    collection = get_collection()
    total_chunks = 0
    embedded = 0
    skipped = 0

    for pdf_path in pdf_paths:
        source_pdf = pdf_path.name
        records = chunk_pages(extract_pages(pdf_path))
        total_chunks += len(records)
        if not records:
            print(f"[ingest] {source_pdf}: no extractable text, skipped", file=sys.stderr)
            continue

        ids = [_chunk_id(source_pdf, page, idx, text) for page, idx, text in records]
        texts = [text for _, _, text in records]
        metadatas: list[chromadb.types.Metadata] = [
            {"source_pdf": source_pdf, "page_number": page, "chunk_index": idx}
            for page, idx, _ in records
        ]

        existing = set(collection.get(ids=ids).get("ids") or [])
        keep = [i for i in range(len(records)) if ids[i] not in existing]
        skipped += len(records) - len(keep)

        if keep:
            k_texts = [texts[i] for i in keep]
            collection.upsert(
                ids=[ids[i] for i in keep],
                embeddings=cast(Any, embed_documents(k_texts)),  # -> Chroma's numpy-ish type
                documents=k_texts,
                metadatas=[metadatas[i] for i in keep],
            )
            embedded += len(keep)

        print(f"[ingest] {source_pdf}: {embedded} embedded / {skipped} skipped", file=sys.stderr)

    print(
        f"[ingest] done — {embedded} embedded, {skipped} already present, {total_chunks} total",
        file=sys.stderr,
    )
    return total_chunks, len(pdf_paths)
