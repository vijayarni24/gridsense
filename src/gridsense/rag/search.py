"""Day 5 RAG — pure retrieval over the Chroma store (no LLM)."""

from __future__ import annotations

from typing import Any, cast

from gridsense.rag.ingest import embed_query, get_collection


def search(query: str, k: int = 5) -> list[dict[str, Any]]:
    """Embed the query and return the top-k chunks from Chroma.

    Each result: {chunk_text, source_pdf, page_number, similarity_score}.
    similarity_score = 1 - cosine_distance (higher is closer).
    """
    query_embedding = embed_query(query)

    collection = get_collection()
    result = collection.query(query_embeddings=cast(Any, [query_embedding]), n_results=k)

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    hits: list[dict[str, Any]] = []
    for document, metadata, distance in zip(documents, metadatas, distances, strict=False):
        hits.append(
            {
                "chunk_text": document,
                "source_pdf": metadata.get("source_pdf", ""),
                "page_number": metadata.get("page_number", -1),
                "similarity_score": 1.0 - float(distance),
            }
        )
    return hits
