"""Policy retrieval tools backed by the RAG pipeline."""

import os

from rag.pipeline import (
    load_policy_documents,
    chunk_documents,
    build_index,
    retrieve_policy_evidence,
)


_POLICY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "policies",
)

_index = None


def _get_index():
    global _index
    if _index is None:
        documents = load_policy_documents(_POLICY_DIR)
        chunks = chunk_documents(documents)
        _index = build_index(chunks)
    return _index


def search_policy(
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """Retrieve policy evidence relevant to a natural-language query."""
    if not query:
        return []
    return retrieve_policy_evidence(_get_index(), query, top_k=top_k)


def get_policy_document(source: str) -> dict:
    """Retrieve a complete policy document by source name."""
    if not source:
        return {"error": "source is required"}

    filename = os.path.basename(source)
    path = os.path.join(_POLICY_DIR, filename)
    if not os.path.isfile(path):
        return {"error": f"Unknown policy document: {filename}"}

    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    return {"source": filename, "text": text}
