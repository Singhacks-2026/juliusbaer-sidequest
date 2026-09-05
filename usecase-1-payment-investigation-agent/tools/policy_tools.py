"""
Policy retrieval tool interfaces.

The agent should use these methods to obtain policy evidence rather than
opening policy files directly.

The implementation should preserve the source document name so that the final
assistant can cite the evidence.
"""

import os

from rag.pipeline import (
    build_index,
    chunk_documents,
    load_policy_documents,
    retrieve_policy_evidence,
)

_POLICY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "policies",
)

_index = None


def _get_index():
    """Build the RAG index once and reuse it across calls."""
    global _index
    if _index is None:
        docs = load_policy_documents(_POLICY_DIR)
        chunks = chunk_documents(docs)
        _index = build_index(chunks)
    return _index


def search_policy(
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve policy evidence relevant to a natural-language query.

    Returns ``[{source, text, score}]`` — source filenames preserved for
    citations.
    """
    return retrieve_policy_evidence(_get_index(), query, top_k=top_k)


def get_policy_document(source: str) -> dict:
    """
    Retrieve a complete policy document by source name.
    """
    for doc in load_policy_documents(_POLICY_DIR):
        if doc["source"] == source:
            return doc
    return {"error": f"unknown policy source: {source}"}
