"""
Policy retrieval tools.

Connects to ``rag/pipeline.py`` and builds the RAG index once.
"""

from __future__ import annotations

import os
from pathlib import Path

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

    Returns list of {source, text, score} dicts.
    """
    return retrieve_policy_evidence(_get_index(), query, top_k=top_k)


def get_policy_document(source: str) -> dict:
    """Retrieve a complete policy document by source name."""
    path = Path(_POLICY_DIR) / source
    if not path.is_file():
        # Allow bare stem without .md
        if not source.endswith(".md"):
            path = Path(_POLICY_DIR) / f"{source}.md"
    if not path.is_file():
        return {
            "error": "policy_not_found",
            "source": source,
            "message": f"No policy document named {source}",
        }
    return {
        "source": path.name,
        "text": path.read_text(encoding="utf-8"),
    }
