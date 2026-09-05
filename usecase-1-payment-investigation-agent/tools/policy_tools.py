"""
Policy retrieval tools.

The agent obtains policy evidence through ``search_policy`` rather than
opening files directly. The index is built once per process and reused.
"""

from __future__ import annotations

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
_documents: dict[str, str] = {}


def _get_index():
    global _index
    if _index is None:
        docs = load_policy_documents(_POLICY_DIR)
        _documents.update({d["source"]: d["text"] for d in docs})
        _index = build_index(chunk_documents(docs))
    return _index


def index_backend() -> str:
    """Name of the retrieval backend actually in use (for the audit trace)."""
    return getattr(_get_index(), "backend", "unknown")


def search_policy(query: str, top_k: int = 5) -> list[dict]:
    """
    Retrieve policy evidence relevant to a natural-language query.

    Each result carries ``source`` (filename), ``text`` (the policy rule),
    ``heading`` and ``score`` so the final answer can cite it.
    """
    results = retrieve_policy_evidence(_get_index(), query, top_k=top_k)
    return [
        {
            "source": r["source"],
            "heading": r.get("heading", ""),
            "text": r["text"],
            "score": r.get("rerank_score", r.get("score", 0.0)),
        }
        for r in results
    ]


def get_policy_document(source: str) -> dict:
    """Retrieve a complete policy document by filename."""
    _get_index()
    text = _documents.get(source)
    if text is None:
        return {"found": False, "source": source, "error": "document not found"}
    return {"found": True, "source": source, "text": text}


def list_policy_sources() -> list[str]:
    _get_index()
    return sorted(_documents.keys())
