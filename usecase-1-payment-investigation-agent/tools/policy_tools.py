"""
Policy retrieval tools.

The agent obtains policy evidence through these functions rather than opening
files directly, and every result keeps its source filename so the final answer
can cite it.

The RAG index is built once on first use and reused for the whole run.
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
_documents: dict[str, dict] | None = None


def _get_index():
    global _index, _documents

    if _index is None:
        documents = load_policy_documents(_POLICY_DIR)
        _documents = {document["source"]: document for document in documents}
        _index = build_index(chunk_documents(documents))

    return _index


def search_policy(query: str, top_k: int = 5) -> list[dict]:
    """
    Retrieve policy evidence relevant to a natural-language query.

    Returns ``{"source", "text", "score"}`` records, reranked so that decoy
    and non-actionable passages are excluded.
    """
    evidence = retrieve_policy_evidence(_get_index(), query, top_k=top_k)

    return [
        {
            "source": chunk["source"],
            "text": chunk["rule"],
            "heading": chunk["heading"],
            "score": chunk["score"],
        }
        for chunk in evidence
    ]


def get_policy_document(source: str) -> dict:
    """Retrieve a complete policy document by source filename."""
    _get_index()
    document = (_documents or {}).get((source or "").strip())

    if document is None:
        return {
            "found": False,
            "source": source,
            "error": f"No policy document named {source!r}.",
            "available": sorted(_documents or {}),
        }

    return {"found": True, "source": document["source"], "text": document["raw"]}
