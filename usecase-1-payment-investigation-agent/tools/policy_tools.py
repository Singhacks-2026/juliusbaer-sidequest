"""
Policy retrieval tools.

The agent uses these methods to obtain policy evidence rather than opening
policy files directly.  Every result preserves its source document name so the
final answer can cite it.
"""

import os

from langchain_core.tools import tool

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
    """Build the RAG index once and reuse it across all queries."""
    global _index
    if _index is None:
        documents = load_policy_documents(_POLICY_DIR)
        chunks = chunk_documents(documents)
        _index = build_index(chunks)
    return _index


@tool
def search_policy(
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """Retrieve policy evidence relevant to a natural-language query.

    Returns a list of ``{"source", "text", "score"}`` dicts, most relevant
    first, drawn from distinct policy documents.
    """
    evidence = retrieve_policy_evidence(_get_index(), query, top_k=top_k)

    return [
        {
            "source": item["source"],
            "text": item["text"],
            "score": item["score"],
        }
        for item in evidence
    ]


@tool
def get_policy_document(source: str) -> dict:
    """Retrieve a complete policy document by source filename."""
    path = os.path.join(_POLICY_DIR, os.path.basename(source))

    if not os.path.isfile(path):
        return {"error": f"Policy document not found: {source}"}

    with open(path, "r", encoding="utf-8") as file:
        return {"source": os.path.basename(source), "text": file.read()}
