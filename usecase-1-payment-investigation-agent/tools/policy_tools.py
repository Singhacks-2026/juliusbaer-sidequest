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
_documents_by_source = None


def _get_index():
    global _index, _documents_by_source
    if _index is None:
        docs = load_policy_documents(_POLICY_DIR)
        _documents_by_source = {doc["source"]: doc for doc in docs}
        chunks = chunk_documents(docs)
        _index = build_index(chunks)
    return _index


def search_policy(
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve policy evidence relevant to a natural-language query.

    Parameters
    ----------
    query:
        Example:
        ``"high value payment enhanced review threshold"``.

    top_k:
        Maximum number of results.

    Returns
    -------
    list[dict]
        Suggested result:

        {
            "source": "global_payment_policy.md",
            "text": "...relevant passage...",
            "score": 0.91
        }

    Implementation
    --------------
    Connect this method to ``rag/pipeline.py``.

    Build the RAG index **once** (e.g., at module level or on first call
    using a cache) and reuse it across all calls.  Do not rebuild the
    index on every query.
    """
    return retrieve_policy_evidence(_get_index(), query, top_k=top_k)


def get_policy_document(source: str) -> dict:
    """
    OPTIONAL: Retrieve a complete policy document by source name.

    Useful after the agent has already identified the relevant document.
    """
    _get_index()  # ensure _documents_by_source is populated
    return _documents_by_source.get(source, {})


def list_all_policy_documents() -> list[dict]:
    """
    Return every loaded policy document (source + full text).

    Used by deterministic evaluators (e.g. tools/threshold_tools.py) that
    need to parse rule text directly, rather than relying on top-k RAG
    retrieval to happen to surface the right document.
    """
    _get_index()
    return list(_documents_by_source.values())
