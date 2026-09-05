"""
Policy retrieval tool interfaces.

The agent should use these methods to obtain policy evidence rather than
opening policy files directly.

The implementation should preserve the source document name so that the final
assistant can cite the evidence.
"""

import os
from functools import lru_cache

_POLICY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "policies",
)


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

    Suggested wiring:

        from rag.pipeline import (
            load_policy_documents, chunk_documents, build_index, retrieve,
        )

        _index = None

        def _get_index():
            global _index
            if _index is None:
                docs = load_policy_documents(_POLICY_DIR)
                chunks = chunk_documents(docs)
                _index = build_index(chunks)
            return _index

        def search_policy(query, top_k=5):
            return retrieve(_get_index(), query, top_k)
    """
    return retrieve_policy_evidence(_get_index(), query, top_k=top_k)


def get_policy_document(source: str) -> dict:
    """
    OPTIONAL: Retrieve a complete policy document by source name.

    Useful after the agent has already identified the relevant document.
    """
    safe_source = os.path.basename(source)
    if safe_source != source or not safe_source.endswith(".md"):
        return {}
    path = os.path.join(_POLICY_DIR, safe_source)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return {"source": safe_source, "text": handle.read()}


@lru_cache(maxsize=1)
def _get_index():
    from rag.pipeline import build_index, chunk_documents, load_policy_documents
    return build_index(chunk_documents(load_policy_documents(_POLICY_DIR)))


from rag.pipeline import retrieve_policy_evidence
