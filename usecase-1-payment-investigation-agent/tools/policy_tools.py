"""Agent-facing access to a cached local policy retrieval index."""
from functools import lru_cache
from pathlib import Path
from rag.pipeline import build_index, chunk_documents, load_policy_documents, retrieve

_POLICY_DIR = Path(__file__).resolve().parents[1] / "data" / "policies"

@lru_cache(maxsize=1)
def _get_index():
    return build_index(chunk_documents(load_policy_documents(str(_POLICY_DIR))))

def search_policy(query: str, top_k: int = 5) -> list[dict]:
    """Return up to top_k relevant passages with their source filenames."""
    return retrieve(_get_index(), query, top_k=top_k)

def get_policy_document(source: str) -> dict:
    """Return a known complete policy after discovery, or an empty dict."""
    if source != Path(source).name:
        return {}
    return next((doc for doc in load_policy_documents(str(_POLICY_DIR))
                 if doc["source"] == source), {})
