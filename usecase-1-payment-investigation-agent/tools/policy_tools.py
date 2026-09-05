"""Cached policy retrieval and safe source lookup."""
from functools import lru_cache
from pathlib import Path

from rag.pipeline import build_index, chunk_documents, load_policy_documents, retrieve_policy_evidence

_POLICY_DIR = Path(__file__).resolve().parents[1] / "data" / "policies"


@lru_cache(maxsize=1)
def _get_index():
    return build_index(chunk_documents(load_policy_documents(str(_POLICY_DIR))))


def search_policy(query: str, top_k: int = 5) -> list[dict]:
    """Search policy passages; filenames and scores accompany every result."""
    return retrieve_policy_evidence(_get_index(), query, top_k)


def get_policy_document(source: str) -> dict:
    """Read a source identified by retrieval; reject paths outside the corpus."""
    if Path(source).name != source or not source.endswith(".md"):
        return {"error": "Invalid policy source"}
    path = _POLICY_DIR / source
    if not path.is_file() or path.resolve().parent != _POLICY_DIR.resolve():
        return {"error": "Policy source not found", "source": source}
    return {"source": source, "text": path.read_text(encoding="utf-8")}
