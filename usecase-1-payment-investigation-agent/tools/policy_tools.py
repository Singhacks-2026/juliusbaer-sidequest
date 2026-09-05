"""Cached access to policy retrieval and complete source documents."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from rag.pipeline import build_index, chunk_documents, load_policy_documents, retrieve_policy_evidence

_POLICY_DIR = Path(__file__).resolve().parents[1] / "data" / "policies"


@lru_cache(maxsize=1)
def _get_index() -> dict:
    return build_index(chunk_documents(load_policy_documents(str(_POLICY_DIR))))


def search_policy(query: str, top_k: int = 5) -> list[dict]:
    """Return relevant passages with exact source filenames."""
    return retrieve_policy_evidence(_get_index(), str(query), top_k=top_k)


def get_policy_document(source: str) -> dict:
    """Return a complete policy only when `source` is a safe known filename."""
    name = Path(str(source)).name
    path = _POLICY_DIR / name
    if name != str(source) or not path.is_file() or path.suffix != ".md":
        return {}
    return {"source": name, "text": path.read_text(encoding="utf-8")}
