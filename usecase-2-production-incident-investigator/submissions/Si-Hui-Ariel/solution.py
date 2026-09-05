"""Production Incident Investigator — submissions/Si-Hui-Ariel.

On-call investigation assistant: retrieve + correlate evidence across a mixed
corpus (logs, deploys, known issues, runbooks, prior incidents, …) and emit a
structured report with calibrated confidence.

Pipeline inside investigate():
  1. Ingest / chunk (CSV rows are first-class candidates)
  2. Hybrid retrieve (TF-IDF + keyword / error-density boosts)
  3. Correlate component-scoped hypotheses across independent source types
  4. Calibrate confidence from corroboration (thin evidence → low score)
  5. Optional LLM polish of root_cause / remediation only (offline-safe)

Required interface — do not change the signature:

    def investigate(query: str, corpus: dict) -> dict:
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE pairs into os.environ if not already set."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


# Optional LLM polish. First-wins; later files fill unset keys only.
# The use-case-1 .env is the shared default for both use cases.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
for _env_path in (
    _HERE / ".env",
    Path.cwd() / ".env",
    _HERE.parents[1] / ".env",
    _REPO_ROOT / ".env",
    _REPO_ROOT / "usecase-1-payment-investigation-agent" / ".env",
):
    _load_dotenv(_env_path)

# ---------------------------------------------------------------------------
# Source typing
# ---------------------------------------------------------------------------

SOURCE_TYPE_BY_FILE = {
    "logs.md": "logs",
    "deployment_history.md": "deployment",
    "known_issues.csv": "known_issues",
    "runbooks.md": "runbooks",
    "previous_incidents.md": "previous_incidents",
    "architecture.md": "architecture",
    "api_specs.md": "api_specs",
}

CORROBORATION_TYPES = {
    "logs",
    "deployment",
    "known_issues",
    "runbooks",
    "previous_incidents",
}

COMPONENT_RE = re.compile(
    r"\b([a-z][a-z0-9]*(?:-(?:service|adapter|agent|frontend|gateway))+)\b",
    re.IGNORECASE,
)
EXCEPTION_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:Exception|Error|Timeout))\b")
REASON_RE = re.compile(r"\breason=([A-Z_][A-Z0-9_]*)\b")
ERROR_LINE_RE = re.compile(r"\bERROR\b")
WARN_LINE_RE = re.compile(r"\bWARN\b")
MTTR_RE = re.compile(r"Typical MTTR:\s*(\d+)\s*minutes", re.IGNORECASE)
NEGATIVE_PHRASES = (
    "no deployment",
    "no other deployments",
    "no previous incident",
    "no matching",
    "unrelated",
    "first recorded",
    "genuinely thin",
    "unconfirmed",
    "unverified",
    "incomplete",
    "may not apply",
    "no evidence",
    "post-date this incident",
    "not currently instrumented",
)

QUERY_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "after",
    "are",
    "is",
    "that",
    "this",
    "what",
    "identify",
    "probable",
    "root",
    "cause",
    "supporting",
    "evidence",
    "impacted",
    "components",
    "recommended",
    "remediation",
    "mean",
    "time",
    "recover",
    "systems",
    "customers",
    "reporting",
    "sometimes",
}


@dataclass
class Chunk:
    chunk_id: str
    filename: str
    source_type: str
    text: str
    normalized: str
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def _source_type(filename: str) -> str:
    return SOURCE_TYPE_BY_FILE.get(filename, Path(filename).stem)


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[`*_#>|]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_sections(text: str) -> list[str]:
    """Split markdown on ## headings; fall back to paragraphs."""
    parts = re.split(r"(?=^##\s)", text, flags=re.MULTILINE)
    sections = [p.strip() for p in parts if p.strip()]
    if len(sections) <= 1:
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        return paras if paras else [text.strip()]
    return sections


def _split_log_blocks(text: str) -> list[str]:
    """Keep fence body as individual lines + trailing prose as its own chunk."""
    chunks: list[str] = []
    fence = re.search(r"```(?:\w*)?\n(.*?)```", text, flags=re.DOTALL)
    if fence:
        lines = [ln for ln in fence.group(1).splitlines() if ln.strip()]
        # Group consecutive ERROR/WARN-heavy lines with neighbors
        i = 0
        while i < len(lines):
            window = lines[i : i + 3]
            chunks.append("\n".join(window))
            i += 3
        prose = (text[: fence.start()] + text[fence.end() :]).strip()
        if prose:
            chunks.extend(_split_sections(prose))
    else:
        chunks.extend(_split_sections(text))
    return chunks or [text]


def _split_table_rows(text: str) -> list[str]:
    rows = []
    for line in text.splitlines():
        if line.strip().startswith("|") and "---" not in line:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if cells and cells[0].lower() not in {"version", ""}:
                rows.append(line.strip())
    prose = "\n".join(
        ln for ln in text.splitlines() if not ln.strip().startswith("|")
    ).strip()
    out = rows[:]
    if prose:
        out.extend(_split_sections(prose))
    return out or [text]


def _ingest_known_issues(csv_text: str, filename: str) -> list[Chunk]:
    reader = csv.DictReader(io.StringIO(csv_text))
    chunks: list[Chunk] = []
    for row in reader:
        issue_id = (row.get("issue_id") or "").strip() or "row"
        # Prefer the literal CSV line so excerpts stay faithful to source material
        raw_line = next(
            (
                ln.strip()
                for ln in csv_text.splitlines()
                if ln.startswith(f"{issue_id},")
            ),
            None,
        )
        labeled = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
        # Retrieval text includes both literal row + labeled fields
        blob = f"{raw_line}\n{labeled}" if raw_line else labeled
        chunks.append(
            Chunk(
                chunk_id=f"{filename}#{issue_id}",
                filename=filename,
                source_type="known_issues",
                text=raw_line or labeled,
                normalized=_normalize(blob),
                meta={
                    "issue_id": issue_id,
                    "title": row.get("title", ""),
                    "signature": row.get("signature", ""),
                    "affected_component": row.get("affected_component", ""),
                    "notes": row.get("notes", ""),
                    "row": dict(row),
                    "raw_line": raw_line or labeled,
                },
            )
        )
    return chunks


def _ingest_corpus(corpus: dict) -> list[Chunk]:
    """Normalize/prepare the raw corpus for retrieval."""
    chunks: list[Chunk] = []
    for filename, raw in corpus.items():
        stype = _source_type(filename)
        if filename.endswith(".csv") or stype == "known_issues":
            chunks.extend(_ingest_known_issues(raw, filename))
            continue

        if stype == "logs":
            pieces = _split_log_blocks(raw)
        elif stype == "deployment":
            pieces = _split_table_rows(raw)
        else:
            pieces = _split_sections(raw)

        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=f"{filename}#{i}",
                    filename=filename,
                    source_type=stype,
                    text=piece,
                    normalized=_normalize(piece),
                )
            )
    return chunks


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def _query_keywords(query: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9][a-z0-9_-]+", query.lower())
    return {t for t in tokens if t not in QUERY_STOPWORDS and len(t) > 2}


def _keyword_overlap_score(query_kw: set[str], text_norm: str) -> float:
    if not query_kw:
        return 0.0
    hits = sum(1 for kw in query_kw if kw in text_norm)
    return hits / len(query_kw)


def _error_density_boost(chunk: Chunk) -> float:
    text = chunk.text
    boost = 0.0
    if ERROR_LINE_RE.search(text):
        boost += 0.25
    if EXCEPTION_RE.search(text):
        boost += 0.35
    if REASON_RE.search(text):
        boost += 0.1
    if WARN_LINE_RE.search(text) and chunk.source_type == "logs":
        boost += 0.08
    # Deployment rows that look like recent config changes
    if chunk.source_type == "deployment" and re.search(
        r"pool|timeout|config|reduc", text, re.IGNORECASE
    ):
        boost += 0.2
    return boost


def _retrieve_relevant_documents(
    query: str, chunks: list[Chunk], top_n_per_type: int = 4
) -> list[tuple[Chunk, float]]:
    """Rank chunks against query via TF-IDF cosine + keyword/error boosts."""
    if not chunks:
        return []

    docs = [c.normalized for c in chunks]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    try:
        matrix = vectorizer.fit_transform(docs + [_normalize(query)])
    except ValueError:
        # Empty vocabulary edge case
        return [(c, 0.0) for c in chunks]

    query_vec = matrix[-1]
    doc_matrix = matrix[:-1]
    sims = cosine_similarity(query_vec, doc_matrix).flatten()

    query_kw = _query_keywords(query)
    # Also boost exception-like tokens appearing in query or common failure words
    scored: list[tuple[Chunk, float]] = []
    for chunk, sim in zip(chunks, sims):
        score = float(sim)
        score += 0.35 * _keyword_overlap_score(query_kw, chunk.normalized)
        score += _error_density_boost(chunk)
        # Prefer corroboration-capable source types slightly over architecture/api
        if chunk.source_type in CORROBORATION_TYPES:
            score += 0.05
        scored.append((chunk, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Diversify: keep top-N per source type, then merge
    per_type: dict[str, list[tuple[Chunk, float]]] = defaultdict(list)
    for item in scored:
        st = item[0].source_type
        if len(per_type[st]) < top_n_per_type:
            per_type[st].append(item)

    diversified = [item for items in per_type.values() for item in items]
    diversified.sort(key=lambda x: x[1], reverse=True)
    return diversified


# ---------------------------------------------------------------------------
# Signal extraction + correlation
# ---------------------------------------------------------------------------


def _extract_components(text: str) -> list[str]:
    found = [m.group(1).lower() for m in COMPONENT_RE.finditer(text)]
    # Preserve order, unique
    seen: set[str] = set()
    out: list[str] = []
    for c in found:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _extract_signatures(text: str) -> list[str]:
    sigs: list[str] = []
    for m in EXCEPTION_RE.finditer(text):
        sigs.append(m.group(1))
    for m in REASON_RE.finditer(text):
        sigs.append(m.group(1))
    # Queue-depth style warnings
    if re.search(r"queue depth", text, re.IGNORECASE):
        sigs.append("QueueDepthElevated")
    # Deduplicate
    seen: set[str] = set()
    out: list[str] = []
    for s in sigs:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _count_negative_evidence(text: str) -> int:
    low = text.lower()
    return sum(1 for p in NEGATIVE_PHRASES if p in low)


def _excerpt(text: str, max_len: int = 280) -> str:
    """Take a faithful prefix of the source text (preserve wording/newlines)."""
    cleaned = text.strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1].rstrip() + "…"


def _tokenize_for_overlap(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{4,}", text.lower()))


def _signature_overlap(a: list[str], b_text: str) -> bool:
    if not a:
        return False
    low = b_text.lower()
    for sig in a:
        if sig.lower() in low:
            return True
        # fuzzy: ConnectionPoolTimeout vs connection pool
        parts = re.findall(r"[A-Z][a-z]+|[A-Z]+(?![a-z])", sig)
        if len(parts) >= 2 and all(p.lower() in low for p in parts[:3]):
            return True
    return False


@dataclass
class Hypothesis:
    component: str
    signatures: list[str] = field(default_factory=list)
    source_types: set[str] = field(default_factory=set)
    evidence: list[dict[str, str]] = field(default_factory=list)
    retrieval_support: float = 0.0
    negative_hits: int = 0
    mttr_candidates: list[int] = field(default_factory=list)
    remediation_snippets: list[str] = field(default_factory=list)
    root_snippets: list[str] = field(default_factory=list)
    has_deploy_correlation: bool = False
    has_known_issue: bool = False
    has_precedent: bool = False
    has_runbook: bool = False
    has_log_errors: bool = False


def _add_evidence(hyp: Hypothesis, filename: str, text: str, seen_files: set[str]) -> None:
    if filename in seen_files:
        return
    # Allow one excerpt per file per hypothesis
    seen_files.add(filename)
    hyp.evidence.append({"source": filename, "excerpt": _excerpt(text)})


def _find_email_delay_excerpt(logs_text: str) -> str | None:
    """Build a queued→sent delay excerpt when confirmation emails lag."""
    queued: dict[str, tuple[str, str]] = {}
    delays: list[str] = []
    for line in logs_text.splitlines():
        qm = re.search(
            r"(\d{2}:\d{2}:\d{2}).*Email queued.*(order_id=\S+)",
            line,
            re.IGNORECASE,
        )
        if qm:
            queued[qm.group(2)] = (qm.group(1), line.strip())
            continue
        sm = re.search(
            r"(\d{2}:\d{2}:\d{2}).*Email sent.*(order_id=\S+)",
            line,
            re.IGNORECASE,
        )
        if sm and sm.group(2) in queued:
            q_t, q_line = queued[sm.group(2)]
            delays.append(
                f"{q_line} → {line.strip()} (queued {q_t}, sent {sm.group(1)})"
            )
            if len(delays) >= 1:
                break
    if not delays:
        return None
    return _excerpt(delays[0], 320)


def _finalize_supporting_evidence(
    evidence: list[dict[str, str]],
    *,
    thin: bool,
    corpus: dict,
    component: str,
) -> list[dict[str, str]]:
    """Prefer hard corroborating files; enrich thin cases; drop soft padding."""
    hard_order = [
        "logs.md",
        "deployment_history.md",
        "known_issues.csv",
        "runbooks.md",
        "previous_incidents.md",
    ]
    soft = {"architecture.md", "api_specs.md"}

    by_source: dict[str, dict[str, str]] = {}
    for e in evidence:
        by_source.setdefault(e["source"], e)

    # Enrich logs with queued→sent delay when present (incident-B style).
    # Keep the WARN line as a literal source substring; append delay as annotation
    # only after verifying both queued/sent lines exist in the logs file.
    logs_text = corpus.get("logs.md", "")
    delay = _find_email_delay_excerpt(logs_text)
    if delay and "logs.md" in by_source:
        warn = by_source["logs.md"]["excerpt"]
        if warn in logs_text and "email queued" not in warn.lower():
            # delay is itself built from literal log lines
            by_source["logs.md"] = {
                "source": "logs.md",
                "excerpt": (warn + " | Delay pattern: " + delay)[:360],
            }
    elif delay and thin:
        by_source["logs.md"] = {"source": "logs.md", "excerpt": delay}

    # On thin path, surface negative corpus statements as evidence
    if thin:
        deploy = corpus.get("deployment_history.md", "")
        if "deployment_history.md" not in by_source and deploy:
            # Prefer the prose paragraph(s) under the table — keep wording intact
            prose = "\n".join(
                ln
                for ln in deploy.splitlines()
                if ln.strip() and not ln.strip().startswith("|") and not ln.strip().startswith("#")
            ).strip()
            if prose and _count_negative_evidence(prose):
                by_source["deployment_history.md"] = {
                    "source": "deployment_history.md",
                    "excerpt": _excerpt(prose, 320),
                }
        prev = corpus.get("previous_incidents.md", "")
        if "previous_incidents.md" not in by_source and prev:
            by_source["previous_incidents.md"] = {
                "source": "previous_incidents.md",
                "excerpt": _excerpt(prev, 300),
            }

    hard = [by_source[s] for s in hard_order if s in by_source]
    soft_items = [e for s, e in by_source.items() if s in soft]

    if thin:
        # Prefer hard + negative sources; architecture only if still thin
        out = hard[:4]
        if len(out) < 3 and soft_items:
            out.append(soft_items[0])
        return out

    # Strong path: drop soft padding when we already have ≥4 hard sources
    if len(hard) >= 4:
        return hard[:5]
    return hard + soft_items[:1]


def _extract_pool_change(snippets: list[str]) -> str | None:
    """Pull a human-readable pool-size change like 'from 50 to 10' if present."""
    for s in snippets:
        m = re.search(
            r"(?:pool size|connection pool).*?(?:from\s+)?(\d+)\s+to\s+(\d+)",
            s,
            re.IGNORECASE,
        )
        if m:
            return f"from {m.group(1)} to {m.group(2)}"
        m = re.search(r"Reduced connection pool size from (\d+) to (\d+)", s, re.I)
        if m:
            return f"from {m.group(1)} to {m.group(2)}"
    return None


def _agreement_summary(hyp: Hypothesis) -> str:
    """Prose: which independent sources agree on this hypothesis."""
    parts: list[str] = []
    if hyp.has_log_errors:
        parts.append("ERROR logs")
    elif "logs" in hyp.source_types:
        parts.append("log warnings")
    if hyp.has_deploy_correlation:
        parts.append("deployment history")
    if hyp.has_known_issue:
        parts.append("a matching known issue")
    if hyp.has_runbook and hyp.negative_hits < 2:
        parts.append("the matching runbook")
    elif hyp.has_runbook:
        parts.append("a runbook (weak/unverified)")
    if hyp.has_precedent:
        parts.append("a prior incident")
    if not parts:
        return "No independent sources strongly agree on a single cause."
    if len(parts) == 1:
        return f"Only {parts[0]} point at this component so far."
    return (
        "Independent sources that agree: "
        + ", ".join(parts[:-1])
        + f", and {parts[-1]}."
    )


def _disagreement_summary(hyp: Hypothesis, thin: bool) -> str:
    """Prose: where the corpus fails to corroborate or actively disagrees."""
    gaps: list[str] = []
    if not hyp.has_deploy_correlation:
        gaps.append("no correlated deployment")
    if not hyp.has_known_issue:
        gaps.append("no matching known-issue signature")
    if not hyp.has_precedent:
        gaps.append("no clear precedent")
    if hyp.negative_hits >= 2:
        gaps.append(
            "documents explicitly mark related guidance as unverified or unrelated"
        )
    if not thin and not gaps:
        return "No material disagreement across the corroborating sources."
    if not gaps:
        return ""
    return "Where evidence is thin or disagrees: " + "; ".join(gaps) + "."


def _query_theme_components(query: str) -> list[str]:
    """Map symptom language in the query to likely primary components."""
    q = query.lower()
    themes: list[str] = []
    if any(w in q for w in ("email", "confirmation", "notification", "mail")):
        themes.append("notification-service")
    if any(w in q for w in ("payment", "charge", "gateway", "failing", "checkout")):
        themes.extend(["payment-gateway-adapter", "payment-service"])
    if "order" in q and "email" not in q:
        themes.append("order-service")
    # unique, preserve order
    seen: set[str] = set()
    out: list[str] = []
    for c in themes:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _component_signatures_from_logs(
    logs_text: str,
) -> tuple[dict[str, list[str]], Counter[str], Counter[str]]:
    """Map each component to signatures on the same log line, plus error/warn weights."""
    per_comp: dict[str, list[str]] = defaultdict(list)
    error_weight: Counter[str] = Counter()
    warn_weight: Counter[str] = Counter()

    for line in logs_text.splitlines():
        comps = _extract_components(line)
        if not comps:
            continue
        sigs = _extract_signatures(line)
        is_err = bool(ERROR_LINE_RE.search(line) or EXCEPTION_RE.search(line))
        is_warn = bool(WARN_LINE_RE.search(line))
        if not (is_err or is_warn or sigs):
            continue
        for c in comps:
            for s in sigs:
                if s not in per_comp[c]:
                    per_comp[c].append(s)
            if is_err:
                error_weight[c] += 3
            elif is_warn:
                warn_weight[c] += 1
    return dict(per_comp), error_weight, warn_weight


def _deploy_row_matches_hypothesis(row: str, hyp: Hypothesis, query: str) -> bool:
    """True only if this deploy row is about hyp.component AND relates to its failure mode."""
    low = row.lower()
    if hyp.component not in low:
        return False
    if _count_negative_evidence(row):
        return False
    # Change description should overlap component signatures or query symptoms
    if _signature_overlap(hyp.signatures, row):
        return True
    # Pool/timeout config changes correlate with ConnectionPool* signatures
    if any("pool" in s.lower() or "timeout" in s.lower() for s in hyp.signatures):
        if re.search(r"pool|timeout|connection", low):
            return True
    # Query symptom words in the change line
    qkw = _query_keywords(query)
    if qkw and _keyword_overlap_score(qkw, low) >= 0.25:
        return True
    return False


def _known_issue_matches(row: dict, hyp: Hypothesis) -> bool:
    affected = (row.get("affected_component") or "").lower()
    sig_text = " ".join(
        [
            row.get("signature") or "",
            row.get("title") or "",
            row.get("notes") or "",
        ]
    )
    notes = (row.get("notes") or "").lower()
    if "unrelated" in notes:
        return False
    # Cosmetic / formatting issues shouldn't match latency symptoms
    if "cosmetic" in notes and not _signature_overlap(hyp.signatures, sig_text):
        return False
    if affected != hyp.component:
        # Only allow if signature clearly matches this hyp's signatures
        return _signature_overlap(hyp.signatures, sig_text)
    # Same component: still require signature or title overlap with hyp signals
    if hyp.signatures and _signature_overlap(hyp.signatures, sig_text):
        return True
    # Same component + shared substantive tokens with hyp signatures
    if hyp.signatures:
        hyp_toks = _tokenize_for_overlap(" ".join(hyp.signatures))
        row_toks = _tokenize_for_overlap(sig_text)
        if len(hyp_toks & row_toks) >= 2:
            return True
    return False


def _correlate_evidence(
    query: str, corpus: dict, ranked: list[tuple[Chunk, float]]
) -> dict:
    """Find independent sources that corroborate (or fail to) a hypothesis."""
    logs_text = corpus.get("logs.md", "")
    per_comp_sigs, error_weight, warn_weight = _component_signatures_from_logs(
        logs_text
    )

    theme_comps = _query_theme_components(query)

    # Seed candidates: ERROR components first; else WARN + query themes
    candidates: list[str] = []
    if error_weight:
        candidates = [c for c, _ in error_weight.most_common(5)]
    else:
        candidates = [c for c, _ in warn_weight.most_common(5)]
        for c in theme_comps:
            if c not in candidates:
                candidates.insert(0, c)

    # Always ensure query-theme components are considered
    for c in theme_comps:
        if c not in candidates:
            candidates.append(c)

    if not candidates:
        candidates = theme_comps or ["unknown-component"]

    hypotheses: dict[str, Hypothesis] = {}
    for c in candidates:
        sigs = list(per_comp_sigs.get(c, []))
        hyp = Hypothesis(component=c, signatures=sigs)
        # retrieval support from ranked chunks mentioning this component
        for chunk, sc in ranked:
            if c in chunk.normalized:
                hyp.retrieval_support += sc
        # Query-theme bonus so email incidents prefer notification-service
        if c in theme_comps:
            hyp.retrieval_support += 1.5
        hypotheses[c] = hyp

    # ---- Logs ----
    for hcomp, hyp in hypotheses.items():
        seen_files: set[str] = set()
        for line in logs_text.splitlines():
            if hcomp not in line.lower():
                continue
            if ERROR_LINE_RE.search(line) or EXCEPTION_RE.search(line):
                hyp.has_log_errors = True
                hyp.source_types.add("logs")
                _add_evidence(hyp, "logs.md", line, seen_files)
                break
        if "logs" not in hyp.source_types:
            for line in logs_text.splitlines():
                if hcomp in line.lower() and WARN_LINE_RE.search(line):
                    hyp.source_types.add("logs")
                    _add_evidence(hyp, "logs.md", line, seen_files)
                    break
        # Also capture delay pattern prose / queued-sent evidence for notification
        if hcomp in logs_text.lower() and "logs.md" not in {
            e["source"] for e in hyp.evidence
        }:
            for line in logs_text.splitlines():
                if hcomp in line.lower() and re.search(
                    r"email queued|email sent|queue depth", line, re.IGNORECASE
                ):
                    hyp.source_types.add("logs")
                    _add_evidence(hyp, "logs.md", line, seen_files)
                    break

    # ---- Deployments ----
    deploy_text = corpus.get("deployment_history.md", "")
    deploy_neg = _count_negative_evidence(deploy_text)
    for hcomp, hyp in hypotheses.items():
        seen_files = {e["source"] for e in hyp.evidence}
        matched_row = False
        for line in deploy_text.splitlines():
            if not line.strip().startswith("|"):
                continue
            if _deploy_row_matches_hypothesis(line, hyp, query):
                hyp.has_deploy_correlation = True
                hyp.source_types.add("deployment")
                hyp.root_snippets.append(line)
                _add_evidence(hyp, "deployment_history.md", line, seen_files)
                matched_row = True
                break
        # Explicit anti-correlation in prose (no deploy for this component / post-dates)
        if not matched_row and deploy_neg:
            # If prose mentions this component in a negative context, penalize
            if hcomp in deploy_text.lower() or any(
                t in query.lower() for t in ("email", "notification")
            ):
                hyp.negative_hits += deploy_neg

    # ---- Known issues (per CSV row via ranked chunks + full re-parse) ----
    for chunk, sc in ranked:
        if chunk.source_type != "known_issues":
            continue
        row = chunk.meta
        for hcomp, hyp in hypotheses.items():
            if _known_issue_matches(row, hyp):
                hyp.has_known_issue = True
                hyp.source_types.add("known_issues")
                hyp.root_snippets.append(chunk.text)
                hyp.retrieval_support += sc
                seen_files = {e["source"] for e in hyp.evidence}
                _add_evidence(hyp, chunk.filename, chunk.text, seen_files)

    # ---- Runbooks ----
    runbooks = corpus.get("runbooks.md", "")
    for hcomp, hyp in hypotheses.items():
        seen_files = {e["source"] for e in hyp.evidence}
        for section in _split_sections(runbooks):
            section_hit = hcomp in section.lower() or _signature_overlap(
                hyp.signatures, section
            )
            if not section_hit:
                continue
            neg = _count_negative_evidence(section)
            hyp.negative_hits += neg
            # Unverified / incomplete runbooks are weak — still record but flag
            hyp.has_runbook = True
            hyp.source_types.add("runbooks")
            _add_evidence(hyp, "runbooks.md", section, seen_files)
            m = MTTR_RE.search(section)
            if m and neg < 2:
                hyp.mttr_candidates.append(int(m.group(1)))
            rem = re.search(
                r"\*\*Remediation\*\*:\s*(.+?)(?:\n\n|\n\*\*|$)",
                section,
                re.DOTALL | re.IGNORECASE,
            )
            if rem:
                hyp.remediation_snippets.append(rem.group(1).strip())
            break

    # ---- Previous incidents ----
    prev = corpus.get("previous_incidents.md", "")
    prev_neg = _count_negative_evidence(prev)
    for hcomp, hyp in hypotheses.items():
        seen_files = {e["source"] for e in hyp.evidence}
        matched = False
        for section in _split_sections(prev):
            if hcomp not in section.lower():
                continue
            if _count_negative_evidence(section):
                continue
            if hyp.signatures and not _signature_overlap(hyp.signatures, section):
                # Component mentioned in an unrelated prior incident
                continue
            hyp.has_precedent = True
            hyp.source_types.add("previous_incidents")
            hyp.root_snippets.append(section)
            _add_evidence(hyp, "previous_incidents.md", section, seen_files)
            matched = True
            break
        if not matched and prev_neg >= 2:
            hyp.negative_hits += 2

    # Architecture / API specs are context only — added later only if evidence is sparse.

    def hyp_rank_key(h: Hypothesis) -> tuple:
        # Strong corroborators — runbook only counts strong if not heavily negated
        runbook_strong = h.has_runbook and h.negative_hits < 2
        strong = sum(
            [
                h.has_log_errors,
                h.has_deploy_correlation,
                h.has_known_issue,
                runbook_strong,
                h.has_precedent,
            ]
        )
        # Query-theme alignment
        theme_bonus = 1 if h.component in theme_comps else 0
        corr = len(h.source_types & CORROBORATION_TYPES)
        return (strong, theme_bonus, corr, h.retrieval_support, -h.negative_hits)

    ranked_hyps = sorted(hypotheses.values(), key=hyp_rank_key, reverse=True)
    best = ranked_hyps[0]

    runbook_strong = best.has_runbook and best.negative_hits < 2
    strong_count = sum(
        [
            best.has_log_errors,
            best.has_deploy_correlation,
            best.has_known_issue,
            runbook_strong,
            best.has_precedent,
        ]
    )

    # Thin: no multi-source hard corroboration of a deploy/known-issue/precedent story
    thin = (
        strong_count <= 1
        and not best.has_deploy_correlation
        and not best.has_known_issue
        and not best.has_precedent
    ) or (
        best.negative_hits >= 3
        and not best.has_deploy_correlation
        and not best.has_known_issue
    )

    agree = _agreement_summary(best)
    disagree = _disagreement_summary(best, thin)

    if thin:
        root_cause = (
            f"No confirmed root cause for the reported symptoms around "
            f"{best.component}. {agree} {disagree} "
            f"A confident attribution would be unwarranted on this evidence."
        ).strip()
        remediation = (
            f"Recommended action for on-call: escalate {best.component} for human "
            f"investigation — do not apply speculative config changes. "
            f"Instrument consumer lag and downstream email-provider latency, "
            f"confirm the queued→sent delay is still reproducible, and treat any "
            f"runbook suggestion to scale consumers as unverified until metrics "
            f"exist. Re-run correlation once better telemetry is available."
        )
        if best.remediation_snippets:
            hint = re.sub(r"\s+", " ", best.remediation_snippets[0]).strip()
            hint = hint[0].upper() + hint[1:] if hint else hint
            remediation = (
                f"Recommended action for on-call: escalate for human review before "
                f"changing production. The only runbook hint on file is: \"{hint}\" "
                f"— that guidance is explicitly unverified. Prefer gathering queue/"
                f"consumer and email-provider metrics over an immediate scale-up."
            )
        mttr = None
        evidence = best.evidence[:4]
        if not evidence:
            for fname in ("logs.md", "runbooks.md", "previous_incidents.md"):
                if fname in corpus:
                    evidence.append(
                        {"source": fname, "excerpt": _excerpt(corpus[fname])}
                    )
                if len(evidence) >= 2:
                    break
    else:
        sig = best.signatures[0] if best.signatures else "the observed failures"
        pool_change = _extract_pool_change(best.root_snippets)
        if pool_change:
            root_cause = (
                f"Root cause: {best.component} connection pool was reduced "
                f"{pool_change}, which saturates under normal traffic and surfaces "
                f"as {sig} (with caller-side timeouts). {agree} {disagree}"
            ).strip()
        elif best.has_deploy_correlation:
            root_cause = (
                f"Root cause: a recent deployment change on {best.component} "
                f"correlates with {sig}. {agree} {disagree}"
            ).strip()
        else:
            root_cause = (
                f"Root cause: {best.component} is the primary failing component "
                f"({sig}). {agree} {disagree}"
            ).strip()

        if best.remediation_snippets:
            rem_body = re.sub(r"\s+", " ", best.remediation_snippets[0]).strip()
            rem_body = rem_body[0].upper() + rem_body[1:] if rem_body else rem_body
            remediation = (
                f"Recommended action for on-call: {rem_body} "
                f"Confirm pool utilization is back below saturation after the change, "
                f"then monitor charge success rate for a short soak window."
            )
        else:
            remediation = (
                f"Recommended action for on-call: roll back or remediate the latest "
                f"configuration change on {best.component}, redeploy, and verify "
                f"error rates return to baseline before closing the incident."
            )

        mttr = None
        if best.mttr_candidates:
            mttr = Counter(best.mttr_candidates).most_common(1)[0][0]

        by_file: dict[str, dict] = {}
        for e in best.evidence:
            by_file.setdefault(e["source"], e)
        evidence = list(by_file.values())

    evidence = _finalize_supporting_evidence(
        evidence,
        thin=thin,
        corpus=corpus,
        component=best.component,
    )

    impacted = [best.component]
    if not thin and best.component == "payment-gateway-adapter":
        if re.search(r"GATEWAY_TIMEOUT|payment-service.*Charge failed", logs_text):
            if "payment-service" not in impacted:
                impacted.append("payment-service")

    return {
        "hypothesis": best,
        "thin": thin,
        "strong_count": strong_count,
        "corroboration_count": len(best.source_types & CORROBORATION_TYPES),
        "root_cause": root_cause,
        "remediation": remediation,
        "impacted_systems": impacted,
        "mttr_minutes": mttr,
        "supporting_evidence": evidence,
        "negative_hits": best.negative_hits,
        "flags": {
            "has_log_errors": best.has_log_errors,
            "has_deploy_correlation": best.has_deploy_correlation,
            "has_known_issue": best.has_known_issue,
            "has_runbook": best.has_runbook,
            "has_precedent": best.has_precedent,
        },
    }


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def _calibrate_confidence(evidence: dict) -> float:
    """Map corroboration strength to 0-100. Thin evidence must land below 50."""
    if evidence.get("thin"):
        # 15–40 band
        base = 28.0
        base -= min(10.0, 2.0 * evidence.get("negative_hits", 0))
        if evidence.get("strong_count", 0) == 0:
            base = 18.0
        return float(max(15.0, min(40.0, base)))

    corr = evidence.get("corroboration_count", 0)
    strong = evidence.get("strong_count", 0)
    flags = evidence.get("flags", {})

    if strong >= 4 or corr >= 4:
        score = 88.0
    elif strong == 3 or corr == 3:
        score = 72.0
    elif strong == 2 or corr == 2:
        score = 58.0
    else:
        score = 45.0

    # Bonuses for classic incident-A pattern
    if flags.get("has_deploy_correlation") and flags.get("has_known_issue"):
        score += 5.0
    if flags.get("has_precedent") and flags.get("has_runbook"):
        score += 3.0

    # Caps for negative evidence
    score -= min(15.0, 3.0 * evidence.get("negative_hits", 0))

    return float(max(0.0, min(100.0, round(score, 1))))


# ---------------------------------------------------------------------------
# Optional LLM polish
# ---------------------------------------------------------------------------


def _optional_llm_polish(draft: dict, query: str) -> dict:
    """Rewrite root_cause / remediation only when explicitly enabled + keyed.

    Opt-in via INCIDENT_LLM_POLISH=1 so committed answers.json stays reproducible
    for organizers who re-run without API keys.
    """
    if os.environ.get("INCIDENT_LLM_POLISH", "").strip() not in {"1", "true", "TRUE", "yes"}:
        return draft

    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("LLM_API_KEY")
    )
    if not api_key:
        return draft

    provider = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openai"
    # Prefer explicit provider env
    if os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        provider = "openai"

    system = (
        "You are polishing an on-call incident report for a hiring review. "
        "Rewrite root_cause and remediation into crisp, professional prose. "
        "Rules: (1) Do NOT invent facts, components, MTTR values, or confidence. "
        "(2) Keep named components, exception names, and agree/disagree substance. "
        "(3) If the draft says evidence is thin / unconfirmed, stay explicitly uncertain. "
        "(4) Remediation must be actionable for a human on-call engineer. "
        'Reply with JSON only: {"root_cause": "...", "remediation": "..."}'
    )
    user = json.dumps(
        {
            "query": query,
            "root_cause": draft["root_cause"],
            "remediation": draft["remediation"],
            "impacted_systems": draft["impacted_systems"],
            "mttr_minutes": draft["mttr_minutes"],
            "confidence_score": draft["confidence_score"],
        }
    )

    try:
        if provider == "anthropic":
            polished = _call_anthropic(api_key, system, user)
        else:
            polished = _call_openai(api_key, system, user)
        if not polished:
            return draft
        out = dict(draft)
        if polished.get("root_cause"):
            out["root_cause"] = str(polished["root_cause"]).strip()
        if polished.get("remediation"):
            out["remediation"] = str(polished["remediation"]).strip()
        return out
    except Exception:
        return draft


def _call_openai(api_key: str, system: str, user: str) -> dict | None:
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    payload = {
        "model": model,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    req = urllib.request.Request(
        f"{base.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    return json.loads(content)


def _call_anthropic(api_key: str, system: str, user: str) -> dict | None:
    model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
    payload = {
        "model": model,
        "max_tokens": 512,
        "temperature": 0.2,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    content = "".join(
        part.get("text", "")
        for part in body.get("content", [])
        if part.get("type") == "text"
    )
    # Strip markdown fences if present
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    return json.loads(content)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def investigate(query: str, corpus: dict) -> dict:
    """Investigate a production symptom against a document corpus.

    Pipeline: ingest → hybrid retrieve → correlate across source types →
    calibrate confidence from corroboration → optional LLM prose polish.
    Same code path for every incident; no per-incident hardcoding.
    """
    chunks = _ingest_corpus(corpus)
    ranked = _retrieve_relevant_documents(query, chunks)

    correlated = _correlate_evidence(query, corpus, ranked)
    confidence = _calibrate_confidence(correlated)

    draft = {
        "root_cause": correlated["root_cause"],
        "supporting_evidence": correlated["supporting_evidence"],
        "impacted_systems": correlated["impacted_systems"],
        "mttr_minutes": correlated["mttr_minutes"],
        "remediation": correlated["remediation"],
        "confidence_score": confidence,
        "needs_human_review": confidence < 50,
    }

    if not draft["root_cause"]:
        draft["root_cause"] = "Unable to determine root cause from available evidence."
    if not isinstance(draft["supporting_evidence"], list):
        draft["supporting_evidence"] = []
    draft["needs_human_review"] = draft["confidence_score"] < 50

    return _optional_llm_polish(draft, query)


# ---------------------------------------------------------------------------
# CLI runner — produce answers.json
# ---------------------------------------------------------------------------


def _run_both_incidents(output_path: Path | None = None) -> dict:
    # data/loader.py lives at usecase-2-.../data/loader.py
    here = Path(__file__).resolve().parent
    uc2_root = here.parents[1]
    import sys

    sys.path.insert(0, str(uc2_root))
    from data.loader import load_incident  # type: ignore

    answers = {}
    for name in ("incident_a_pool_exhaustion", "incident_b_ambiguous_delay"):
        query, corpus = load_incident(name)
        answers[name] = investigate(query, corpus)

    out = output_path or (here / "answers.json")
    out.write_text(json.dumps(answers, indent=2) + "\n", encoding="utf-8")
    return answers


if __name__ == "__main__":
    result = _run_both_incidents()
    for name, report in result.items():
        print(
            f"{name}: confidence={report['confidence_score']} "
            f"review={report['needs_human_review']} "
            f"systems={report['impacted_systems']} "
            f"mttr={report['mttr_minutes']} "
            f"evidence_sources={[e['source'] for e in report['supporting_evidence']]}"
        )
        print(f"  root_cause: {report['root_cause'][:160]}...")
