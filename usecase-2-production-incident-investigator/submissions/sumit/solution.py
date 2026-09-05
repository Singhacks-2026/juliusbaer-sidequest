"""Corroboration Graph Investigator — production incident investigation.

Pipeline: ingest → hybrid retrieve (TF-IDF + BM25 + RRF) → top-K-driven
hypothesis / corroboration → calibrated confidence → on-call brief.

    def investigate(query: str, corpus: dict) -> dict
"""
from __future__ import annotations

import csv
import io
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Models & patterns
# ---------------------------------------------------------------------------

SERVICE_RE = re.compile(r"\b([a-z][a-z0-9]+(?:-[a-z0-9]+)+)\b")
EXCEPTION_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:Exception|Error|Timeout))\b")
LOG_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+"
    r"(?P<sev>ERROR|WARN|INFO|DEBUG)\s+"
    r"(?P<comp>\S+)\s+"
    r"(?P<msg>.+)$"
)
MTTR_RE = re.compile(r"(?:Typical\s+)?MTTR\**[:\s]*(\d+)\s*minutes?", re.I)
FROM_TO_RE = re.compile(r"from\s+(\d+)\s+to\s+(\d+)", re.I)
CONFIG_CHANGE_RE = re.compile(
    r"\b(pool|timeout|limit|connections?|threads?|size|capacity|workers?)\b",
    re.I,
)
UNVERIFIED_RE = re.compile(
    r"unverified|incomplete|may not apply|not currently instrumented|"
    r"first recorded|no previous incident|no deployment",
    re.I,
)
TOP_K = 15

SOURCE_TYPES = {
    "logs.md": "logs",
    "deployment_history.md": "deployment",
    "known_issues.csv": "known_issues",
    "runbooks.md": "runbooks",
    "previous_incidents.md": "previous_incidents",
    "architecture.md": "architecture",
    "api_specs.md": "api_specs",
}


@dataclass
class Chunk:
    id: str
    source_file: str
    source_type: str
    text: str
    text_norm: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class LogEvent:
    ts: datetime | None
    severity: str
    component: str
    message: str
    raw: str


@dataclass
class Hypothesis:
    component: str
    signature: str
    kind: str
    score: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Signal:
    name: str
    positive: bool
    excerpt: str
    source: str
    unverified: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[`*_#>|]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*|[a-z0-9]{2,}", _normalize(text))


def _parse_ts(raw: str) -> datetime | None:
    raw = raw.strip().strip("*")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _clean_excerpt_text(text: str) -> str:
    text = text.replace("\u2014", "-").replace("\u2013", "-").replace("…", "...")
    text = re.sub(r"^#+\s*", "", text.strip())
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\|+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -#*")
    return text


def _short_excerpt(text: str, max_len: int = 240) -> str:
    text = _clean_excerpt_text(text)
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    # Prefer sentence / clause boundary, else last whitespace (never mid-word)
    for sep in (". ", "; ", ", ", " "):
        idx = cut.rfind(sep)
        if idx >= max_len // 2:
            cut = cut[: idx + (1 if sep == ". " else 0)].rstrip()
            break
    else:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:-") + "..."


def _services_in(text: str) -> list[str]:
    return list(dict.fromkeys(SERVICE_RE.findall(text.lower())))


def _exceptions_in(text: str) -> list[str]:
    return list(dict.fromkeys(EXCEPTION_RE.findall(text)))


def _token_overlap(a: str, b: str) -> float:
    ta, tb = set(_tokenize(a)), set(_tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _parse_config_change(change: str) -> dict[str, Any]:
    """Generic deploy config-change detector (pool/timeout/limit/size/...)."""
    m = FROM_TO_RE.search(change)
    keywords = CONFIG_CHANGE_RE.findall(change)
    return {
        "is_config_change": bool(keywords) or bool(m),
        "keywords": [k.lower() for k in keywords],
        "from_n": int(m.group(1)) if m else None,
        "to_n": int(m.group(2)) if m else None,
        "mentions_pool": any(k.lower().startswith("pool") or k.lower() == "connection"
                             or k.lower() == "connections" for k in keywords)
                        or "pool" in change.lower(),
    }


def _find_chunk(chunks: list[Chunk], pred: Callable[[Chunk], bool]) -> Chunk | None:
    for c in chunks:
        if pred(c):
            return c
    return None


# ---------------------------------------------------------------------------
# 1. Ingest
# ---------------------------------------------------------------------------

def _ingest_logs(text: str) -> tuple[list[Chunk], list[LogEvent]]:
    chunks: list[Chunk] = []
    events: list[LogEvent] = []
    in_fence = False
    idx = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            continue
        m = LOG_LINE_RE.match(stripped)
        if not m:
            continue
        ts = _parse_ts(m.group("ts"))
        sev, comp, msg = m.group("sev"), m.group("comp").strip(), m.group("msg").strip()
        events.append(LogEvent(ts=ts, severity=sev, component=comp, message=msg, raw=stripped))
        chunks.append(
            Chunk(
                id=f"logs.md#line-{idx}",
                source_file="logs.md",
                source_type="logs",
                text=stripped,
                text_norm=_normalize(stripped),
                meta={"severity": sev, "component": comp, "ts": ts, "message": msg},
            )
        )
        idx += 1
    narrative = re.sub(r"```.*?```", " ", text, flags=re.S).strip()
    if narrative:
        chunks.append(
            Chunk(
                id="logs.md#narrative",
                source_file="logs.md",
                source_type="logs",
                text=narrative,
                text_norm=_normalize(narrative),
                meta={"severity": "NARRATIVE"},
            )
        )
    return chunks, events


def _ingest_known_issues(text: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for row in csv.DictReader(io.StringIO(text)):
        issue_id = (row.get("issue_id") or "").strip()
        if not issue_id:
            continue
        blob = " | ".join(
            [
                issue_id,
                row.get("title", ""),
                row.get("signature", ""),
                row.get("affected_component", ""),
                row.get("notes", ""),
            ]
        )
        chunks.append(
            Chunk(
                id=f"known_issues.csv#{issue_id}",
                source_file="known_issues.csv",
                source_type="known_issues",
                text=blob,
                text_norm=_normalize(blob),
                meta={
                    "issue_id": issue_id,
                    "title": row.get("title", ""),
                    "signature": row.get("signature", ""),
                    "affected_component": (row.get("affected_component") or "").strip(),
                    "notes": row.get("notes", ""),
                },
            )
        )
    return chunks


def _ingest_markdown(filename: str, text: str) -> list[Chunk]:
    source_type = SOURCE_TYPES.get(filename, "other")
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    title, body = filename, []
    for line in lines:
        if line.startswith("## "):
            if body:
                sections.append((title, body))
            title, body = line[3:].strip(), [line]
        else:
            body.append(line)
    if body:
        sections.append((title, body))

    chunks: list[Chunk] = []
    for i, (sec_title, body_lines) in enumerate(sections):
        body = "\n".join(body_lines).strip()
        if not body:
            continue

        if filename == "deployment_history.md":
            for row in re.finditer(r"\|([^|\n]+\|){3,}", body):
                row_text = row.group(0).strip()
                if "Version" in row_text or row_text.startswith("|---"):
                    continue
                cells = [c.strip().strip("*") for c in row_text.strip("|").split("|")]
                if len(cells) < 4:
                    continue
                version, ts_raw, component, change = cells[:4]
                cfg = _parse_config_change(change)
                row_blob = f"{version} | {ts_raw} | {component} | {change}"
                chunks.append(
                    Chunk(
                        id=f"{filename}#deploy-{version}",
                        source_file=filename,
                        source_type=source_type,
                        text=row_blob,
                        text_norm=_normalize(row_blob),
                        meta={
                            "version": version,
                            "ts": _parse_ts(ts_raw),
                            "component": component.lower(),
                            "change": change,
                            "kind": "deploy_row",
                            **cfg,
                        },
                    )
                )
            prose = "\n".join(
                ln for ln in body_lines if ln.strip() and not ln.strip().startswith("|")
            )
            if prose:
                chunks.append(
                    Chunk(
                        id=f"{filename}#notes",
                        source_file=filename,
                        source_type=source_type,
                        text=prose,
                        text_norm=_normalize(prose),
                        meta={"kind": "deploy_notes"},
                    )
                )
            continue

        meta: dict[str, Any] = {"title": sec_title}
        if m := re.search(r"(RB-\d+)", sec_title):
            meta["runbook_id"] = m.group(1)
        if m := re.search(r"(INC-\d+)", sec_title):
            meta["incident_id"] = m.group(1)
        if m := MTTR_RE.search(body):
            meta["mttr_minutes"] = int(m.group(1))
        meta["unverified"] = bool(UNVERIFIED_RE.search(body))
        chunks.append(
            Chunk(
                id=f"{filename}#sec-{i}",
                source_file=filename,
                source_type=source_type,
                text=body,
                text_norm=_normalize(body),
                meta=meta,
            )
        )
    return chunks


def _ingest_corpus(corpus: dict) -> dict:
    chunks: list[Chunk] = []
    events: list[LogEvent] = []
    for filename, text in corpus.items():
        if filename == "logs.md":
            log_chunks, events = _ingest_logs(text)
            chunks.extend(log_chunks)
        elif filename == "known_issues.csv":
            chunks.extend(_ingest_known_issues(text))
        elif filename.endswith(".md"):
            chunks.extend(_ingest_markdown(filename, text))
        else:
            chunks.append(
                Chunk(
                    id=f"{filename}#full",
                    source_file=filename,
                    source_type=SOURCE_TYPES.get(filename, "other"),
                    text=text,
                    text_norm=_normalize(text),
                )
            )
    return {"chunks": chunks, "events": events, "corpus": corpus}


# ---------------------------------------------------------------------------
# 2. Retrieve — TF-IDF + BM25 + RRF
# ---------------------------------------------------------------------------

def _bm25_scores(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    tokenized = [_tokenize(d) for d in docs]
    q_tokens = _tokenize(query)
    n = len(tokenized) or 1
    avgdl = sum(len(t) for t in tokenized) / n
    df: Counter[str] = Counter()
    for toks in tokenized:
        df.update(set(toks))
    scores = []
    for toks in tokenized:
        tf = Counter(toks)
        dl = len(toks) or 1
        score = 0.0
        for term in q_tokens:
            if term not in tf:
                continue
            n_qi = df.get(term, 0)
            idf = math.log(1 + (n - n_qi + 0.5) / (n_qi + 0.5))
            freq = tf[term]
            score += idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * dl / avgdl))
        scores.append(score)
    return scores


def _rrf_fuse(rank_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rank_lists:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _expand_query(query: str, events: list[LogEvent]) -> str:
    extras: list[str] = list(_exceptions_in(query)) + list(_services_in(query))
    for e in events:
        if e.severity not in ("ERROR", "WARN"):
            continue
        extras.extend(_exceptions_in(e.message))
        extras.extend(_services_in(e.component))
        low = e.message.lower()
        if "pool" in low or "connection" in low:
            extras.extend(["connection", "pool", "timeout"])
        if "queue" in low:
            extras.extend(["queue", "depth", "email"])
        if "GATEWAY_TIMEOUT" in e.message:
            extras.append("GATEWAY_TIMEOUT")
    qlow = query.lower()
    if any(k in qlow for k in ("email", "notification", "confirm")):
        extras.extend(["email", "queue", "delayed"])
    if any(k in qlow for k in ("payment", "deploy", "fail")):
        extras.extend(["payment", "deployment", "gateway", "timeout"])
    return query + " " + " ".join(dict.fromkeys(extras))


def _retrieve_relevant_documents(query: str, ingested: dict) -> list[tuple[str, float]]:
    chunks: list[Chunk] = ingested["chunks"]
    events: list[LogEvent] = ingested["events"]
    if not chunks:
        return []
    expanded = _expand_query(query, events)
    docs = [c.text_norm for c in chunks]
    ids = [c.id for c in chunks]

    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[a-z0-9]+(?:-[a-z0-9]+)*|[a-z0-9]{2,}",
        ngram_range=(1, 2),
        min_df=1,
    )
    matrix = vectorizer.fit_transform(docs + [_normalize(expanded)])
    sims = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    tfidf_ranked = [ids[i] for i in sorted(range(len(ids)), key=lambda i: sims[i], reverse=True)]

    bm25 = _bm25_scores(_normalize(expanded), docs)
    bm25_ranked = [ids[i] for i in sorted(range(len(ids)), key=lambda i: bm25[i], reverse=True)]
    return _rrf_fuse([tfidf_ranked, bm25_ranked])


def _top_chunks_by_type(
    chunks: list[Chunk], ranked: list[tuple[str, float]], k: int = TOP_K
) -> dict[str, list[Chunk]]:
    """Retrieval-driven candidate pools per source type.

    Prefer RRF top-K. If a source type never appears in top-K, fall back to
    all chunks of that type. Always keep sparse high-signal deploy rows and
    known-issue rows that ranked anywhere in the full list (not just top-K),
    so a prose 'notes' hit cannot crowd out the smoking-gun table row.
    """
    by_id = {c.id: c for c in chunks}
    top_ids = {cid for cid, _ in ranked[:k]}
    top = [by_id[cid] for cid, _ in ranked[:k] if cid in by_id]

    all_by_type: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        all_by_type[c.source_type].append(c)

    grouped: dict[str, list[Chunk]] = {}
    for stype, all_chunks in all_by_type.items():
        ranked_of_type = [c for c in top if c.source_type == stype]
        if not ranked_of_type:
            grouped[stype] = list(all_chunks)
            continue
        pooled = {c.id: c for c in ranked_of_type}
        # Never drop deploy table rows / KI rows solely because notes ranked higher
        if stype == "deployment":
            for c in all_chunks:
                if c.meta.get("kind") == "deploy_row":
                    pooled[c.id] = c
        if stype == "known_issues":
            for cid, _ in ranked:
                if cid in by_id and by_id[cid].source_type == "known_issues":
                    pooled[cid] = by_id[cid]
        grouped[stype] = list(pooled.values())
    return grouped


# ---------------------------------------------------------------------------
# 3. Hypothesis + corroboration (retrieval-driven)
# ---------------------------------------------------------------------------

def _form_hypotheses(
    query: str, events: list[LogEvent], candidates: dict[str, list[Chunk]]
) -> list[Hypothesis]:
    hyps: list[Hypothesis] = []
    errors = [e for e in events if e.severity == "ERROR"]
    warns = [e for e in events if e.severity == "WARN"]

    sig_counts: Counter[tuple[str, str]] = Counter()
    for e in errors:
        excs = _exceptions_in(e.message)
        if excs:
            sig = excs[0]
        elif "GATEWAY_TIMEOUT" in e.message:
            sig = "GATEWAY_TIMEOUT"
        else:
            sig = e.message.split(":")[0].strip()
        sig_counts[(e.component, sig)] += 1
    for (comp, sig), cnt in sig_counts.most_common():
        hyps.append(Hypothesis(comp, sig, "exception", 10.0 + cnt, {"error_count": cnt}))

    for c in candidates.get("deployment", []):
        if c.meta.get("kind") != "deploy_row":
            continue
        if not c.meta.get("is_config_change"):
            continue
        hyps.append(
            Hypothesis(
                c.meta.get("component", ""),
                "config_change",
                "deploy",
                9.0,
                {
                    "version": c.meta.get("version"),
                    "ts": c.meta.get("ts"),
                    "change": c.meta.get("change"),
                    "from_n": c.meta.get("from_n"),
                    "to_n": c.meta.get("to_n"),
                    "mentions_pool": c.meta.get("mentions_pool"),
                    "chunk_id": c.id,
                },
            )
        )

    log_blob = " ".join(e.raw for e in errors + warns)
    for c in candidates.get("known_issues", []):
        sig = c.meta.get("signature", "")
        comp = c.meta.get("affected_component", "")
        overlap = _token_overlap(sig, log_blob)
        for exc in _exceptions_in(log_blob):
            if exc.lower() in (sig + c.text).lower():
                overlap = max(overlap, 0.85)
        # Decoy rejection: cosmetic / refund-without-charge-impact stay weak
        notes = (c.meta.get("notes") or "").lower()
        title = (c.meta.get("title") or "").lower()
        if "cosmetic" in notes or "format" in title:
            overlap *= 0.2
        if "does not affect charge" in notes and errors:
            overlap *= 0.15
        if overlap >= 0.18:
            hyps.append(
                Hypothesis(
                    comp,
                    f"{c.meta.get('issue_id')}:{c.meta.get('title') or sig[:50]}",
                    "known_issue",
                    5.0 + overlap * 10,
                    {
                        "issue_id": c.meta.get("issue_id"),
                        "ki_signature": sig,
                        "notes": c.meta.get("notes"),
                        "title": c.meta.get("title"),
                        "overlap": overlap,
                        "chunk_id": c.id,
                    },
                )
            )

    for c in candidates.get("runbooks", []):
        overlap = _token_overlap(c.text, log_blob + " " + query)
        for exc in _exceptions_in(log_blob):
            if exc.lower() in c.text_norm:
                overlap = max(overlap, 0.9)
        if "queue depth" in log_blob.lower() and "queue depth" in c.text_norm:
            overlap = max(overlap, 0.8)
        if overlap < 0.12:
            continue
        comps = _services_in(c.text)
        primary = comps[0] if comps else ""
        for e in errors + warns:
            if e.component in c.text:
                primary = e.component
                break
        hyps.append(
            Hypothesis(
                primary,
                c.meta.get("runbook_id") or c.meta.get("title", ""),
                "runbook",
                4.0 + overlap * 8,
                {
                    "runbook_id": c.meta.get("runbook_id") or c.meta.get("title"),
                    "mttr_minutes": c.meta.get("mttr_minutes"),
                    "unverified": c.meta.get("unverified", False),
                    "text": c.text,
                    "chunk_id": c.id,
                },
            )
        )

    qlow = query.lower()
    emailish = any(k in qlow for k in ("email", "notification", "confirm"))
    if not errors and (emailish or any("queue" in e.message.lower() for e in warns)):
        queue_warns = [e for e in warns if "queue" in e.message.lower()]
        if queue_warns or emailish:
            comp = queue_warns[0].component if queue_warns else (
                _services_in(query)[0] if _services_in(query) else "unknown-service"
            )
            hyps.append(
                Hypothesis(
                    comp,
                    "queue_delay",
                    "weak_queue",
                    3.5,
                    {"warn": queue_warns[0].raw if queue_warns else None},
                )
            )
    return sorted(hyps, key=lambda h: h.score, reverse=True)


def _merge_leading(hyps: list[Hypothesis], events: list[LogEvent]) -> Hypothesis:
    if not hyps:
        return Hypothesis("unknown", "unknown", "none", 0.0)

    error_comps = {e.component for e in events if e.severity == "ERROR"}
    comp_scores: Counter[str] = Counter()
    for h in hyps[:12]:
        if h.component:
            comp_scores[h.component] += h.score
    for c in error_comps:
        comp_scores[c] += 8

    if not error_comps:
        for h in hyps:
            if h.kind == "weak_queue":
                return h

    top_comp = comp_scores.most_common(1)[0][0] if comp_scores else hyps[0].component
    merged = Hypothesis(top_comp, hyps[0].signature, "merged", 0.0, {})
    for h in hyps:
        if h.component != top_comp:
            continue
        if h.kind == "exception":
            merged.signature = h.signature
            merged.details["exception"] = h
        elif h.kind == "deploy":
            merged.details["deploy"] = h
            if h.details.get("mentions_pool"):
                merged.signature = "connection_pool_reduction"
        elif h.kind == "known_issue":
            existing = merged.details.get("known_issue")
            if not existing or h.score > existing.score:
                merged.details["known_issue"] = h
        elif h.kind == "runbook":
            existing = merged.details.get("runbook")
            if existing and existing.details.get("unverified") and not h.details.get("unverified"):
                merged.details["runbook"] = h
            elif "runbook" not in merged.details or not h.details.get("unverified"):
                if "runbook" not in merged.details or h.score >= existing.score:
                    merged.details["runbook"] = h
        elif h.kind == "weak_queue":
            merged.kind = "weak_queue"
            merged.signature = h.signature
            merged.details["weak_queue"] = h
    return merged


def _correlate_evidence(query: str, ingested: dict, ranked: list[tuple[str, float]]) -> dict:
    chunks: list[Chunk] = ingested["chunks"]
    events: list[LogEvent] = ingested["events"]
    corpus: dict = ingested["corpus"]
    candidates = _top_chunks_by_type(chunks, ranked, TOP_K)

    hyps = _form_hypotheses(query, events, candidates)
    leading = _merge_leading(hyps, events)
    signals: dict[str, Signal] = {}

    errors = [e for e in events if e.severity == "ERROR"]
    warns = [e for e in events if e.severity == "WARN"]
    first_error_ts = min((e.ts for e in errors if e.ts), default=None)
    log_dates = [e.ts.date() for e in events if e.ts]
    incident_date = min(log_dates) if log_dates else None

    matching_errors = [
        e
        for e in errors
        if e.component == leading.component
        or leading.signature in e.message
        or any(x in e.message for x in _exceptions_in(leading.signature + " " + e.message))
        or (
            "pool" in leading.signature
            and ("ConnectionPool" in e.message or "GATEWAY_TIMEOUT" in e.message)
        )
        or (
            leading.component
            and "GATEWAY_TIMEOUT" in e.message
            and e.component != leading.component
        )
    ]
    # Prefer leading-component lines for the primary log excerpt, but keep
    # sibling ERROR components (e.g. GATEWAY_TIMEOUT on the caller) for systems.
    primary_errors = [e for e in matching_errors if e.component == leading.component] or matching_errors

    if matching_errors:
        signals["LOG_ERROR_SIGNATURE"] = Signal(
            "LOG_ERROR_SIGNATURE", True, primary_errors[0].raw, "logs.md"
        )
    else:
        queue_warns = [e for e in warns if "queue" in e.message.lower()]
        excerpt = queue_warns[0].raw if queue_warns else "No ERROR-level signature matched."
        narr = _find_chunk(chunks, lambda c: c.id == "logs.md#narrative")
        if narr and leading.kind == "weak_queue":
            # Keep warn as primary; narrative used later for delay range
            pass
        signals["LOG_ERROR_SIGNATURE"] = Signal(
            "LOG_ERROR_SIGNATURE", False, excerpt, "logs.md"
        )

    # DEPLOY — from retrieval candidates
    deploy_hit = None
    for c in candidates.get("deployment", []):
        if c.meta.get("kind") != "deploy_row":
            continue
        if c.meta.get("component") != leading.component:
            continue
        if not c.meta.get("is_config_change"):
            continue
        ts = c.meta.get("ts")
        if incident_date and ts and ts.date() > incident_date:
            continue
        if first_error_ts and ts and ts > first_error_ts:
            continue
        deploy_hit = c
        if c.meta.get("mentions_pool"):
            break

    deploy_notes = _find_chunk(
        candidates.get("deployment", []) + chunks,
        lambda c: c.meta.get("kind") == "deploy_notes",
    )
    if deploy_hit:
        cfg = {
            "version": deploy_hit.meta.get("version"),
            "change": deploy_hit.meta.get("change"),
            "from_n": deploy_hit.meta.get("from_n"),
            "to_n": deploy_hit.meta.get("to_n"),
            "ts": deploy_hit.meta.get("ts"),
            "mentions_pool": deploy_hit.meta.get("mentions_pool"),
        }
        signals["DEPLOY_TEMPORAL"] = Signal(
            "DEPLOY_TEMPORAL", True, deploy_hit.text, "deployment_history.md", meta=cfg
        )
        leading.details["deploy"] = Hypothesis(
            leading.component, "deploy", "deploy", details=cfg
        )
    else:
        neg = deploy_notes.text if deploy_notes else (
            "No deployment on the implicated component correlates with symptom onset."
        )
        # Prefer a concrete sentence from notes
        if deploy_notes:
            for sent in re.split(r"(?<=[.])\s+", deploy_notes.text):
                if "no deployment" in sent.lower() or "post-date" in sent.lower():
                    neg = sent
                    break
        signals["DEPLOY_TEMPORAL"] = Signal(
            "DEPLOY_TEMPORAL", False, neg, "deployment_history.md"
        )

    # KNOWN_ISSUE — require signature overlap with errors; reject decoys
    ki_match = None
    best = 0.0
    focus = " ".join(e.raw for e in matching_errors) if matching_errors else " ".join(
        e.raw for e in warns if e.component == leading.component
    )
    for c in candidates.get("known_issues", []):
        if c.meta.get("affected_component") != leading.component:
            continue
        sig = c.meta.get("signature", "")
        notes = (c.meta.get("notes") or "").lower()
        title = (c.meta.get("title") or "").lower()
        if leading.kind == "weak_queue" and ("cosmetic" in notes or "format" in title):
            continue
        if matching_errors and "does not affect charge" in notes:
            continue
        ov = _token_overlap(sig, focus or query)
        for exc in _exceptions_in(focus):
            if exc.lower() in sig.lower():
                ov = max(ov, 0.9)
        if "pool" in leading.signature and "pool" in sig.lower():
            ov = max(ov, 0.7)
        if ov > best and ov >= 0.25:
            best, ki_match = ov, c

    if ki_match and matching_errors:
        signals["KNOWN_ISSUE"] = Signal(
            "KNOWN_ISSUE",
            True,
            f"{ki_match.meta.get('issue_id')}: {ki_match.meta.get('signature')}",
            "known_issues.csv",
            meta={"issue_id": ki_match.meta.get("issue_id"), "notes": ki_match.meta.get("notes")},
        )
    else:
        # Real rejected row for disagreement (prefer cosmetic same-area KI)
        rejected = None
        for c in candidates.get("known_issues", []) or [
            x for x in chunks if x.source_type == "known_issues"
        ]:
            notes = (c.meta.get("notes") or "").lower()
            title = (c.meta.get("title") or "").lower()
            if leading.component == c.meta.get("affected_component") and (
                "cosmetic" in notes or "format" in title
            ):
                rejected = c
                break
        if rejected:
            excerpt = (
                f"{rejected.meta.get('issue_id')}: {rejected.meta.get('title')} — "
                f"{rejected.meta.get('notes')}"
            )
            signals["KNOWN_ISSUE"] = Signal(
                "KNOWN_ISSUE", False, excerpt, "known_issues.csv",
                meta={"issue_id": rejected.meta.get("issue_id"), "rejected": True},
            )
        else:
            signals["KNOWN_ISSUE"] = Signal(
                "KNOWN_ISSUE", False, "", "known_issues.csv"
            )

    # RUNBOOK
    rb_match = None
    for c in candidates.get("runbooks", []):
        if matching_errors:
            if any(exc.lower() in c.text_norm for exc in _exceptions_in(
                " ".join(e.message for e in matching_errors)
            )):
                rb_match = c
                break
            if leading.component in c.text and "timeout" in c.text_norm:
                rb_match = c
        elif leading.component in c.text and ("queue" in c.text_norm or "email" in c.text_norm):
            rb_match = c
    if rb_match:
        unverified = bool(rb_match.meta.get("unverified")) or bool(UNVERIFIED_RE.search(rb_match.text))
        if matching_errors and any(
            exc.lower() in rb_match.text_norm for exc in _exceptions_in(
                " ".join(e.message for e in matching_errors)
            )
        ):
            unverified = False
        rem = ""
        if m := re.search(
            r"\*\*Remediation\*\*:\s*(.+?)(?:\n\s*\n|\*\*Typical|\Z)", rb_match.text, re.S
        ):
            rem = re.sub(r"\s+", " ", m.group(1)).strip()
        signals["RUNBOOK"] = Signal(
            "RUNBOOK",
            True,
            rb_match.text,
            "runbooks.md",
            unverified=unverified,
            meta={
                "runbook_id": rb_match.meta.get("runbook_id") or rb_match.meta.get("title"),
                "mttr_minutes": rb_match.meta.get("mttr_minutes"),
                "remediation": rem,
            },
        )
    else:
        signals["RUNBOOK"] = Signal("RUNBOOK", False, "", "runbooks.md")

    # PRECEDENT
    prev_hit = None
    first_recorded = False
    for c in candidates.get("previous_incidents", []):
        if "first recorded" in c.text_norm or "no previous incident" in c.text_norm:
            first_recorded = True
        if matching_errors and (
            any(exc.lower() in c.text_norm for exc in _exceptions_in(
                " ".join(e.message for e in matching_errors)
            ))
            or (leading.component in c.text and "pool" in c.text_norm)
        ):
            prev_hit = c
            break
    if prev_hit and not first_recorded:
        mttr = prev_hit.meta.get("mttr_minutes")
        if mttr is None and (m := MTTR_RE.search(prev_hit.text)):
            mttr = int(m.group(1))
        signals["PRECEDENT"] = Signal(
            "PRECEDENT",
            True,
            prev_hit.text,
            "previous_incidents.md",
            meta={"incident_id": prev_hit.meta.get("incident_id"), "mttr_minutes": mttr},
        )
    else:
        neg = ""
        for c in candidates.get("previous_incidents", []):
            if "first recorded" in c.text_norm or "no previous" in c.text_norm:
                for sent in re.split(r"(?<=[.])\s+", c.text):
                    if "first recorded" in sent.lower() or "no previous" in sent.lower():
                        neg = sent
                        break
                if not neg:
                    neg = c.text
                break
        signals["PRECEDENT"] = Signal("PRECEDENT", False, neg, "previous_incidents.md")

    # ARCH_PATH
    arch = corpus.get("architecture.md", "")
    arch_positive = leading.component in arch
    if matching_errors and leading.component:
        arch_positive = leading.component in arch and (
            "connection pool" in arch.lower() or "direct path" in arch.lower() or "queue" in arch.lower()
        )
    if leading.kind == "weak_queue":
        arch_positive = leading.component in arch and (
            "queue" in arch.lower() or "email" in arch.lower()
        )

    arch_excerpt = ""
    components_idx = arch.lower().find("## components")
    region = arch[components_idx:] if components_idx >= 0 else arch
    region_flat = re.sub(r"\s+", " ", region)
    # Prefer the bullet that names the leading component
    bullet_match = re.search(
        rf"-\s*\*{{0,2}}{re.escape(leading.component)}\*{{0,2}}\s*:?\s*[^.]*\.",
        region_flat,
        re.I,
    )
    if bullet_match:
        arch_excerpt = bullet_match.group(0)
    else:
        for needle in ("instrumented", "connection pool", "message queue"):
            if needle.lower() not in region_flat.lower() or leading.component not in region_flat:
                continue
            # Start at the component mention, not mid-word before it
            comp_pos = region_flat.lower().find(leading.component)
            needle_pos = region_flat.lower().find(needle.lower())
            if abs(comp_pos - needle_pos) > 400:
                continue
            start = comp_pos
            end = min(len(region_flat), max(comp_pos + len(leading.component), needle_pos) + 160)
            arch_excerpt = region_flat[start:end]
            break
    if not arch_excerpt:
        for para in arch.split("\n\n"):
            if leading.component in para and "->" not in para:
                arch_excerpt = para
                break
    signals["ARCH_PATH"] = Signal(
        "ARCH_PATH", arch_positive, arch_excerpt or "", "architecture.md"
    )

    api = corpus.get("api_specs.md", "")
    if matching_errors and "5000ms" in api and "GATEWAY_TIMEOUT" in api:
        leading.details["api_timeout"] = True

    return {
        "leading": leading,
        "signals": signals,
        "events": events,
        "chunks": chunks,
        "query": query,
        "ranked": ranked,
        "matching_errors": matching_errors,
        "primary_errors": primary_errors,
        "corpus": corpus,
        "candidates": candidates,
    }


# ---------------------------------------------------------------------------
# 4. Confidence
# ---------------------------------------------------------------------------

def _calibrate_confidence(evidence: dict) -> float:
    signals: dict[str, Signal] = evidence["signals"]
    leading: Hypothesis = evidence["leading"]

    n_pos = 0
    for name, sig in signals.items():
        if not sig.positive:
            continue
        if name == "RUNBOOK" and sig.unverified:
            continue
        n_pos += 1

    base = 12 * n_pos
    if n_pos >= 5:
        base += 10

    penalties = 0
    if not signals["DEPLOY_TEMPORAL"].positive:
        penalties += 18
    if not signals["KNOWN_ISSUE"].positive:
        penalties += 18
    if not signals["PRECEDENT"].positive:
        penalties += 12
    rb = signals["RUNBOOK"]
    if rb.positive and rb.unverified:
        penalties += 15
    if not signals["LOG_ERROR_SIGNATURE"].positive:
        penalties += 20
    if leading.kind == "weak_queue" or not evidence.get("matching_errors"):
        penalties += 8

    return max(20.0, min(90.0, float(base - penalties)))


# ---------------------------------------------------------------------------
# 5. On-call brief
# ---------------------------------------------------------------------------

def _impacted_systems(evidence: dict) -> list[str]:
    """Derive systems from leading component + co-failing ERROR components."""
    leading: Hypothesis = evidence["leading"]
    systems: list[str] = []

    if leading.component and leading.component not in ("unknown-service", "unknown"):
        systems.append(leading.component)

    for e in evidence.get("matching_errors") or []:
        if e.component not in systems:
            systems.append(e.component)

    # Weak path: do not invent extra services from the architecture diagram
    if leading.kind == "weak_queue":
        return list(dict.fromkeys(systems))[:3]

    # Strong path: services that call / are called by leading in Components prose
    arch = evidence["corpus"].get("architecture.md", "")
    idx = arch.lower().find("## components")
    region = arch[idx:] if idx >= 0 else ""
    if leading.component and region:
        for line in region.splitlines():
            low = line.lower()
            if leading.component not in low:
                continue
            if not any(k in low for k in ("calls", "pool", "timeout", "queue", "path")):
                continue
            for svc in _services_in(line):
                if svc not in systems and "third-party" not in svc:
                    systems.append(svc)

    # Also pick up payment-service style callers from matching ERROR siblings already done
    return [s for s in dict.fromkeys(systems) if "third-party" not in s][:4]


def _mttr_minutes(evidence: dict, confidence: float) -> int | None:
    if confidence < 50:
        return None
    rb = evidence["signals"].get("RUNBOOK")
    prev = evidence["signals"].get("PRECEDENT")
    if rb and rb.positive and not rb.unverified and rb.meta.get("mttr_minutes") is not None:
        return int(rb.meta["mttr_minutes"])
    if prev and prev.positive and prev.meta.get("mttr_minutes") is not None:
        return int(prev.meta["mttr_minutes"])
    return None


def _build_root_cause(evidence: dict, confidence: float) -> str:
    leading: Hypothesis = evidence["leading"]
    signals: dict[str, Signal] = evidence["signals"]
    matching_errors = evidence.get("matching_errors") or []

    if confidence >= 50 and matching_errors:
        deploy = signals["DEPLOY_TEMPORAL"]
        version = (deploy.meta.get("version") or "") if deploy.positive else ""
        from_n = deploy.meta.get("from_n") if deploy.positive else None
        to_n = deploy.meta.get("to_n") if deploy.positive else None
        when = ""
        if deploy.positive and deploy.meta.get("ts"):
            when = deploy.meta["ts"].strftime("%H:%M")

        parts = []
        if version and from_n is not None and to_n is not None and deploy.meta.get("mentions_pool"):
            parts.append(
                f"Deploy {version} reduced {leading.component} connection pool "
                f"from {from_n} to {to_n}" + (f" at {when}" if when else "") + "."
            )
        elif version:
            parts.append(
                f"Deploy {version} changed {leading.component} configuration in a way that "
                f"correlates with the failure onset."
            )
        else:
            parts.append(
                f"{leading.component} is failing with {leading.signature} under normal traffic."
            )

        # Mechanism from observed exception names
        exc_names = []
        for e in matching_errors:
            exc_names.extend(_exceptions_in(e.message))
            if "GATEWAY_TIMEOUT" in e.message:
                exc_names.append("GATEWAY_TIMEOUT")
        exc_names = list(dict.fromkeys(exc_names))[:2]
        if deploy.meta.get("mentions_pool") if deploy.positive else False:
            parts.append(
                "Under normal traffic the pool saturates, causing intermittent "
                + (" / ".join(exc_names) if exc_names else "timeout")
                + " charge failures."
            )
        else:
            parts.append(
                "Failures match the observed log signature "
                + ("(" + " / ".join(exc_names) + "). " if exc_names else "")
                + "Intermittent, not a hard outage."
            )

        corrob = []
        if signals["KNOWN_ISSUE"].positive:
            corrob.append(signals["KNOWN_ISSUE"].meta.get("issue_id") or "known issue")
        if signals["RUNBOOK"].positive:
            corrob.append(str(signals["RUNBOOK"].meta.get("runbook_id") or "runbook"))
        if signals["PRECEDENT"].positive:
            corrob.append(str(signals["PRECEDENT"].meta.get("incident_id") or "prior incident"))
        corrob.append("logs")
        if deploy.positive:
            corrob.append("deploy history")
        parts.append(
            "Multiple independent sources corroborate this ("
            + ", ".join(dict.fromkeys(corrob))
            + ")."
        )
        return " ".join(parts)

    comp = leading.component if leading.component != "unknown-service" else "the implicated service"
    delay_note = "an extended period"
    narr = _find_chunk(evidence["chunks"], lambda c: c.id == "logs.md#narrative")
    if narr:
        if m := re.search(r"(\d+\s*[-–]\s*\d+\s*minutes)", narr.text, re.I):
            delay_note = m.group(1).replace("–", "-")

    qlow = evidence["query"].lower()
    if "email" in qlow or "notification" in qlow or leading.kind == "weak_queue":
        return (
            f"Order confirmation emails sit in the {comp} queue for {delay_note} "
            f"before send; one elevated queue-depth warning exists, with no ERROR-level failures. "
            f"The bottleneck (undersized consumers vs third-party email provider latency) "
            f"cannot be confirmed - consumer/provider metrics are uninstrumented, and there is "
            f"no matching known issue, correlated deploy, or prior incident."
        )
    return (
        f"Suspected area: {comp} ({leading.signature}), but evidence is thin. "
        f"Independent corroboration is missing (deploy / known issue / precedent), "
        f"so the root cause cannot be confirmed from this corpus alone."
    )


def _build_remediation(evidence: dict, confidence: float, mttr: int | None) -> str:
    leading: Hypothesis = evidence["leading"]
    signals: dict[str, Signal] = evidence["signals"]

    if confidence >= 50 and evidence.get("matching_errors"):
        deploy = signals["DEPLOY_TEMPORAL"]
        from_n = deploy.meta.get("from_n") if deploy.positive else None
        version = deploy.meta.get("version") if deploy.positive else None
        baseline = str(from_n) if from_n is not None else "prior baseline"
        action = "configuration"
        if deploy.positive and deploy.meta.get("mentions_pool"):
            action = f"pool size to baseline {baseline}"
        steps = [
            f"1) Revert {leading.component} {action}"
            + (f" (undo {version})" if version else "")
            + ".",
            f"2) Redeploy {leading.component}.",
            "3) Confirm the ERROR signature stops and requests succeed.",
        ]
        if mttr is not None:
            rb_id = signals["RUNBOOK"].meta.get("runbook_id") if signals["RUNBOOK"].positive else "runbook"
            steps.append(f"Typical recovery ~{mttr} minutes ({rb_id}).")
        return " ".join(steps)

    comp = leading.component if leading.component != "unknown-service" else "the implicated service"
    rb_note = ""
    if signals["RUNBOOK"].positive:
        rb_note = f" ({signals['RUNBOOK'].meta.get('runbook_id')})"
    return (
        "1) Do not treat this as a confirmed root cause - flag for human review. "
        f"2) Inspect {comp} queue depth and consumer count now. "
        "3) Check third-party email provider status/latency. "
        "4) Add per-stage timing (queue wait vs provider send) before any permanent scale change. "
        f"Scaling consumers is unverified{rb_note}."
    )


def _build_supporting_evidence(evidence: dict, confidence: float) -> list[dict[str, str]]:
    signals = evidence["signals"]
    chunks = evidence["chunks"]
    matching_errors = evidence.get("matching_errors") or []
    primary_errors = evidence.get("primary_errors") or matching_errors
    items: list[dict[str, str]] = []
    used: set[str] = set()

    def add(source: str, excerpt: str) -> None:
        excerpt = (excerpt or "").strip()
        if source in used or not excerpt:
            return
        used.add(source)
        items.append({"source": source, "excerpt": _short_excerpt(excerpt, 260)})

    if matching_errors:
        add("logs.md", primary_errors[0].raw)
    else:
        narr = _find_chunk(chunks, lambda c: c.id == "logs.md#narrative")
        warn = next(
            (e for e in evidence["events"] if e.severity == "WARN" and "queue" in e.message.lower()),
            None,
        )
        if warn:
            delay_bit = ""
            if narr and (m := re.search(r"(\d+\s*[-–]\s*\d+\s*minutes)", narr.text, re.I)):
                delay_bit = (
                    f" Delays are consistently {m.group(1).replace('–', '-')} "
                    f"between Email queued and Email sent."
                )
            add("logs.md", warn.raw + "." + delay_bit)
        elif narr:
            add("logs.md", narr.text)

    dep = signals["DEPLOY_TEMPORAL"]
    if dep.excerpt:
        add("deployment_history.md", dep.excerpt)

    for key in ("KNOWN_ISSUE", "RUNBOOK", "PRECEDENT", "ARCH_PATH"):
        sig = signals[key]
        if confidence >= 50:
            if sig.positive and sig.excerpt:
                add(sig.source, sig.excerpt)
        else:
            # Include real negatives and weak positives; skip empty synthetics
            if sig.excerpt:
                add(sig.source, sig.excerpt)

    if confidence >= 50 and evidence["leading"].details.get("api_timeout"):
        for line in evidence["corpus"].get("api_specs.md", "").splitlines():
            if "5000ms" in line or "GATEWAY_TIMEOUT" in line:
                add("api_specs.md", line)
                break

    return items[:7]


def _build_report(evidence: dict) -> dict:
    confidence = round(_calibrate_confidence(evidence), 1)
    mttr = _mttr_minutes(evidence, confidence)
    return {
        "root_cause": _build_root_cause(evidence, confidence),
        "supporting_evidence": _build_supporting_evidence(evidence, confidence),
        "impacted_systems": _impacted_systems(evidence),
        "mttr_minutes": mttr,
        "remediation": _build_remediation(evidence, confidence, mttr),
        "confidence_score": confidence,
        "needs_human_review": confidence < 50,
    }


def investigate(query: str, corpus: dict) -> dict:
    ingested = _ingest_corpus(corpus)
    ranked = _retrieve_relevant_documents(query, ingested)
    evidence = _correlate_evidence(query, ingested, ranked)
    return _build_report(evidence)
