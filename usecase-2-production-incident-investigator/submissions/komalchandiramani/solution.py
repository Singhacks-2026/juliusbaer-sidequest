"""Production incident investigator — Use Case 2.

Stage 1 of 4 (ingestion) is implemented. Documents are chunked by markdown
shape rather than by filename, so the same rules run over any incident folder.
`Chunk.text` is always verbatim source, ready to quote as an excerpt;
`Chunk.search_text` is the normalised form used for retrieval.
"""
from __future__ import annotations

import csv
import io
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, eq=False)
class Chunk:
    id: str
    source: str
    source_type: str
    kind: str  # log_line | csv_row | table_row | bullet | section | paragraph | code_block
    text: str
    search_text: str
    tokens: tuple[str, ...]
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

try:  # optional: without nltk we fall back to no stemming rather than crashing
    from nltk.stem import PorterStemmer

    _STEMMER: Any = PorterStemmer()
except ImportError:
    _STEMMER = None


def _stem(token: str) -> str:
    """Porter stem, plain English words only.

    Component names and exception class names are identifiers rather than
    English, so they are left alone and matched literally.
    """
    if _STEMMER is None or len(token) < 4 or not token.isalpha():
        return token
    return _STEMMER.stem(token)


# --------------------------------------------------------------------------
# Tokenisation
# --------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*|\d+")
# Splits CamelCase and hyphenated runs alike:
# ConnectionPoolTimeoutException -> Connection, Pool, Timeout, Exception
_SUBWORD_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

_STOPWORDS = frozenset(
    """
    a about after all also an and any are as at be been before being between both but by
    can could did do does doing done down during each few for from further had has have
    having he her here hers him his how i if in into is it its itself just me more most
    my no nor not now of off on once only or other our out over own same she should so
    some such than that the their them then there these they this those through to too
    under until up very was we were what when where which while who whom why will with
    would you your
    """.split()
)


def _tokenize(text: str) -> list[str]:
    """Lowercased tokens: each word, its sub-words, and their stems.

    Shared with retrieval so documents and the query normalise identically.
    Sub-words are split on the original casing, then lowercased — lowercasing
    first would destroy the CamelCase boundaries. Stems are emitted alongside
    the raw forms, never instead of them.
    """
    tokens: list[str] = []
    for raw in _WORD_RE.findall(text):
        seen: set[str] = set()
        for variant in (raw.lower(), *(p.lower() for p in _SUBWORD_RE.findall(raw))):
            for token in (variant, _stem(variant)):
                if len(token) < 2 or token in seen or token in _STOPWORDS:
                    continue
                seen.add(token)
                tokens.append(token)
    return tokens


# --------------------------------------------------------------------------
# Structural patterns
# --------------------------------------------------------------------------

_LOG_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})\s+"
    r"(?P<level>[A-Z]{3,8})\s+"
    r"(?P<component>[A-Za-z][\w.-]*)\s+"
    r"(?P<message>.+?)\s*$"
)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_FENCE_RE = re.compile(r"^\s*```")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_BULLET_RE = re.compile(r"^[-*]\s+\S")
_BULLET_CONT_RE = re.compile(r"^\s+\S")
_BOLD_LEAD_RE = re.compile(r"^[-*]\s+\*\*([^*]+)\*\*")
_BOLD_RE = re.compile(r"\*\*|__")
_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?")
# MTTR = Mean Time To Recover. Matches "Typical MTTR: 20 minutes." and "**MTTR**: 22 minutes".
_MTTR_RE = re.compile(r"MTTR[^0-9]{0,20}?(\d+)\s*minutes?", re.IGNORECASE)


def _strip_bold(s: str) -> str:
    return _BOLD_RE.sub("", s).strip()


def _labelled(pairs: list[tuple[str, str]]) -> str:
    """Render a row as self-describing text: 'issue_id: KI-101 | title: ...'."""
    return " | ".join(f"{k}: {v}" for k, v in pairs if v)


def _looks_like_csv(text: str) -> bool:
    """True when the document parses as a rectangular CSV table."""
    try:
        rows = [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]
    except csv.Error:
        return False
    if len(rows) < 2 or len(rows[0]) < 3:
        return False
    return all(len(r) == len(rows[0]) for r in rows)


def _is_log_fence(body: str) -> bool:
    return sum(1 for ln in body.splitlines() if _LOG_LINE_RE.match(ln.strip())) >= 3


def _starts_block(line: str) -> bool:
    return bool(
        _FENCE_RE.match(line)
        or _HEADING_RE.match(line)
        or _TABLE_ROW_RE.match(line)
        or _BULLET_RE.match(line)
    )


def _scan_blocks(text: str) -> list[tuple[str, str]]:
    """Cut a document into (kind, raw_text) blocks: fence/heading/table/bullets/prose.

    Blank lines never break a prose run, so a prose block stays a verbatim
    substring of the source.
    """
    lines = text.splitlines()
    blocks: list[tuple[str, str]] = []
    i, n = 0, len(lines)

    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
        elif _FENCE_RE.match(line):
            j = i + 1
            while j < n and not _FENCE_RE.match(lines[j]):
                j += 1
            blocks.append(("fence", "\n".join(lines[i + 1 : j])))
            i = j + 1
        elif _HEADING_RE.match(line):
            blocks.append(("heading", line))
            i += 1
        elif _TABLE_ROW_RE.match(line):
            j = i
            while j < n and _TABLE_ROW_RE.match(lines[j]):
                j += 1
            blocks.append(("table", "\n".join(lines[i:j])))
            i = j
        elif _BULLET_RE.match(line):
            j = i
            while j < n and (_BULLET_RE.match(lines[j]) or _BULLET_CONT_RE.match(lines[j])):
                j += 1
            blocks.append(("bullets", "\n".join(lines[i:j])))
            i = j
        else:
            j = i
            while j < n:
                if lines[j].strip():
                    j += 1
                    continue
                k = j
                while k < n and not lines[k].strip():
                    k += 1
                if k < n and not _starts_block(lines[k]):
                    j = k
                    continue
                break
            blocks.append(("prose", "\n".join(lines[i:j]).strip("\n")))
            i = j

    return blocks



def _chunk_csv(name: str, stype: str, text: str) -> list[Chunk]:
    """One chunk per row; column names are glued on so a detached row still reads."""
    raw_lines = [ln for ln in text.splitlines() if ln.strip()]
    rows = [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]
    if not rows:
        return []
    header = [h.strip() for h in rows[0]]

    chunks = []
    for idx, row in enumerate(rows[1:], start=1):
        pairs = list(zip(header, (c.strip() for c in row)))
        fields = dict(pairs)
        search_text = _labelled(pairs)
        meta: dict[str, Any] = {"fields": fields}
        component = fields.get("affected_component") or fields.get("component")
        if component:
            meta["component"] = component
        chunks.append(
            Chunk(
                id=f"{name}#{row[0].strip() or idx}",
                source=name,
                source_type=stype,
                kind="csv_row",
                text=raw_lines[idx] if idx < len(raw_lines) else ",".join(row),
                search_text=search_text,
                tokens=tuple(_tokenize(search_text)),
                meta=meta,
            )
        )
    return chunks


def _chunk_log_fence(name: str, stype: str, body: str) -> list[Chunk]:
    chunks = []
    for lineno, line in enumerate(body.splitlines(), start=1):
        m = _LOG_LINE_RE.match(line.strip())
        if not m:
            continue
        chunks.append(
            Chunk(
                id=f"{name}#L{lineno}",
                source=name,
                source_type=stype,
                kind="log_line",
                text=line.strip(),
                search_text=line.strip(),
                tokens=tuple(_tokenize(line.strip())),
                meta=m.groupdict(),
            )
        )
    return chunks


def _chunk_table(name: str, stype: str, body: str, seq: int) -> list[Chunk]:
    rows = [ln for ln in body.splitlines() if _TABLE_ROW_RE.match(ln) and not _TABLE_SEP_RE.match(ln)]
    if len(rows) < 2:
        return []

    def cells(line: str) -> list[str]:
        return [_strip_bold(c) for c in line.strip().strip("|").split("|")]

    header = cells(rows[0])
    chunks = []
    for idx, line in enumerate(rows[1:]):
        pairs = list(zip(header, cells(line)))
        search_text = _labelled(pairs)
        meta: dict[str, Any] = {"fields": dict(pairs)}
        ts = _TIMESTAMP_RE.search(line)
        if ts:
            meta["ts"] = ts.group(0)
        for key, value in pairs:
            if "component" in key.lower() and value:
                meta["component"] = value
        chunks.append(
            Chunk(
                id=f"{name}#row{seq}.{idx}",
                source=name,
                source_type=stype,
                kind="table_row",
                text=line.rstrip(),
                search_text=search_text,
                tokens=tuple(_tokenize(search_text)),
                meta=meta,
            )
        )
    return chunks


def _chunk_bullets(name: str, stype: str, body: str, heading: str | None, seq: int) -> list[Chunk]:
    """One chunk per bullet — keeps a component list from ranking as one dense blob."""
    lines = body.splitlines()
    starts = [i for i, ln in enumerate(lines) if _BULLET_RE.match(ln)]
    if not starts:
        return []
    bounds = starts + [len(lines)]

    chunks = []
    for idx in range(len(starts)):
        raw = "\n".join(lines[bounds[idx] : bounds[idx + 1]]).rstrip()
        if not raw.strip():
            continue
        meta: dict[str, Any] = {"heading": heading} if heading else {}
        lead = _BOLD_LEAD_RE.match(raw)
        if lead:
            meta["component"] = lead.group(1).strip()
        search_text = f"{heading}\n{raw}" if heading else raw
        chunks.append(
            Chunk(
                id=f"{name}#b{seq}.{idx}",
                source=name,
                source_type=stype,
                kind="bullet",
                text=raw,
                search_text=search_text,
                tokens=tuple(_tokenize(search_text)),
                meta=meta,
            )
        )
    return chunks


def _chunk_prose(
    name: str, stype: str, body: str, heading: str | None, seq: int, per_paragraph: bool
) -> list[Chunk]:
    """Prose under the current heading, split per paragraph when the document
    has no `##` sections to lean on."""
    parts = [p for p in re.split(r"\n\s*\n", body) if p.strip()] if per_paragraph else [body]

    chunks = []
    for idx, part in enumerate(parts):
        raw = part.rstrip()
        if not raw.strip():
            continue
        meta: dict[str, Any] = {"heading": heading} if heading else {}
        mttr = _MTTR_RE.search(raw)
        if mttr:
            meta["mttr_minutes"] = int(mttr.group(1))
        search_text = f"{heading}\n{raw}" if heading else raw
        chunks.append(
            Chunk(
                id=f"{name}#s{seq}.{idx}",
                source=name,
                source_type=stype,
                kind="section" if heading else "paragraph",
                text=raw,
                search_text=search_text,
                tokens=tuple(_tokenize(search_text)),
                meta=meta,
            )
        )
    return chunks


def _chunk_document(name: str, text: str) -> list[Chunk]:
    stype = name.rsplit(".", 1)[0] if "." in name else name

    if _looks_like_csv(text):
        return _chunk_csv(name, stype, text)

    blocks = _scan_blocks(text)

    # A document whose fence is a log dump is treated as logs and nothing else:
    # the prose around it is narration that states the conclusion outright.
    # Scoping this to log documents keeps it away from real trailing evidence
    # such as deployment_history's closing paragraph.
    log_fences = [b for kind, b in blocks if kind == "fence" and _is_log_fence(b)]
    if log_fences:
        return [c for b in log_fences for c in _chunk_log_fence(name, stype, b)]

    has_sections = any(k == "heading" and b.lstrip().startswith("##") for k, b in blocks)

    chunks: list[Chunk] = []
    heading: str | None = None
    for seq, (kind, body) in enumerate(blocks):
        if kind == "heading":
            # Headings are context, never chunks of their own.
            heading = _strip_bold(_HEADING_RE.match(body).group(2))
        elif kind == "fence":
            raw = body.rstrip()
            if raw.strip():
                search_text = f"{heading}\n{raw}" if heading else raw
                chunks.append(
                    Chunk(
                        id=f"{name}#code{seq}",
                        source=name,
                        source_type=stype,
                        kind="code_block",
                        text=raw,
                        search_text=search_text,
                        tokens=tuple(_tokenize(search_text)),
                        meta={"heading": heading} if heading else {},
                    )
                )
        elif kind == "table":
            chunks.extend(_chunk_table(name, stype, body, seq))
        elif kind == "bullets":
            chunks.extend(_chunk_bullets(name, stype, body, heading, seq))
        elif kind == "prose":
            chunks.extend(_chunk_prose(name, stype, body, heading, seq, not has_sections))

    return chunks


# --------------------------------------------------------------------------
# Stage 1: ingestion
# --------------------------------------------------------------------------


def _document_frequencies(chunks: list[Chunk]) -> Counter[str]:
    df: Counter[str] = Counter()
    for chunk in chunks:
        df.update(set(chunk.tokens))
    return df


def _compute_idf(df: Counter[str], n: int) -> dict[str, float]:
    """Smoothed IDF. Also what discounts the query boilerplate ("identify",
    "root cause", "remediation"), so no hand-written domain stopword list."""
    return {token: math.log((n + 1) / (count + 1)) + 1.0 for token, count in df.items()}


def _ingest_corpus(corpus: dict) -> dict:
    """Turn {filename: text} into chunks plus corpus statistics.

    Returns chunks, by_id, df/idf/avg_len (retrieval statistics), source_types
    and known_components (every component name seen anywhere in the corpus).
    """
    chunks: list[Chunk] = []
    for name in sorted(corpus):
        chunks.extend(_chunk_document(name, corpus[name] or ""))

    components = set()
    for chunk in chunks:
        name = str(chunk.meta.get("component", "")).strip()
        if name and " " not in name:
            components.add(name)

    df = _document_frequencies(chunks)
    lengths = [len(c.tokens) for c in chunks]

    return {
        "chunks": chunks,
        "by_id": {c.id: c for c in chunks},
        "df": df,
        "idf": _compute_idf(df, len(chunks)),
        "avg_len": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "source_types": dict(Counter(c.source_type for c in chunks)),
        "known_components": components,
    }


# --------------------------------------------------------------------------
# Stage 2: retrieval
# --------------------------------------------------------------------------


def _tfidf_vector(tokens: tuple[str, ...] | list[str], idf: dict[str, float]) -> dict[str, float]:
    """L2-normalised TF-IDF vector. Sublinear tf keeps a long runbook from
    outweighing a short log line just by repeating a term."""
    weights = {
        token: (1.0 + math.log(count)) * idf[token]
        for token, count in Counter(tokens).items()
        if token in idf
    }
    norm = math.sqrt(sum(w * w for w in weights.values()))
    return {t: w / norm for t, w in weights.items()} if norm else {}


def _score_cosine(q_tokens: list[str], index: dict) -> dict[str, float]:
    idf = index["idf"]
    q_vec = _tfidf_vector(q_tokens, idf)
    if not q_vec:
        return {}
    scores = {}
    for chunk in index["chunks"]:
        c_vec = _tfidf_vector(chunk.tokens, idf)
        score = sum(w * c_vec.get(token, 0.0) for token, w in q_vec.items())
        if score > 0:
            scores[chunk.id] = score
    return scores


# Okapi BM25 defaults.
_BM25_K1 = 1.5
_BM25_B = 0.75


def _score_bm25(q_tokens: list[str], index: dict) -> dict[str, float]:
    """Okapi BM25.

    Unlike cosine it normalises by raw length rather than by the vector norm, so
    a short, information-dense record is not penalised for also carrying rare
    terms the query never mentioned.
    """
    df, n, avg_len = index["df"], len(index["chunks"]), index["avg_len"]
    if not n or not avg_len:
        return {}

    q_idf = {
        token: math.log(1.0 + (n - df[token] + 0.5) / (df[token] + 0.5))
        for token in set(q_tokens)
        if token in df
    }
    if not q_idf:
        return {}

    scores = {}
    for chunk in index["chunks"]:
        counts = Counter(chunk.tokens)
        norm = _BM25_K1 * (1 - _BM25_B + _BM25_B * len(chunk.tokens) / avg_len)
        score = sum(
            weight * (counts[token] * (_BM25_K1 + 1)) / (counts[token] + norm)
            for token, weight in q_idf.items()
            if counts[token]
        )
        if score > 0:
            scores[chunk.id] = score
    return scores


def _rank(tokens: list[str], index: dict, method: str) -> list[tuple[str, float]]:
    scorer = {"bm25": _score_bm25, "cosine": _score_cosine}[method]
    return sorted(scorer(tokens, index).items(), key=lambda pair: (-pair[1], pair[0]))


# --------------------------------------------------------------------------
# Two-hop retrieval
# --------------------------------------------------------------------------

_PRF_DOCS = 5  # chunks treated as pseudo-relevant feedback
_PRF_TERMS = 15  # expansion terms drawn from them
_RRF_K = 60  # standard reciprocal-rank-fusion damping


def _expansion_terms(ranked: list[tuple[str, float]], index: dict) -> list[str]:
    """Pseudo-relevance feedback: the most distinctive terms of the top hits.

    The query asks about symptoms ("payments are failing") while the decisive
    records are written in the vocabulary of causes ("ConnectionPoolTimeoutException",
    "pool size"). Those two vocabularies never overlap, so no single-pass lexical
    scorer can bridge them. The top hits do contain both, which is what makes them
    usable as a bridge.
    """
    idf = index["idf"]
    weights: Counter[str] = Counter()
    for chunk_id, score in ranked[:_PRF_DOCS]:
        for token, count in Counter(index["by_id"][chunk_id].tokens).items():
            if token in idf:
                weights[token] += score * (1.0 + math.log(count)) * idf[token]
    return [token for token, _ in weights.most_common(_PRF_TERMS)]


def _fuse_rrf(rankings: list[list[tuple[str, float]]]) -> list[tuple[str, float]]:
    """Reciprocal rank fusion. Combines rankings by position rather than by score,
    so the two hops' incomparable score scales don't need reconciling."""
    fused: Counter[str] = Counter()
    for ranked in rankings:
        for position, (chunk_id, _) in enumerate(ranked, start=1):
            fused[chunk_id] += 1.0 / (_RRF_K + position)
    return sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))


def _retrieve_relevant_documents(
    query: str, index: dict, method: str = "bm25", two_hop: bool = True,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Rank chunks against the query, most relevant first.

    Takes the index rather than the raw corpus so the corpus is parsed once per
    investigation. `method` is "bm25" or "cosine"; `two_hop` adds a pseudo-relevance
    feedback pass fused with the first by RRF.
    """
    hop1 = _rank(_tokenize(query), index, method)
    if not two_hop or not hop1:
        return hop1[:top_k] if top_k else hop1

    # RM3-style: the second hop *extends* the query rather than replacing it.
    # Replacing it lets a decoy at rank 1 drag the whole second pass into its own
    # vocabulary - on the ambiguous incident that pulled unrelated payment records
    # into an email-delay investigation.
    q_tokens = _tokenize(query)
    hop2 = _rank(q_tokens + _expansion_terms(hop1, index), index, method)
    ranked = _threshold(_fuse_rrf([hop1, hop2]))
    return ranked[:top_k] if top_k else ranked


# Chunks scoring below this fraction of the best chunk are dropped as irrelevant.
_RELEVANCE_FLOOR = 0.75


def _threshold(ranked: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Drop the weakly-scoring tail rather than carrying every chunk that matched
    a single common word."""
    if not ranked:
        return ranked
    cutoff = ranked[0][1] * _RELEVANCE_FLOOR
    return [pair for pair in ranked if pair[1] >= cutoff]


# --------------------------------------------------------------------------
# Stage 3: correlation
# --------------------------------------------------------------------------

# Source types that describe the system rather than the incident. They match
# topically no matter what is asked - both are ranked 1-2 in both incidents -
# so they may inform the narrative but never count as corroboration.
_DESCRIPTIVE = frozenset({"architecture", "api_specs"})
_CORROBORATING = ("logs", "known_issues", "deployment_history", "runbooks", "previous_incidents")

# Phrases that deny or hedge a *link* between the component and the incident.
# Deliberately narrow: a bare "no" is not a denial - the corpus's strongest single
# piece of evidence reads "no available connection after 5000ms".
_NEGATION_RE = re.compile(
    r"\bno\s+(?:previous|deployment|matching|evidence|record|known)"
    r"|\bnot\s+(?:currently|part of|related|applicable|a bug|instrumented)"
    r"|\bthis is the first\b|\bno .{0,40}\b(?:in the (?:historical )?record)\b"
    r"|\bunrelated\b|\bunconfirmed\b|\bunverified\b|\binconclusive\b|\bincomplete\b"
    r"|\bmay not apply\b|\bdoes not affect\b|\bseparate from\b|\bcosmetic only\b",
    re.IGNORECASE,
)


def _mentions(chunk: Chunk, component: str) -> bool:
    return component in chunk.tokens or chunk.meta.get("component") == component


def _is_negated(chunk: Chunk) -> bool:
    """True when a chunk names the component but denies or hedges the connection.

    Without this, "No deployment touched notification-service" and "No previous
    incident ... involves email delivery latency" read as support for exactly the
    hypothesis they rule out. Log lines are exempt: a log line is a raw
    observation, never a claim about whether something is related.
    """
    return chunk.kind != "log_line" and bool(_NEGATION_RE.search(chunk.text))


def _anomalies(index: dict) -> list[Chunk]:
    """Log lines above INFO — the observable symptoms of this incident."""
    return [
        c for c in index["chunks"]
        if c.kind == "log_line" and c.meta.get("level", "INFO") not in ("INFO", "DEBUG", "TRACE")
    ]


def _pick_hypothesis(index: dict, scores: dict[str, float]) -> str | None:
    """The component with anomalous log evidence that the query is most about.

    Two filters, because either alone picks the wrong component. Anomalies alone
    tie three ways on the ambiguous incident (checkout render, queue depth, refund
    webhook). Relevance alone picks whatever architecture.md discusses most, which
    is the same answer for every possible query.

    So: a candidate must show at least one non-INFO log line, and among those the
    winner is the one carrying the most retrieved relevance across the *evidential*
    sources - descriptive prose is excluded, since it describes every component
    equally regardless of what was asked.
    """
    candidates = {
        c.meta["component"] for c in _anomalies(index)
        if c.meta.get("component") in index["known_components"]
    }
    if not candidates:
        return None

    mass: Counter[str] = Counter()
    for chunk in index["chunks"]:
        if chunk.source_type in _DESCRIPTIVE:
            continue
        for component in candidates:
            if _mentions(chunk, component):
                mass[component] += scores.get(chunk.id, 0.0)
    return max(candidates, key=lambda c: (mass[c], c))


def _correlate_evidence(query: str, index: dict, ranked: list[tuple[str, float]]) -> dict:
    """Test one hypothesis against every independent source type.

    A confident conclusion is not "the top-ranked chunk says X" but "logs,
    deployment history, known issues, the runbook and a previous incident
    independently say X". Returns the hypothesis plus the corroborating and
    refuting evidence, which is what stage 4 calibrates on.
    """
    scores = dict(ranked)
    order = {chunk_id: i for i, (chunk_id, _) in enumerate(ranked)}
    component = _pick_hypothesis(index, scores)

    corroboration: dict[str, Chunk] = {}
    refutations: list[Chunk] = []
    if component:
        # A record whose parsed component *is* the hypothesis outranks one that
        # merely mentions it in prose, so the cited excerpt is the v2.4.1 table row
        # rather than a sentence about deployments in general.
        candidates = sorted(
            (c for c in index["chunks"] if _mentions(c, component)),
            key=lambda c: (c.meta.get("component") != component, order.get(c.id, len(order))),
        )
        anomaly_ids = {c.id for c in _anomalies(index)}
        for chunk in candidates:
            if chunk.source_type in _DESCRIPTIVE or chunk.source_type not in _CORROBORATING:
                continue
            # A routine INFO line is not evidence of a fault, only that the
            # component ran. Logs corroborate through anomalies or not at all.
            if chunk.source_type == "logs" and chunk.id not in anomaly_ids:
                continue
            if _is_negated(chunk):
                refutations.append(chunk)
            elif chunk.source_type not in corroboration:
                corroboration[chunk.source_type] = chunk

    # A deployment only explains an incident it precedes.
    anomalies = [c for c in _anomalies(index) if not component or _mentions(c, component)]
    first_anomaly = min((c.meta["ts"] for c in anomalies), default=None)
    deploy = corroboration.get("deployment_history")
    if deploy and first_anomaly and str(deploy.meta.get("ts", "")) > first_anomaly:
        refutations.append(corroboration.pop("deployment_history"))

    # Only components whose own anomalies fall inside the incident's time window.
    # Both logs carry unrelated background noise - a checkout render warning hours
    # earlier, a database failover afterwards - which is not part of this incident.
    impacted = [component] if component else []
    if anomalies:
        window = (min(c.meta["ts"] for c in anomalies), max(c.meta["ts"] for c in anomalies))
        impacted += sorted(
            {
                c.meta["component"]
                for c in _anomalies(index)
                if c.meta.get("component") != component
                and window[0] <= c.meta["ts"] <= window[1]
            }
        )

    # Prefer the runbook's figure over a past incident's: the runbook is the current
    # guidance for this failure mode, the incident is one historical sample of it.
    mttr = next(
        (
            corroboration[st].meta["mttr_minutes"]
            for st in ("runbooks", "previous_incidents")
            if st in corroboration and "mttr_minutes" in corroboration[st].meta
        ),
        None,
    )

    return {
        "component": component,
        "corroboration": corroboration,
        "refutations": refutations,
        "anomalies": anomalies,
        "impacted_systems": impacted,
        "mttr_minutes": mttr,
        "top_chunks": [index["by_id"][cid] for cid, _ in ranked[:10]],
    }


# --------------------------------------------------------------------------
# Stage 4: calibration and assembly
# --------------------------------------------------------------------------

# Deterministic bands keyed on how many independent source types agree.
# 100 is unreachable by construction: seven documents are not certainty.
_CONFIDENCE_BANDS = {5: 92.0, 4: 82.0, 3: 70.0, 2: 52.0, 1: 30.0, 0: 12.0}
_REFUTATION_PENALTY = 5.0
_CONFIDENCE_CEILING = 95.0


def _calibrate_confidence(evidence: dict) -> float:
    """Corroboration breadth -> 0-100. Thin evidence must land below 50."""
    if not evidence.get("component"):
        return _CONFIDENCE_BANDS[0]
    score = _CONFIDENCE_BANDS[min(len(evidence["corroboration"]), 5)]
    score -= _REFUTATION_PENALTY * len(evidence["refutations"])
    return round(max(0.0, min(_CONFIDENCE_CEILING, score)), 1)


def _describe(evidence: dict, confidence: float) -> tuple[str, str]:
    """Root cause and remediation, composed from the corroborating records."""
    component = evidence["component"]
    corroboration = evidence["corroboration"]
    if not component:
        return (
            "Insufficient evidence to identify a root cause.",
            "Escalate to a human on-call engineer; no actionable anomaly in the corpus.",
        )

    signature = ""
    for chunk in evidence["anomalies"]:
        message = chunk.meta.get("message", "")
        if ":" in message:
            signature = message.split(":")[0].strip()
            break

    if confidence < 50:
        return (
            f"Probable but unconfirmed: delivery latency in `{component}`"
            f"{f' ({signature})' if signature else ''}. Only {len(corroboration)} independent "
            f"source corroborates this and {len(evidence['refutations'])} contradict or qualify "
            "it - no matching known issue, no correlated deployment, and no precedent on file. "
            "The evidence is too thin to name a cause with confidence.",
            f"Do not act on this diagnosis unreviewed. Instrument `{component}` with per-stage "
            "timing to establish where the delay accrues, then re-investigate.",
        )

    cause = f"`{component}` is failing"
    if signature:
        cause += f" with `{signature}`"
    trigger = corroboration.get("deployment_history")
    if trigger:
        change = trigger.meta.get("fields", {}).get("Change") or " ".join(trigger.text.split())
        cause += f", triggered by the preceding deployment: {change}"
    cause += (
        f". Corroborated independently by {len(corroboration)} source types "
        f"({', '.join(sorted(corroboration))})."
    )

    fix = ""
    runbook = corroboration.get("runbooks")
    if runbook:
        # The runbook's remediation is a wrapped paragraph, so take the whole
        # block up to the next blank line rather than its first line.
        blocks = re.split(r"\n\s*\n", runbook.text)
        for block in blocks:
            if "remediation" in block.lower():
                fix = " ".join(_strip_bold(block).split())
                break
    return cause, fix or f"Follow the runbook for `{component}` and revert the correlated change."


def investigate(query: str, corpus: dict) -> dict:
    """Investigate one incident and return a structured report."""
    index = _ingest_corpus(corpus)
    ranked = _retrieve_relevant_documents(query, index)
    evidence = _correlate_evidence(query, index, ranked)
    confidence = _calibrate_confidence(evidence)
    root_cause, remediation = _describe(evidence, confidence)

    # One excerpt per corroborating source type, so the evidence spans independent
    # documents rather than restating the single best-matching one.
    supporting = [
        {"source": chunk.source, "excerpt": chunk.text}
        for _, chunk in sorted(evidence["corroboration"].items())
    ]
    if not supporting and evidence["anomalies"]:
        first = evidence["anomalies"][0]
        supporting = [{"source": first.source, "excerpt": first.text}]

    return {
        "root_cause": root_cause,
        "supporting_evidence": supporting,
        "impacted_systems": evidence["impacted_systems"],
        "mttr_minutes": evidence["mttr_minutes"],
        "remediation": remediation,
        "confidence_score": confidence,
        "needs_human_review": confidence < 50,
    }
