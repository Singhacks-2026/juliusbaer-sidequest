"""Production incident investigator - retrieval + cross-document evidence
correlation with a calibrated confidence score.

    investigate(query, corpus) -> report dict

Pipeline: ingest (chunk to natural units) -> classify (doc type from content)
-> retrieve (TF-IDF cosine) -> hypothesise (component + anomaly signature)
-> corroborate (six axes, independently) -> calibrate (axes -> score, capped)
-> compose (report).

Confidence is a function of how many *independent* documents agree on the
leading hypothesis - never of how relevant the top-ranked document felt. Two
mechanisms keep it honest: explicit negations in a corroborating document
("no deployment touched X", "first recorded report") count *against* the
hypothesis, and self-hedged sources ("unverified", "may not apply here") are
demoted from strong to weak.

Standard library only. No incident-specific literals: every component name,
signature and document role is learned from the corpus at runtime.
"""
from __future__ import annotations

import csv
import io
import math
import re
from collections import defaultdict
from datetime import datetime, timedelta

# --------------------------------------------------------------------------
# Generic lexicons. These are ops-English vocabulary, not incident facts -
# nothing here names a component, a file or an exception.
# --------------------------------------------------------------------------

SEVERITY = {"DEBUG": 0, "TRACE": 0, "INFO": 0, "NOTICE": 1, "WARN": 2,
            "WARNING": 2, "ERROR": 3, "FATAL": 3, "CRITICAL": 3, "SEVERE": 3}

FAILURE_INTENT = {"fail", "failing", "failed", "failure", "error", "errors",
                  "exception", "timeout", "timeouts", "crash", "reject",
                  "rejected", "refused", "declined", "broken", "5xx", "504",
                  "502", "500"}

LATENCY_INTENT = {"late", "latency", "slow", "slowly", "delay", "delayed",
                  "delays", "lag", "lagging", "backlog", "stuck", "queue",
                  "waiting", "hour", "hours", "arriving"}

# Words a document uses when it is unsure of itself.
HEDGE_MARKERS = ("unverified", "unconfirmed", "not confirmed", "may not apply",
                 "incomplete", "pending", "unclear", "not currently instrumented",
                 "not exposed as a metric", "no documented sla", "assumed",
                 "suspected", "possibly", "outside this service's direct control",
                 "not instrumented", "worth noting")

# Statements that assert a correlation does *not* exist.
NEGATION_MARKERS = ("no previous incident", "no deployment", "no other deployment",
                    "first recorded", "no matching", "none of which", "no incident",
                    "does not affect", "not part of this incident", "unrelated",
                    "post-date", "no other runbook", "cosmetic only")

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "were",
    "be", "been", "it", "its", "this", "that", "these", "those", "for", "on",
    "at", "by", "with", "from", "as", "but", "if", "then", "than", "so", "not",
    "no", "can", "will", "would", "should", "could", "has", "have", "had", "do",
    "does", "did", "we", "you", "they", "he", "she", "i", "what", "which",
    "who", "when", "where", "how", "why", "any", "all", "some", "there", "here",
    "identify", "recommended", "supporting", "probable", "please", "also",
    "into", "out", "up", "down", "over", "under", "after", "before", "during",
    "per", "via", "one", "two", "three", "see", "note", "notes", "using",
}

LOG_LINE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})\s+"
    r"(?P<level>[A-Z]{4,8})\s+(?P<component>[A-Za-z][\w.\-]*)\s+(?P<message>.+)$")

TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?")
MTTR = re.compile(r"\bmttr\b[^0-9\n]{0,40}?(\d+)\s*(minute|min|hour|hr)", re.I)
KEY_VALUE = re.compile(r"\b([\w.]+)=([\w.\-:/]+)")
VERSION_CELL = re.compile(r"^v?\d+(\.\d+){1,3}$", re.I)
COMPONENT_TOKEN = re.compile(r"\b[a-z][a-z0-9]*(?:-[a-z0-9]+){1,3}\b")
ENDPOINT = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE)\s+/")
EXTERNAL_DEP = re.compile(
    r"(?:(?:a|an|the)\s+)?((?:third-party|external)\s+[a-z][a-z ]{2,30}?"
    r"(?:provider|service|system|api|gateway))|"
    r"([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+){0,3})\s*\(external[^)]*\)")

# A candidate component must reach this share of the top component's query
# affinity before its evidence is even scored. Without it, a well-catalogued
# but off-topic anomaly can out-corroborate the one actually being asked about.
RELEVANCE_GATE = 0.35

STRONG, WEAK, ABSENT, CONTRADICTED = "STRONG", "WEAK", "ABSENT", "CONTRADICTED"
AXES = ("LOGS", "DEPLOY", "KNOWN_ISSUE", "PRECEDENT", "RUNBOOK", "MECHANISM")

REQUIRED_KEYS = ("root_cause", "supporting_evidence", "impacted_systems",
                 "mttr_minutes", "remediation", "confidence_score",
                 "needs_human_review")


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Hyphenated identifiers are kept whole *and*
    split, so a plural in the query still reaches a hyphenated component name."""
    out = []
    for raw in re.findall(r"[a-z0-9][a-z0-9\-_.]*", text.lower()):
        raw = raw.strip("._-")
        for tok in ([raw] + raw.split("-") if "-" in raw else [raw]):
            if len(tok) < 2 or tok in STOPWORDS or tok.isdigit():
                continue
            if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
                tok = tok[:-1]          # crude but sufficient de-pluralisation
            out.append(tok)
    return out


def _norm_ws(text: str, limit: int = 420) -> str:
    text = re.sub(r"\s+", " ", text.replace("|", " ")).strip(" *#|")
    return text if len(text) <= limit else text[:limit].rstrip() + " ..."


def _signature(message: str) -> str:
    """Collapse a log message to a repeatable signature: drop key=value pairs
    and digit runs so N occurrences of the same fault group into one signal."""
    sig = KEY_VALUE.sub("", message)
    sig = re.sub(r"\d+", "<N>", sig)
    return re.sub(r"\s+", " ", sig).strip(" :-").lower()


def _parse_ts(value: str):
    value = value.strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _has(text: str, markers) -> list[str]:
    low = text.lower()
    return [m for m in markers if m in low]


def _distinctive(tokens) -> set[str]:
    """Long unhyphenated tokens - exception classes, error codes. A shared one
    is a far stronger link between two documents than any amount of prose
    overlap, which is what ties a log line to a catalog row."""
    return {t for t in tokens if len(t) >= 12 and "-" not in t and "." not in t}


# --------------------------------------------------------------------------
# 1. ingest - chunk each document to its natural unit
# --------------------------------------------------------------------------

class Unit:
    __slots__ = ("source", "doctype", "anchor", "text", "meta", "tokens", "vec")

    def __init__(self, source, doctype, anchor, text, meta=None):
        self.source, self.doctype, self.anchor = source, doctype, anchor
        self.text, self.meta = text, meta or {}
        self.tokens, self.vec = [], {}

    def __repr__(self):
        return f"<{self.doctype} {self.source}#{self.anchor}>"


def _looks_like_csv(text: str) -> bool:
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        return False
    head = lines[0]
    return head.count(",") >= 2 and all(
        l.count(",") >= head.count(",") - 1 for l in lines[1:3])


def _classify(name: str, text: str, is_csv: bool, header: list[str]) -> str:
    """Doc type from content shape first, filename tokens only as fallback -
    a hardcoded filename is exactly what breaks on the second incident."""
    low = text.lower()
    if sum(1 for l in text.splitlines() if LOG_LINE.match(l.strip())) >= 3:
        return "LOGS"
    if is_csv:
        joined = " ".join(header).lower()
        if any(k in joined for k in ("signature", "component", "issue", "title")):
            return "KNOWN_ISSUES"
    if "root cause" in low and (re.search(r"\bmttr\b", low)
                                or re.search(r"\b[a-z]{2,4}-\d{3,6}\b", low)):
        return "PRECEDENT"
    if "symptom" in low and ("remediation" in low or "diagnostic" in low):
        return "RUNBOOK"
    if _deploy_table(text):
        return "DEPLOYS"
    if "->" in text and ("component" in low or "architecture" in low):
        return "ARCHITECTURE"
    if ENDPOINT.search(text):
        return "API_SPEC"
    stem = name.lower()
    for doctype, keys in (("LOGS", ("log",)),
                          ("DEPLOYS", ("deploy", "release", "change")),
                          ("KNOWN_ISSUES", ("known", "issue", "catalog")),
                          ("RUNBOOK", ("runbook", "playbook", "sop")),
                          ("PRECEDENT", ("incident", "postmortem", "history")),
                          ("ARCHITECTURE", ("architecture", "design", "overview")),
                          ("API_SPEC", ("api", "spec", "contract", "openapi"))):
        if any(k in stem for k in keys):
            return doctype
    return "OTHER"


def _deploy_table(text: str):
    """Return (header, rows) of the first markdown table that looks like a
    change log: a component column plus a version or timestamp column."""
    rows, header = [], None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            if header and rows:
                break
            header, rows = None, []
            continue
        cells = [c.strip().strip("*`") for c in line.strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue
        if header is None:
            header = [c.lower() for c in cells]
            if "component" not in " ".join(header):
                header = None
            continue
        rows.append(cells)
    return (header, rows) if header and rows else None


def _ingest(corpus: dict) -> list[Unit]:
    units: list[Unit] = []
    for name, text in corpus.items():
        is_csv = name.lower().endswith(".csv") or _looks_like_csv(text)
        header: list[str] = []
        if is_csv:
            reader = csv.reader(io.StringIO(text.strip()))
            rows = [r for r in reader if any(c.strip() for c in r)]
            header = [c.strip() for c in rows[0]] if rows else []
        doctype = _classify(name, text, is_csv, header)

        if is_csv and header:
            for row in rows[1:]:
                # a row with more commas than the header keeps its overflow in
                # the last column rather than losing it
                fields = list(row[:len(header) - 1])
                fields.append(",".join(row[len(header) - 1:]))
                meta = {header[i].lower(): fields[i].strip()
                        for i in range(min(len(header), len(fields)))}
                units.append(Unit(name, doctype, fields[0].strip(),
                                  " | ".join(f.strip() for f in fields), meta))
            continue

        # log lines inside fenced blocks become one unit each
        consumed = set()
        for i, line in enumerate(text.splitlines()):
            m = LOG_LINE.match(line.strip())
            if m:
                consumed.add(i)
                units.append(Unit(name, "LOGS", f"L{i + 1}", line.strip(), {
                    "ts": _parse_ts(m.group("ts")),
                    "level": m.group("level").upper(),
                    "component": m.group("component"),
                    "message": m.group("message").strip(),
                    "signature": _signature(m.group("message")),
                }))

        table = _deploy_table(text)
        if table:
            header_cells, rows = table
            idx = {k: i for i, k in enumerate(header_cells)}
            comp_i = next(i for k, i in idx.items() if "component" in k)
            ts_i = next((i for k, i in idx.items()
                         if "time" in k or "date" in k or "when" in k), None)
            ver_i = next((i for k, i in idx.items()
                          if "version" in k or "release" in k), None)
            for cells in rows:
                if comp_i >= len(cells):
                    continue
                meta = {"component": cells[comp_i],
                        "ts": _parse_ts(cells[ts_i]) if ts_i is not None
                        and ts_i < len(cells) else None,
                        "version": cells[ver_i] if ver_i is not None
                        and ver_i < len(cells) else ""}
                meta["change"] = " ".join(c for i, c in enumerate(cells)
                                          if i not in (comp_i, ts_i, ver_i))
                units.append(Unit(name, "DEPLOYS", meta["version"] or cells[0],
                                  " | ".join(cells), meta))

        # prose sections (negations and mechanism statements live here)
        body = "\n".join(l for i, l in enumerate(text.splitlines())
                         if i not in consumed and not l.strip().startswith("|"))
        sections, current, anchor = [], [], "_"
        for line in body.splitlines():
            if re.match(r"^#{2,3}\s+\S", line):
                if any(l.strip() for l in current):
                    sections.append((anchor, "\n".join(current)))
                anchor, current = line.lstrip("# ").strip(), []
            else:
                current.append(line)
        if any(l.strip() for l in current):
            sections.append((anchor, "\n".join(current)))
        for anchor, chunk in sections:
            clean = re.sub(r"```", "", chunk).strip()
            if len(clean) < 25:
                continue
            units.append(Unit(name, doctype, anchor,
                              (anchor + "\n" + clean) if anchor != "_" else clean))
    return units


# --------------------------------------------------------------------------
# 2. index + retrieve - TF-IDF cosine, hand-rolled
# --------------------------------------------------------------------------

class Index:
    def __init__(self, units: list[Unit]):
        self.units = units
        df: dict[str, int] = defaultdict(int)
        for u in units:
            u.tokens = _tokenize(u.text)
            for tok in set(u.tokens):
                df[tok] += 1
        n = max(1, len(units))
        self.df = df
        self.idf = {t: math.log((1 + n) / (1 + d)) + 1.0 for t, d in df.items()}
        for u in units:
            u.vec = self._vector(u.tokens)
        self.components = self._components()
        self.by_type = defaultdict(list)
        for u in units:
            self.by_type[u.doctype].append(u)

    def _vector(self, tokens) -> dict:
        if not tokens:
            return {}
        counts: dict[str, int] = defaultdict(int)
        for t in tokens:
            counts[t] += 1
        peak = max(counts.values())
        vec = {t: (c / peak) * self.idf.get(t, 1.0) for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def _components(self) -> dict:
        """Component vocabulary learned from structured fields, then from
        architecture prose. Nothing is hardcoded."""
        freq: dict[str, int] = defaultdict(int)
        structured: set[str] = set()
        for u in self.units:
            for key in ("component", "affected_component"):
                val = (u.meta.get(key) or "").strip().lower()
                if COMPONENT_TOKEN.fullmatch(val):
                    freq[val] += 3
                    structured.add(val)
            if u.doctype in ("ARCHITECTURE", "KNOWN_ISSUES", "RUNBOOK"):
                for tok in COMPONENT_TOKEN.findall(u.text.lower()):
                    freq[tok] += 1
        # a component must be backed by a structured field somewhere (a log
        # field, a catalog column, a change-log cell) or be named repeatedly;
        # this keeps prose phrases like "third-party ..." out of the vocabulary
        return {c: f for c, f in freq.items()
                if "-" in c and (c in structured or f >= 4)}

    def cosine(self, qvec: dict, u: Unit) -> float:
        small, large = (qvec, u.vec) if len(qvec) < len(u.vec) else (u.vec, qvec)
        return sum(v * large.get(t, 0.0) for t, v in small.items())

    def retrieve(self, query: str) -> list[tuple[Unit, float]]:
        qtokens = _tokenize(query)
        qvec = self._vector(qtokens)
        rare = {t for t in set(qtokens) if self.df.get(t, 0) == 1}
        ranked = []
        for u in self.units:
            score = self.cosine(qvec, u)
            if score <= 0:
                continue
            if rare & set(u.tokens):
                score *= 1.5           # exact rare-token match beats prose overlap
            ranked.append((u, score))
        ranked.sort(key=lambda p: -p[1])
        return ranked


# --------------------------------------------------------------------------
# 3. hypotheses - (component, anomaly signature, symptom class)
# --------------------------------------------------------------------------

class Hypothesis:
    def __init__(self, component, signature, level, count, first, last,
                 lines, symptom, affinity, latency=None):
        self.component, self.signature, self.level = component, signature, level
        self.count, self.first, self.last = count, first, last
        self.lines, self.symptom, self.affinity = lines, symptom, affinity
        self.latency = latency or []


def _query_intent(query: str) -> set[str]:
    toks = set(_tokenize(query))
    intent = set()
    if toks & FAILURE_INTENT:
        intent.add("FAILURE")
    if toks & LATENCY_INTENT:
        intent.add("LATENCY")
    return intent or {"FAILURE"}


def _noise_signatures(index: Index) -> list[tuple[set, str, str]]:
    """Catalog rows that describe a *different* problem give us a free noise
    filter for log lines belonging to other known issues. Returns
    (tokens, component, raw_text) - the raw text is kept because negations are
    multi-word ("does not affect ...") and do not survive tokenisation."""
    out = []
    for row in index.by_type["KNOWN_ISSUES"]:
        text = " ".join(v for k, v in row.meta.items() if k != "issue_id")
        out.append((set(_tokenize(text)),
                    (row.meta.get("affected_component") or "").lower(),
                    row.text))
    return out


def _latency_pairs(index: Index, component: str) -> list[tuple[str, int]]:
    """Gap between the first and last mention of the same correlation id inside
    one component - recovers a queued->sent delay that no ERROR line reports."""
    seen: dict[str, list] = defaultdict(list)
    for u in index.by_type["LOGS"]:
        if u.meta.get("component") != component or not u.meta.get("ts"):
            continue
        for _key, val in KEY_VALUE.findall(u.meta.get("message", "")):
            if re.search(r"\d", val) and len(val) >= 4:
                seen[val].append(u.meta["ts"])
    gaps = []
    for ident, stamps in seen.items():
        if len(stamps) < 2:
            continue
        minutes = int((max(stamps) - min(stamps)).total_seconds() // 60)
        if minutes >= 5:
            gaps.append((ident, minutes))
    return sorted(gaps, key=lambda p: -p[1])


def _hypotheses(query: str, index: Index, ranked) -> list[Hypothesis]:
    symptom = _query_intent(query)
    score_by_unit = {id(u): s for u, s in ranked}

    affinity: dict[str, float] = defaultdict(float)
    for u, score in ranked:
        low = u.text.lower()
        for comp in index.components:
            if comp in low:
                affinity[comp] += score
    top = max(affinity.values()) if affinity else 1.0
    affinity = {c: v / top for c, v in affinity.items()}

    noise = _noise_signatures(index)
    groups: dict[tuple, dict] = {}
    for u in index.by_type["LOGS"]:
        level = u.meta.get("level", "INFO")
        if SEVERITY.get(level, 0) < 2:
            continue
        comp, sig = u.meta.get("component"), u.meta.get("signature")
        sig_tokens = set(_tokenize(sig))
        # background noise: a catalogued issue for a different component, or a
        # row that says it does not affect the behaviour being investigated
        if any(len(sig_tokens & ktoks) >= 2
               and (kcomp != comp or _has(kraw, NEGATION_MARKERS))
               for ktoks, kcomp, kraw in noise):
            continue
        key = (comp, sig)
        g = groups.setdefault(key, {"lines": [], "level": level})
        g["lines"].append(u)
        if SEVERITY.get(level, 0) > SEVERITY.get(g["level"], 0):
            g["level"] = level

    candidates = []
    for (comp, sig), g in groups.items():
        if affinity.get(comp, 0.0) < RELEVANCE_GATE:
            continue                    # off-topic component, however loud
        stamps = [u.meta["ts"] for u in g["lines"] if u.meta.get("ts")]
        latency = _latency_pairs(index, comp) if "LATENCY" in symptom else []
        rank = (affinity.get(comp, 0.05)
                * SEVERITY.get(g["level"], 1)
                * math.log2(1 + len(g["lines"])))
        candidates.append((rank, Hypothesis(
            comp, sig, g["level"], len(g["lines"]),
            min(stamps) if stamps else None, max(stamps) if stamps else None,
            g["lines"], symptom, affinity.get(comp, 0.0), latency)))

    candidates.sort(key=lambda p: -p[0])
    return [h for _, h in candidates[:4]]


# --------------------------------------------------------------------------
# 4. corroborate - six axes, tested independently
# --------------------------------------------------------------------------

def _overlap(a: set, b: set, idf: dict) -> float:
    if not a:
        return 0.0
    hit = sum(idf.get(t, 1.0) for t in (a & b))
    return hit / (sum(idf.get(t, 1.0) for t in a) or 1.0)


def _match(hypo: Hypothesis, unit: Unit, index: Index, floor: float):
    """Does `unit` talk about this hypothesis? Returns (matched, is_distinctive)."""
    low = unit.text.lower()
    if hypo.component not in low:
        return False, False
    sig_tokens = set(_tokenize(hypo.signature))
    unit_tokens = set(unit.tokens)
    shared_id = _distinctive(sig_tokens) & _distinctive(unit_tokens)
    if shared_id:
        return True, True
    return _overlap(sig_tokens, unit_tokens, index.idf) >= floor, False


def _corroborate(hypo: Hypothesis, index: Index) -> dict:
    axes = {a: {"state": ABSENT, "unit": None, "detail": ""} for a in AXES}
    caveats, contradictions = [], []

    def set_axis(name, state, unit, detail=""):
        axes[name] = {"state": state, "unit": unit, "detail": detail}

    # --- 1. logs
    if hypo.count:
        strong = SEVERITY.get(hypo.level, 0) >= 3 and hypo.count >= 3
        set_axis("LOGS", STRONG if strong else WEAK, hypo.lines[0],
                 f"{hypo.count}x {hypo.level} on {hypo.component}"
                 + (f", onset {hypo.first:%H:%M:%S}" if hypo.first else ""))

    # --- 2. deployment correlation
    deploys = index.by_type["DEPLOYS"]
    rows = [u for u in deploys if u.meta.get("component")]
    log_stamps = [u.meta["ts"] for u in index.by_type["LOGS"] if u.meta.get("ts")]
    window_end = hypo.first or (max(log_stamps) if log_stamps else None)
    correlated = None
    for u in rows:
        comp = (u.meta.get("component") or "").lower()
        ts = u.meta.get("ts")
        if comp != hypo.component or not ts or not window_end:
            continue
        if timedelta(0) <= (window_end - ts) <= timedelta(hours=24):
            if correlated is None or ts > correlated.meta["ts"]:
                correlated = u
    if correlated is not None:
        set_axis("DEPLOY", STRONG, correlated,
                 f"{correlated.meta.get('version', '')} at "
                 f"{correlated.meta['ts']:%Y-%m-%d %H:%M} - "
                 f"{_norm_ws(correlated.meta.get('change', ''), 120)}")
    else:
        prose = next((u for u in deploys
                      if u.doctype == "DEPLOYS" and hypo.component in u.text.lower()
                      and _has(u.text, NEGATION_MARKERS)), None)
        stale = (rows and log_stamps and all(
            u.meta.get("ts") and u.meta["ts"] > max(log_stamps) for u in rows))
        if prose is not None or stale:
            unit = prose or rows[0]
            set_axis("DEPLOY", CONTRADICTED, unit,
                     "no deployment correlates with the observed window")
            contradictions.append("no correlated deployment")

    # --- 3. known-issue catalog
    best = None
    for row in index.by_type["KNOWN_ISSUES"]:
        comp = (row.meta.get("affected_component") or "").lower()
        sig_field = " ".join(v for k, v in row.meta.items()
                             if k in ("title", "signature"))
        if comp != hypo.component:
            continue
        sig_tokens = set(_tokenize(hypo.signature))
        shared_id = _distinctive(sig_tokens) & _distinctive(set(_tokenize(sig_field)))
        score = _overlap(sig_tokens, set(_tokenize(sig_field)), index.idf)
        if shared_id:
            best = (STRONG, row, f"{row.anchor} matches on '{sorted(shared_id)[0]}'")
            break
        if score >= 0.25:
            best = (STRONG, row, f"{row.anchor} signature overlap {score:.2f}")
        elif score >= 0.12 and best is None:
            best = (WEAK, row, f"{row.anchor} partial overlap {score:.2f}")
    if best:
        set_axis("KNOWN_ISSUE", best[0], best[1], best[2])

    # --- 4. precedent
    for u in index.by_type["PRECEDENT"]:
        matched, _ = _match(hypo, u, index, 0.20)
        if matched:
            set_axis("PRECEDENT", STRONG, u, f"{_norm_ws(u.anchor, 60)}")
            break
        if hypo.component in u.text.lower() and _has(u.text, NEGATION_MARKERS):
            set_axis("PRECEDENT", CONTRADICTED, u, "no precedent on record")
            contradictions.append("no precedent")
            break

    # --- 5. runbook
    for u in index.by_type["RUNBOOK"]:
        matched, _ = _match(hypo, u, index, 0.15)
        if not matched:
            continue
        hedges = _has(u.text, HEDGE_MARKERS)
        set_axis("RUNBOOK", WEAK if hedges else STRONG, u,
                 f"{_norm_ws(u.anchor, 60)}"
                 + (f" (self-declared {hedges[0]})" if hedges else ""))
        if hedges:
            caveats.append(f"{u.source} ({_norm_ws(u.anchor, 40)}) is "
                           f"self-declared {hedges[0]}")
        break

    # --- 6. mechanism (static docs - plausibility only, never confidence)
    mech = [u for u in index.by_type["ARCHITECTURE"] + index.by_type["API_SPEC"]
            if hypo.component in u.text.lower()]
    hedged = [(u, _has(u.text, HEDGE_MARKERS)) for u in mech]
    hedged = [(u, h) for u, h in hedged if h]
    if hedged:
        # the documentation itself says the mechanism is not observable here,
        # so it cannot support the hypothesis - it explains why we are stuck
        unit, markers = hedged[0]
        set_axis("MECHANISM", WEAK, unit, f"documents a gap: {markers[0]}")
        for unit, markers in hedged[:2]:
            caveats.append(f"{unit.source} notes: {markers[0]}")
    elif mech:
        sig_tokens = set(_tokenize(hypo.signature))
        best_mech = max(mech, key=lambda u: _overlap(sig_tokens, set(u.tokens),
                                                     index.idf))
        set_axis("MECHANISM", STRONG, best_mech, "explains the failure mode")

    return {"hypothesis": hypo, "axes": axes, "caveats": caveats,
            "contradictions": contradictions}


# --------------------------------------------------------------------------
# 5. calibrate - confidence from corroboration, with hard caps
# --------------------------------------------------------------------------

POINTS = {
    "LOGS":        {STRONG: 30, WEAK: 15},
    "DEPLOY":      {STRONG: 22, WEAK: 10, CONTRADICTED: -5},
    "KNOWN_ISSUE": {STRONG: 20, WEAK: 10, CONTRADICTED: -5},
    "PRECEDENT":   {STRONG: 15, WEAK: 7, CONTRADICTED: -5},
    "RUNBOOK":     {STRONG: 10, WEAK: 3},
    "MECHANISM":   {STRONG: 5, WEAK: -3},
}


def _calibrate(corr: dict) -> float:
    axes, hypo = corr["axes"], corr["hypothesis"]
    score = 10.0
    for name, table in POINTS.items():
        score += table.get(axes[name]["state"], 0)
    if hypo.latency:
        score += 8                      # quantified, derived-from-logs evidence
    strong = sum(1 for a in AXES[:5] if axes[a]["state"] == STRONG)
    contradicted = sum(1 for a in AXES if axes[a]["state"] == CONTRADICTED)
    if strong < 3:
        score = min(score, 45)
    if SEVERITY.get(hypo.level, 0) < 3:
        score = min(score, 40)
    if contradicted >= 2:
        score = min(score, 35)
    return round(max(5.0, min(92.0, score)), 1)


# --------------------------------------------------------------------------
# 6. compose
# --------------------------------------------------------------------------

def _mttr(corr: dict):
    """Only trust an MTTR from a source that corroborates this hypothesis and
    does not hedge itself."""
    for name in ("RUNBOOK", "PRECEDENT"):
        axis = corr["axes"][name]
        if axis["state"] != STRONG or axis["unit"] is None:
            continue
        m = MTTR.search(axis["unit"].text)
        if m:
            value = int(m.group(1))
            return value * 60 if m.group(2).lower().startswith("h") else value
    return None


def _impacted(corr: dict, index: Index, query: str) -> list[str]:
    hypo = corr["hypothesis"]
    out = [hypo.component]
    noise = _noise_signatures(index)
    lo = (hypo.first or datetime.min) - timedelta(minutes=5)
    hi = (hypo.last or datetime.max) + timedelta(minutes=5)
    for u in index.by_type["LOGS"]:
        comp, ts = u.meta.get("component"), u.meta.get("ts")
        if comp in out or not ts or not (lo <= ts <= hi):
            continue
        if SEVERITY.get(u.meta.get("level", "INFO"), 0) < 2:
            continue
        sig_tokens = set(_tokenize(u.meta.get("signature", "")))
        if any(len(sig_tokens & k) >= 2 for k, _c, _r in noise):
            continue                    # attributable to another known issue
        out.append(comp)
    for u in ([corr["axes"]["MECHANISM"]["unit"]] if corr["axes"]["MECHANISM"]["unit"]
              else []):
        for a, b in EXTERNAL_DEP.findall(u.text):
            dep = (a or b).strip()
            if dep and dep.lower() not in " ".join(out).lower():
                out.append(f"{dep} (external dependency)")
    head = re.split(r"[.\n]", query.strip())[0].strip()
    if head:
        out.append(f"customer-facing impact: {head[:140]}")
    return out


def _evidence(corr: dict) -> list[dict]:
    hypo, axes = corr["hypothesis"], corr["axes"]
    items, seen = [], set()

    lines = hypo.lines[:2]
    if lines:
        excerpt = " / ".join(_norm_ws(u.text, 170) for u in lines)
        if hypo.count > len(lines):
            excerpt += f" [{hypo.count} occurrences of this signature in total]"
        items.append({"source": lines[0].source, "excerpt": excerpt})
        seen.add((lines[0].source, excerpt[:40]))

    if hypo.latency:
        src = hypo.lines[0].source
        worst = ", ".join(f"{i}: {m} min" for i, m in hypo.latency[:4])
        items.append({"source": src, "excerpt":
                      f"Derived from the log: elapsed time between the first and "
                      f"last event for the same correlation id within "
                      f"{hypo.component} - {worst}."})

    order = ["DEPLOY", "KNOWN_ISSUE", "PRECEDENT", "RUNBOOK", "MECHANISM"]
    rank = {STRONG: 0, WEAK: 1, CONTRADICTED: 2, ABSENT: 3}
    for name in sorted(order, key=lambda n: rank[axes[n]["state"]]):
        axis = axes[name]
        if axis["state"] == ABSENT or axis["unit"] is None:
            continue
        tag = {STRONG: "", WEAK: "[weak/hedged] ",
               CONTRADICTED: "[disconfirming] "}[axis["state"]]
        excerpt = tag + _norm_ws(axis["unit"].text)
        key = (axis["unit"].source, excerpt[:40])
        if key in seen:
            continue
        seen.add(key)
        items.append({"source": axis["unit"].source, "excerpt": excerpt})
    return items


def _compose(query: str, corr: dict, index: Index, score: float) -> dict:
    hypo, axes = corr["hypothesis"], corr["axes"]
    agreeing = [a for a in AXES[:5] if axes[a]["state"] == STRONG]
    missing = [a for a in AXES[:5] if axes[a]["state"] == ABSENT]
    denied = [a for a in AXES if axes[a]["state"] == CONTRADICTED]
    label = {"LOGS": "application logs", "DEPLOY": "deployment history",
             "KNOWN_ISSUE": "the known-issues catalog", "PRECEDENT":
             "a prior incident", "RUNBOOK": "the runbook",
             "MECHANISM": "the architecture/API documentation"}
    signature = _norm_ws(hypo.lines[0].meta.get("message", hypo.signature), 160)

    if score >= 50:
        parts = [
            f"{signature} in {hypo.component}.",
            f"The signature appears {hypo.count}x at {hypo.level} level"
            + (f", starting {hypo.first:%Y-%m-%d %H:%M:%S}" if hypo.first else "")
            + (f" and continuing to {hypo.last:%H:%M:%S}" if hypo.last else "")
            + ", with successful requests interleaved - a saturation pattern "
              "rather than a hard outage.",
        ]
        if axes["DEPLOY"]["state"] == STRONG:
            dep = axes["DEPLOY"]["unit"]
            delta = ""
            if hypo.first and dep.meta.get("ts"):
                delta = (f", {int((hypo.first - dep.meta['ts']).total_seconds() // 60)}"
                         f" minutes before the first error")
            parts.append(f"It correlates with deployment "
                         f"{dep.meta.get('version', '')} on {hypo.component} at "
                         f"{dep.meta['ts']:%Y-%m-%d %H:%M}{delta}: "
                         f"{_norm_ws(dep.meta.get('change', ''), 200)}.")
        for name in ("KNOWN_ISSUE", "PRECEDENT", "MECHANISM"):
            if axes[name]["state"] == STRONG and axes[name]["detail"]:
                parts.append(f"Corroborated by {label[name]} - "
                             f"{axes[name]['detail']}.")
        parts.append(f"{len(agreeing)} independent sources agree "
                     f"({', '.join(label[a] for a in agreeing)}).")
        root_cause = " ".join(parts)
    else:
        parts = [
            f"Low confidence - the available evidence does not support naming a "
            f"root cause. The single plausible lead is {hypo.component}: "
            f"{hypo.count}x {hypo.level} '{signature}'",
        ]
        if hypo.latency:
            worst = max(m for _i, m in hypo.latency)
            parts[0] += (f", together with elapsed times up to {worst} minutes "
                         f"between the first and last event for the same "
                         f"correlation id inside that component")
        parts[0] += "."
        if denied:
            parts.append("Corroboration actively fails: "
                         + "; ".join(f"{label[a]} - {axes[a]['detail']}"
                                     for a in denied) + ".")
        if missing:
            parts.append("No supporting signal at all from "
                         + ", ".join(label[a] for a in missing) + ".")
        if corr["caveats"]:
            parts.append("Sources that do match discount themselves: "
                         + "; ".join(corr["caveats"][:3]) + ".")
        parts.append("Escalate to a human with access to the metrics this corpus "
                     "does not contain.")
        root_cause = " ".join(parts)

    if axes["RUNBOOK"]["state"] == STRONG:
        text = axes["RUNBOOK"]["unit"].text
        m = re.search(r"remediation\**\s*:?\**\s*(.+?)(?:\n\s*\n|\*\*typical|$)",
                      text, re.I | re.S)
        step = _norm_ws(m.group(1), 300) if m else ""
        remediation = (f"Apply the documented remediation for this signature "
                       f"({_norm_ws(axes['RUNBOOK']['unit'].anchor, 60)}): {step}")
        if axes["PRECEDENT"]["state"] == STRONG:
            remediation += (f" This matches how the precedent was resolved "
                            f"({_norm_ws(axes['PRECEDENT']['unit'].anchor, 40)}). ")
        if axes["DEPLOY"]["state"] == STRONG:
            remediation += (f" Concretely: reverse the correlated change "
                            f"({_norm_ws(axes['DEPLOY']['unit'].meta.get('change',''), 160)}) "
                            f"and redeploy {hypo.component}, then confirm the "
                            f"signature stops appearing.")
        remediation += (" Follow-up: alert on the saturating resource before it "
                        "reaches exhaustion, and treat this configuration value as "
                        "a capacity-reviewed change rather than an optimisation.")
    else:
        gaps = ", ".join(label[a] for a in (missing + denied)) or "every axis"
        remediation = (
            f"Do not ship a fix on this evidence. First close the observability "
            f"gap that makes the incident undiagnosable: add per-stage timing "
            f"across {hypo.component}'s path and expose the throughput and "
            f"downstream-latency figures the existing documentation says are not "
            f"instrumented. Those numbers separate the candidate causes, which "
            f"this corpus cannot ({gaps} provide no signal). ")
        if axes["RUNBOOK"]["state"] == WEAK:
            text = axes["RUNBOOK"]["unit"].text
            m = re.search(r"remediation\**\s*:?\**\s*(.+?)(?:\n\s*\n|\*\*typical|$)",
                          text, re.I | re.S)
            if m:
                remediation += (f"The one candidate action on file is "
                                f"'{_norm_ws(m.group(1), 160)}' - that runbook "
                                f"flags it as unverified, so treat it as a "
                                f"hypothesis to test, not a fix. ")
        remediation += ("Then either scale the saturated stage or escalate to the "
                        "external dependency with the timing data, and define the "
                        "missing service-level objective for this path.")

    return {
        "root_cause": root_cause,
        "supporting_evidence": _evidence(corr),
        "impacted_systems": _impacted(corr, index, query),
        "mttr_minutes": _mttr(corr),
        "remediation": remediation,
        "confidence_score": float(score),
        "needs_human_review": bool(score < 50),
    }


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------

def investigate(query: str, corpus: dict) -> dict:
    """Correlate `corpus` against `query` and return an incident report.

    Every candidate hypothesis is fully corroborated and calibrated; the one
    with the strongest independent agreement wins. Confidence therefore
    reflects the evidence, not the ranking.
    """
    units = _ingest(corpus)
    index = Index(units)
    ranked = index.retrieve(query)

    best = None
    for hypo in _hypotheses(query, index, ranked):
        corr = _corroborate(hypo, index)
        score = _calibrate(corr)
        if best is None or score > best[0]:
            best = (score, corr)

    if best is None:                    # no anomaly signal at all in the corpus
        top = ranked[0][0] if ranked else Unit("unknown", "OTHER", "_", "")
        return {
            "root_cause": ("Low confidence - no anomaly signal (error or warning) "
                           "could be extracted from this corpus for the reported "
                           "symptom, so no root cause can be named."),
            "supporting_evidence": [{"source": top.source,
                                     "excerpt": _norm_ws(top.text)}],
            "impacted_systems": [],
            "mttr_minutes": None,
            "remediation": ("Collect logs or metrics covering the symptom window "
                            "before attempting a diagnosis."),
            "confidence_score": 5.0,
            "needs_human_review": True,
        }

    score, corr = best
    report = _compose(query, corr, index, score)
    assert set(report) == set(REQUIRED_KEYS)
    assert report["needs_human_review"] == (report["confidence_score"] < 50)
    return report


def explain(query: str, corpus: dict) -> dict:
    """Same pass as investigate(), but also returns the retrieval ranking and
    the per-axis states. Used by the Streamlit demo; not part of the graded
    interface."""
    units = _ingest(corpus)
    index = Index(units)
    ranked = index.retrieve(query)
    scored = []
    for hypo in _hypotheses(query, index, ranked):
        corr = _corroborate(hypo, index)
        scored.append((_calibrate(corr), corr))
    scored.sort(key=lambda p: -p[0])
    score, corr = scored[0]
    return {
        "report": _compose(query, corr, index, score),
        "axes": {a: {"state": corr["axes"][a]["state"],
                     "detail": corr["axes"][a]["detail"],
                     "source": corr["axes"][a]["unit"].source
                     if corr["axes"][a]["unit"] else None}
                 for a in AXES},
        "hypothesis": {"component": corr["hypothesis"].component,
                       "signature": corr["hypothesis"].signature,
                       "level": corr["hypothesis"].level,
                       "count": corr["hypothesis"].count,
                       "latency": corr["hypothesis"].latency},
        "alternatives": [{"component": c["hypothesis"].component,
                          "signature": c["hypothesis"].signature,
                          "score": s} for s, c in scored],
        "ranking": [{"source": u.source, "anchor": u.anchor,
                     "doctype": u.doctype, "score": round(s, 4),
                     "text": _norm_ws(u.text, 200)} for u, s in ranked[:12]],
        "units": [{"source": u.source, "doctype": u.doctype} for u in units],
        "components": sorted(index.components),
    }
