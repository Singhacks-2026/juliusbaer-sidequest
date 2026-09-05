"""Use case 2 - production incident investigator.

    investigate(query, corpus) -> structured incident report

Pipeline, in the shape the starter suggests:

    ingest     corpus -> small retrieval units (one per log line, CSV row,
               table row or prose paragraph), each tagged with its source
               file and document type
    retrieve   TF-IDF + cosine ranking of units against the query
    correlate  pick candidate components from the top hits, then test
               each against six *independent* evidence axes (logs, known
               issues, deployments, precedent, runbook, topology) and
               collect explicit counter-evidence
    calibrate  confidence from how many independent axes agree, minus a
               penalty for hedges/negations found in the sources - never
               from how relevant the top-ranked hit felt

Standard library only. Nothing is keyed on an incident-specific string:
every rule is expressed over document types, log levels, exception
tokens, timestamps and hyphenated component names, so the same code has
to earn (or fail to earn) its confidence on any incident that ships the
same document types.
"""
from __future__ import annotations

import csv
import io
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Patterns and vocabulary
# ---------------------------------------------------------------------------

LOG_LINE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+"
    r"(?P<level>TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL)\s+"
    r"(?P<component>\S+)\s+(?P<msg>.*)$"
)
TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?")
COMPONENT = re.compile(r"\b[a-z]+(?:-[a-z]+)+\b")  # e.g. payment-gateway-adapter
EXCEPTION = re.compile(r"\b[A-Za-z]+(?:Exception|Error)\b")
REASON = re.compile(r"\breason=([A-Z_]+)")
REF_ID = re.compile(r"\b(?:RB|INC|KI)-\d+\b")
MTTR = re.compile(r"MTTR[^0-9]{0,24}(\d+)\s*min", re.I)
VERSION = re.compile(r"\bv\d+\.\d+(?:\.\d+)?\b")
TOKEN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")
OPAQUE = re.compile(r"^[a-z]{2,4}-\d+$|^\d+[a-z]*$")  # ORD-88350, 5000ms, 340
LABEL = re.compile(r"^\**[A-Za-z /]+\**:\s*")  # "**Remediation**: ..."

ERROR_LEVELS = ("ERROR", "FATAL")
WARN_LEVELS = ("WARN", "WARNING")
LEVEL_WEIGHT = {"ERROR": 3.0, "FATAL": 3.0, "WARN": 1.5, "WARNING": 1.5}

# Phrases that mark a source as *hedging* its own claim.
HEDGE_CUES = (
    "unconfirmed", "unverified", "not confirmed", "may not apply",
    "incomplete", "not currently", "no documented", "not instrumented",
    "outside this service", "gap worth noting",
)
# Phrases that mark a source as *denying* a link to the component.
NEGATION_CUES = (
    "no previous", "no prior", "first recorded", "no deployment",
    "no other runbook", "unrelated", "not involve", "does not affect",
    "none of which",
)

STOPWORDS = set(
    """
    a an the and or of to in on at by for with from as is are was were be
    been being it its this that these those there here after before during
    over under than then also any some all not no yes very more most less
    what which who whom whose why how when where using via per
    identify probable root cause supporting evidence impacted components
    component recommended remediation mean time recover systems system
    yesterday sometimes customers reporting
    """.split()
)

TOP_K = 25                # units considered when nominating components
RELEVANCE_GATE = 0.6      # candidates must be >= 60% as relevant as the leader
DEPLOY_WINDOW = timedelta(days=3)
COUPLING_WINDOW = timedelta(seconds=5)

# Axis weights. They sum to 97; the score is capped at 92 because a
# document corpus alone never proves a root cause.
WEIGHTS = {
    "logs": {"strong": 30, "moderate": 22, "weak": 15},
    "known_issue": 18,
    "deployment": 20,
    "precedent": 14,
    "runbook": 10,
    "topology": 5,
}
CAP, FLOOR = 92, 5
HEDGE_PENALTY_PER_SOURCE, HEDGE_PENALTY_CAP = 3, 12


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Unit:
    """One retrievable piece of a document."""
    uid: str
    source: str            # corpus filename
    doc_type: str          # logs | deployments | known_issues | precedent | runbooks | topology | other
    text: str
    section: str = ""      # nearest markdown heading
    kind: str = "prose"    # prose | log | row
    ts: Optional[datetime] = None
    level: str = ""
    component: str = ""
    message: str = ""
    fields: Dict[str, str] = field(default_factory=dict)

    def mentions(self, component: str) -> bool:
        return component in self.text.lower()


@dataclass
class Axis:
    """Verdict of one independent evidence axis for one hypothesis."""
    name: str
    hit: bool
    strength: str = ""
    units: List[Unit] = field(default_factory=list)
    hedged: bool = False
    section: str = ""

    def weight(self) -> int:
        if not self.hit:
            return 0
        w = WEIGHTS[self.name]
        return w[self.strength] if isinstance(w, dict) else w


@dataclass
class Hypothesis:
    component: str
    relevance: float
    symptom: List[Unit]          # ERROR (or, failing that, WARN) lines from the component
    coupled: List[Unit]          # ERROR lines from other components within seconds
    strong_sig: List[str]        # exception classes / reason codes
    weak_sig: List[str]          # informative words from the symptom messages
    first_ts: Optional[datetime]
    axes: Dict[str, Axis] = field(default_factory=dict)
    counter: List[Unit] = field(default_factory=list)
    deploy_lag: Optional[int] = None
    nominees: List[Tuple[str, float]] = field(default_factory=list)

    def raw_score(self) -> int:
        return sum(a.weight() for a in self.axes.values())

    def hedge_sources(self) -> Set[str]:
        srcs = {u.source for u in self.counter}
        srcs |= {a.units[0].source for a in self.axes.values() if a.hit and a.hedged}
        return srcs


# ---------------------------------------------------------------------------
# 1. Ingest
# ---------------------------------------------------------------------------

def _doc_type(filename: str) -> str:
    """Classify a corpus file by name. Both incidents ship the same document
    types, which is what makes cross-type corroboration meaningful."""
    name = re.sub(r"\.[^.]+$", "", filename).lower()
    if "log" in name:
        return "logs"
    if "deploy" in name or "release" in name:
        return "deployments"
    if "known" in name or "issue" in name:
        return "known_issues"
    if "previous" in name or "incident" in name or "postmortem" in name:
        return "precedent"
    if "runbook" in name or "playbook" in name:
        return "runbooks"
    if any(k in name for k in ("architect", "api", "spec", "topology")):
        return "topology"
    return "other"


def _parse_ts(s: str) -> Optional[datetime]:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def _first_component(text: str) -> str:
    m = COMPONENT.search(text.lower())
    return m.group(0) if m else ""


def _split_cells(line: str) -> List[str]:
    return [re.sub(r"\*+", "", c).strip() for c in line.strip().strip("|").split("|")]


def _ingest_csv(fname: str, text: str) -> List[Unit]:
    units = []
    for i, row in enumerate(csv.DictReader(io.StringIO(text))):
        fields = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
        ref = next((v for v in fields.values() if REF_ID.fullmatch(v)), str(i))
        body = " | ".join(f"{k}: {v}" for k, v in fields.items() if v)
        comp = fields.get("affected_component") or _first_component(body)
        units.append(Unit(uid=f"{fname}#{ref}", source=fname, doc_type=_doc_type(fname),
                          text=body, kind="row", component=comp.lower(), fields=fields))
    return units


def _units_from_block(fname: str, dtype: str, section: str, lines: List[str], n: int) -> List[Unit]:
    out: List[Unit] = []
    matches = [(l, LOG_LINE.match(l.strip())) for l in lines]

    if any(m for _, m in matches):                     # log lines -> one unit each
        rest = []
        for l, m in matches:
            if not m:
                rest.append(l)
                continue
            out.append(Unit(uid=f"{fname}#{n + len(out)}", source=fname, doc_type=dtype,
                            text=l.strip(), section=section, kind="log",
                            ts=_parse_ts(m["ts"]), level=m["level"].upper(),
                            component=m["component"].lower(), message=m["msg"]))
        lines = rest
        if not lines:
            return out

    if len(lines) >= 2 and all(l.strip().startswith("|") for l in lines):   # table -> row units
        header = [h.lower() for h in _split_cells(lines[0])]
        for l in lines[1:]:
            cells = _split_cells(l)
            if all(re.fullmatch(r":?-+:?", c) for c in cells if c):
                continue
            body = " | ".join(cells)
            out.append(Unit(uid=f"{fname}#{n + len(out)}", source=fname, doc_type=dtype,
                            text=body, section=section, kind="row",
                            component=_first_component(body), fields=dict(zip(header, cells))))
        return out

    body = re.sub(r"\s+", " ", " ".join(l.strip() for l in lines)).strip()
    if body:
        out.append(Unit(uid=f"{fname}#{n + len(out)}", source=fname, doc_type=dtype,
                        text=body, section=section, kind="prose"))
    return out


def _ingest_markdown(fname: str, text: str) -> List[Unit]:
    units: List[Unit] = []
    dtype = _doc_type(fname)
    section, in_fence, block = "", False, []  # type: str, bool, List[str]

    def flush():
        if block:
            units.extend(_units_from_block(fname, dtype, section, list(block), len(units)))
            block.clear()

    for raw in text.splitlines():
        s = raw.strip()
        if s.startswith("```"):
            flush()
            in_fence = not in_fence
        elif not in_fence and s.startswith("#"):
            flush()
            section = s.lstrip("#").strip()
        elif not s:
            flush()
        else:
            if not in_fence and re.match(r"[-*+]\s", s):   # each bullet is its own unit
                flush()
            block.append(raw.rstrip())
    flush()
    return units


def _ingest_corpus(corpus: dict) -> List[Unit]:
    units: List[Unit] = []
    for fname, text in corpus.items():
        if fname.lower().endswith(".csv"):
            units.extend(_ingest_csv(fname, text))
        else:
            units.extend(_ingest_markdown(fname, text))
    return units


# ---------------------------------------------------------------------------
# 2. Retrieve (TF-IDF + cosine)
# ---------------------------------------------------------------------------

def _stem(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _tokenize(text: str) -> List[str]:
    out = []
    for tok in TOKEN.findall(text.lower()):
        parts = re.split(r"[-_]", tok)
        for c in ([tok] + parts if len(parts) > 1 else [tok]):
            if len(c) < 2 or c in STOPWORDS or c.isdigit():
                continue
            out.append(_stem(c))
    return out


class _Index:
    def __init__(self, units: List[Unit]):
        self.units = units
        docs = [Counter(_tokenize(u.section + " " + u.text)) for u in units]
        df: Counter = Counter()
        for d in docs:
            df.update(d.keys())
        n = len(units)
        self.idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        self.vecs = [self._vec(d) for d in docs]

    def _vec(self, counts: Counter) -> Dict[str, float]:
        total = sum(counts.values()) or 1
        v = {t: (c / total) * self.idf.get(t, 1.0) for t, c in counts.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def rank(self, query: str) -> List[Tuple[Unit, float]]:
        q = self._vec(Counter(_tokenize(query)))
        scored = [(u, sum(q[t] * v.get(t, 0.0) for t in q)) for u, v in zip(self.units, self.vecs)]
        return sorted(scored, key=lambda x: -x[1])


def _retrieve_relevant_documents(query: str, units: List[Unit]) -> List[Tuple[Unit, float]]:
    return _Index(units).rank(query)


# ---------------------------------------------------------------------------
# 3. Correlate
# ---------------------------------------------------------------------------

def _signature(symptom: List[Unit]) -> Tuple[List[str], List[str]]:
    """Strong signature = exception classes / reason codes; weak signature =
    the informative words of the symptom messages."""
    strong: List[str] = []
    weak: Counter = Counter()
    for u in symptom:
        for s in EXCEPTION.findall(u.message) + REASON.findall(u.message):
            if s not in strong:
                strong.append(s)
        weak.update(t for t in _tokenize(u.message) if not OPAQUE.match(t))
    return strong, [t for t, _ in weak.most_common(8)]


def _matches(unit: Unit, strong: List[str], weak: List[str], query_terms: Set[str]) -> bool:
    low = unit.text.lower()
    if any(s.lower() in low for s in strong):
        return True
    toks = set(_tokenize(unit.text))
    return len(toks & set(weak)) >= 2 or len(toks & query_terms) >= 2


def _has_cue(text: str, cues: Tuple[str, ...]) -> bool:
    low = text.lower()
    return any(c in low for c in cues)


def _section_units(units: List[Unit], u: Unit) -> List[Unit]:
    return [x for x in units if x.source == u.source and x.section == u.section]


def _labeled(section_units: List[Unit], label: str) -> Optional[Unit]:
    for u in section_units:
        if re.match(rf"^\**{label}", u.text, re.I):
            return u
    return None


def _strip_label(text: str) -> str:
    return LABEL.sub("", text, count=1)


def _nominate_components(units: List[Unit], ranked: List[Tuple[Unit, float]]) -> List[Tuple[str, float]]:
    """Components are only things the corpus itself treats as components:
    log emitters, known-issue owners, deployment targets."""
    known = {u.component for u in units if u.component and u.kind in ("log", "row")}
    failures = [u for u in units if u.kind == "log" and u.ts and u.level in ERROR_LEVELS + WARN_LEVELS]
    score: Counter = Counter()
    for u, s in ranked[:TOP_K]:
        if s <= 0:
            continue
        w = s * LEVEL_WEIGHT.get(u.level, 1.0 if u.kind != "log" else 0.6)
        comps = set(COMPONENT.findall(u.text.lower())) | ({u.component} if u.component else set())
        if u.kind == "log" and u.ts and u.level in ERROR_LEVELS + WARN_LEVELS:
            # Failures logged in the same instant are one event: a downstream
            # symptom line is evidence for the upstream component too.
            comps |= {f.component for f in failures if abs(f.ts - u.ts) <= COUPLING_WINDOW}
        for c in comps:
            if c in known:
                score[c] += w
    return score.most_common()


def _evaluate(component: str, relevance: float, units: List[Unit], query_terms: Set[str]) -> Hypothesis:
    logs = [u for u in units if u.kind == "log" and u.ts]
    own = [u for u in logs if u.component == component]
    errors = [u for u in own if u.level in ERROR_LEVELS]
    warns = [u for u in own if u.level in WARN_LEVELS]
    symptom = errors or warns
    first_ts = min(u.ts for u in symptom) if symptom else None
    strong, weak = _signature(symptom)
    coupled = [u for u in logs if u.level in ERROR_LEVELS and u.component != component
               and any(abs(u.ts - s.ts) <= COUPLING_WINDOW for s in symptom)]
    h = Hypothesis(component, relevance, symptom, coupled, strong, weak, first_ts)
    by_type: Dict[str, List[Unit]] = {}
    for u in units:
        by_type.setdefault(u.doc_type, []).append(u)

    # -- logs: repeated ERROR lines carrying an exception/reason code are strong
    if errors:
        strength = "strong" if len(errors) >= 2 and strong else "moderate"
        h.axes["logs"] = Axis("logs", True, strength, errors)
    elif warns:
        h.axes["logs"] = Axis("logs", True, "weak", warns)
    else:
        h.axes["logs"] = Axis("logs", False)

    # -- known issue: a row owned by the component whose signature matches the symptom
    rows = [u for u in by_type.get("known_issues", []) if u.mentions(component)
            and _matches(u, strong, weak, query_terms) and not _has_cue(u.text, NEGATION_CUES)]
    h.axes["known_issue"] = Axis("known_issue", bool(rows), units=rows[:1],
                                 hedged=bool(rows) and _has_cue(rows[0].text, HEDGE_CUES))

    # -- deployment: a deploy to the component shortly *before* the first symptom
    best = None
    for u in by_type.get("deployments", []):
        m = TIMESTAMP.search(u.text)
        ts = _parse_ts(m.group(0)) if m else None
        if not ts or component not in COMPONENT.findall(u.text.lower()) or not first_ts:
            continue
        if ts <= first_ts <= ts + DEPLOY_WINDOW and (best is None or ts > best[1]):
            best = (u, ts)
    if best:
        h.deploy_lag = int((first_ts - best[1]).total_seconds() // 60)
    h.axes["deployment"] = Axis("deployment", bool(best), units=[best[0]] if best else [])

    # -- precedent: a prior incident with the same signature, not phrased as a denial
    prec = [u for u in by_type.get("precedent", []) if u.mentions(component)
            and _matches(u, strong, weak, query_terms) and not _has_cue(u.text, NEGATION_CUES)]
    if prec:
        sec = _section_units(units, prec[0])
        h.axes["precedent"] = Axis("precedent", True, units=prec[:1], section=prec[0].section,
                                   hedged=any(_has_cue(x.text, HEDGE_CUES) for x in sec))
    else:
        h.axes["precedent"] = Axis("precedent", False)

    # -- runbook: a runbook whose symptoms match
    rb = [u for u in by_type.get("runbooks", []) if u.mentions(component)
          and _matches(u, strong, weak, query_terms) and not _has_cue(u.text, NEGATION_CUES)]
    if rb:
        sec = _section_units(units, rb[0])
        h.axes["runbook"] = Axis("runbook", True, units=rb[:1], section=rb[0].section,
                                 hedged=any(_has_cue(x.text, HEDGE_CUES) for x in sec))
    else:
        h.axes["runbook"] = Axis("runbook", False)

    # -- topology: architecture / API docs describe the component (context, low weight)
    topo = [u for u in by_type.get("topology", []) if u.mentions(component)]
    topo.sort(key=lambda u: not _matches(u, strong, weak, query_terms))
    h.axes["topology"] = Axis("topology", bool(topo), units=topo[:1],
                              hedged=any(_has_cue(u.text, HEDGE_CUES) for u in topo))

    # -- counter-evidence: sources that explicitly deny or hedge a link to the
    #    component, *about this symptom* - either they talk about our signature,
    #    or they explain why an axis came up empty. A denial about some other
    #    symptom of the same component is noise, not counter-evidence.
    axis_for = {"known_issues": "known_issue", "deployments": "deployment", "precedent": "precedent",
                "runbooks": "runbook", "topology": "topology"}
    for dtype, axis in axis_for.items():
        for u in by_type.get(dtype, []):
            if not (u.mentions(component) and _has_cue(u.text, NEGATION_CUES + HEDGE_CUES)):
                continue
            if not (h.axes[axis].hit is False or _matches(u, strong, weak, query_terms)):
                continue
            if not any(u is x for a in h.axes.values() for x in a.units if not a.hedged):
                h.counter.append(u)
    return h


def _correlate_evidence(query: str, units: List[Unit], ranked: List[Tuple[Unit, float]]) -> Hypothesis:
    query_terms = set(_tokenize(query))
    nominees = _nominate_components(units, ranked)
    if not nominees:
        return _evaluate("unknown", 0.0, units, query_terms)
    lead = nominees[0][1]
    # Relevance decides *which* components are about this query; corroboration
    # decides between the ones that are comparably relevant.
    contenders = [(c, s) for c, s in nominees if s >= RELEVANCE_GATE * lead]
    hyps = [_evaluate(c, s, units, query_terms) for c, s in contenders]
    best = max(hyps, key=lambda h: (h.raw_score(), h.relevance))
    best.nominees = nominees
    return best


# ---------------------------------------------------------------------------
# 4. Calibrate
# ---------------------------------------------------------------------------

def _calibrate_confidence(h: Hypothesis) -> float:
    raw = h.raw_score()
    penalty = min(HEDGE_PENALTY_CAP, HEDGE_PENALTY_PER_SOURCE * len(h.hedge_sources()))
    return float(max(FLOOR, min(CAP, raw - penalty)))


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def _excerpt(u: Unit, limit: int = 260) -> Dict[str, str]:
    text = u.text if len(u.text) <= limit else u.text[:limit].rstrip() + "..."
    return {"source": u.source, "excerpt": text}


def _ref(u: Optional[Unit]) -> str:
    if not u:
        return ""
    m = REF_ID.search(u.section) or REF_ID.search(u.text)
    return m.group(0) if m else ""


def _mttr(units: List[Unit], h: Hypothesis) -> Tuple[Optional[int], str]:
    """Runbook MTTR first, precedent MTTR second. A figure the source itself
    hedges is reported as None, with the number surfaced in the note."""
    for name in ("runbook", "precedent"):
        a = h.axes[name]
        if not a.hit:
            continue
        for u in _section_units(units, a.units[0]):
            m = MTTR.search(u.text)
            if not m:
                continue
            minutes = int(m.group(1))
            if _has_cue(u.text, HEDGE_CUES):
                return None, (f"{_ref(a.units[0]) or name} quotes an MTTR of {minutes} minutes but "
                              f"flags that figure as unconfirmed for this symptom, so none is reported.")
            return minutes, ""
    return None, "No MTTR figure is documented for this signature."


def _impacted(units: List[Unit], h: Hypothesis) -> List[str]:
    out = [h.component]
    for u in h.coupled:                                     # failed at the same instant
        if u.component not in out:
            out.append(u.component)
    for u in units:                                         # documented callers
        if u.doc_type == "topology" and u.mentions(h.component) \
                and re.search(r"delegates to|calls|depends on", u.text, re.I):
            for c in COMPONENT.findall(u.text.lower()):
                if c != h.component and c not in out and any(x.component == c for x in units if x.kind == "log"):
                    out.append(c)
    return out


def _signature_phrase(h: Hypothesis) -> str:
    if h.strong_sig:
        return " / ".join(h.strong_sig)
    if h.symptom:
        return re.sub(r"\s+\S+=\S+", "", h.symptom[0].message).strip()
    return "no error or warning signal"


def _root_cause(units: List[Unit], h: Hypothesis, confidence: float) -> str:
    ax = h.axes
    sig = _signature_phrase(h)
    when = h.first_ts.strftime("%Y-%m-%d %H:%M") if h.first_ts else "unknown time"

    if confidence >= 50:
        parts = [f"{sig} in {h.component}: {len(h.symptom)} {h.symptom[0].level} entries from {when}"]
        if ax["deployment"].hit:
            d = ax["deployment"].units[0]
            ver = VERSION.search(d.text)
            change = d.fields.get("change") or d.text.split(" | ")[-1]
            parts.append(f"beginning {h.deploy_lag} minutes after deployment "
                         f"{ver.group(0) if ver else ''} to {h.component} ({change})")
        tail = []
        if ax["known_issue"].hit:
            k = ax["known_issue"].units[0]
            tail.append(f"known issue {_ref(k)} ({k.fields.get('title', '').strip()})")
        if ax["precedent"].hit:
            p = ax["precedent"].units[0]
            rc = _labeled(_section_units(units, p), "root cause")
            tail.append(f"prior incident {_ref(p)}"
                        + (f", whose root cause was: {_strip_label(rc.text).rstrip('.')}" if rc else ""))
        text = ", ".join(parts) + "."
        if tail:
            text += " Matches " + " and ".join(tail) + "."
        return text

    missing = []
    if not h.symptom or h.symptom[0].level not in ERROR_LEVELS:
        missing.append("no ERROR-level log entries")
    if not ax["known_issue"].hit:
        missing.append("no known issue with this signature")
    if not ax["deployment"].hit:
        missing.append(f"no deployment to {h.component} before the symptom")
    if not ax["precedent"].hit:
        missing.append("no prior incident with this signature")
    if not ax["runbook"].hit:
        missing.append("no runbook covering it")
    elif ax["runbook"].hedged:
        missing.append(f"the only runbook ({_ref(ax['runbook'].units[0])}) describes itself as unverified")
    lead = (f"'{h.symptom[0].message}' logged by {h.component} at {when}" if h.symptom
            else f"{h.component} is the component the query points at, but nothing in the logs implicates it")
    return (f"UNCONFIRMED. Leading hypothesis: a backlog or slowdown in {h.component} - the only signal is "
            f"{lead}. The evidence is too thin to name a root cause: {'; '.join(missing)}. "
            f"Needs human investigation.")


def _remediation(units: List[Unit], h: Hypothesis, confidence: float, mttr_note: str) -> str:
    ax = h.axes
    steps = []
    if ax["deployment"].hit:
        d = ax["deployment"].units[0]
        ver = VERSION.search(d.text)
        change = d.fields.get("change") or d.text.split(" | ")[-1]
        steps.append(f"Roll back deployment {ver.group(0) if ver else ''} on {h.component} ({change}).")
    if ax["runbook"].hit:
        sec = _section_units(units, ax["runbook"].units[0])
        rem = _labeled(sec, "remediation")
        if rem:
            steps.append(f"Per runbook {_ref(ax['runbook'].units[0])}: {_strip_label(rem.text)}")
    if ax["precedent"].hit:
        res = _labeled(_section_units(units, ax["precedent"].units[0]), "resolution")
        if res:
            steps.append(f"Precedent {_ref(ax['precedent'].units[0])} was resolved by: {_strip_label(res.text)}")
    if confidence < 50:
        diag = _labeled(_section_units(units, ax["runbook"].units[0]), "diagnostic") if ax["runbook"].hit else None
        prefix = ("Do not act on a single-signal hypothesis. Escalate to a human and gather evidence first"
                  + (f" - {_strip_label(diag.text).rstrip('.')}" if diag else "") + ". ")
        gaps = [u for u in h.counter if _has_cue(u.text, HEDGE_CUES)]
        if gaps:
            prefix += ("The sources themselves flag instrumentation gaps (" +
                       "; ".join(sorted({u.source for u in gaps})) + ") that must be closed before a "
                       "confident diagnosis is possible. ")
        steps = [prefix + ("Only then consider: " + " ".join(steps) if steps else "")]
    if mttr_note:
        steps.append(mttr_note)
    return " ".join(s for s in steps if s).strip()


def _evidence(h: Hypothesis) -> List[Dict[str, str]]:
    ordered: List[Unit] = []
    ordered += h.symptom[:2] + h.coupled[:1]
    for name in ("deployment", "known_issue", "precedent", "runbook", "topology"):
        ordered += h.axes[name].units
    ordered += h.counter
    seen, out = set(), []
    for u in ordered:
        if u.uid not in seen:
            seen.add(u.uid)
            out.append(_excerpt(u))
    return out[:10]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def investigate_verbose(query: str, corpus: dict) -> Tuple[dict, Hypothesis]:
    units = _ingest_corpus(corpus)
    ranked = _retrieve_relevant_documents(query, units)
    h = _correlate_evidence(query, units, ranked)
    confidence = _calibrate_confidence(h)
    mttr, mttr_note = _mttr(units, h)
    report = {
        "root_cause": _root_cause(units, h, confidence),
        "supporting_evidence": _evidence(h),
        "impacted_systems": _impacted(units, h),
        "mttr_minutes": mttr,
        "remediation": _remediation(units, h, confidence, mttr_note),
        "confidence_score": confidence,
        "needs_human_review": confidence < 50,
    }
    return report, h


def investigate(query: str, corpus: dict) -> dict:
    """corpus: filename -> full document text (all the files in one
    incident's data/ folder)."""
    return investigate_verbose(query, corpus)[0]
