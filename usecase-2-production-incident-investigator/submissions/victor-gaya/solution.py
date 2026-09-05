"""Production incident investigator — retrieval + cross-source evidence
correlation with calibrated confidence.

Use case 2, Julius Baer AI hackathon sidequest.

The pipeline has four stages, following the shape suggested by the
starter:

    ingest      corpus -> flat list of retrievable Units (log lines,
                markdown sections, CSV rows), each tagged with the kind
                of document it came from
    retrieve    rank units against the query with hand-rolled TF-IDF +
                cosine similarity
    correlate   build candidate hypotheses from anomalous log lines, then
                ask every *other* kind of document whether it independently
                corroborates each one
    calibrate   turn the corroboration profile — not the top hit's
                relevance — into a 0-100 confidence score

The load-bearing idea is in `correlate`: the top-ranked document for a
query is usually the architecture overview, because that is where the
nouns in the query live. It is not evidence. Evidence is a deployment
record, a known-issue row, a runbook and a prior incident independently
pointing at the same component and the same failure signature. So
retrieval ranks, but corroboration decides — and the confidence score is
a function of how many *distinct kinds* of document agree, penalised
whenever the corroborating text hedges itself or the corpus explicitly
records the absence of a correlation.

Standard library only: no numpy, pandas or scikit-learn, so this runs
unchanged wherever Python 3.9+ does.
"""

from __future__ import annotations

import csv
import io
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Calibration constants                                                        #
# --------------------------------------------------------------------------- #
# A hypothesis anchored in the logs but corroborated by nothing else is a
# guess, and must land below the 50-point human-review line. Each distinct
# corroborating document kind is worth a fixed step; hedged corroboration and
# explicitly recorded absences pull back down. The weights are chosen so that
# one lone hedged signal cannot reach 50 no matter how well it matches the
# query, and three or more independent kinds cannot fall below it.
CONF_BASE = 25.0                 # a reproducible anomaly exists in the logs
CONF_PER_CORROBORATION = 15.0    # each distinct corroborating document kind
CONF_TEMPORAL_BONUS = 8.0        # a change demonstrably precedes the symptom
CONF_ALIGNMENT_WEIGHT = 10.0     # how well the hypothesis matches the question
CONF_HEDGE_PENALTY = 10.0        # corroborating source disclaims itself
CONF_ABSENCE_PENALTY = 5.0       # corpus explicitly records "no such correlation"
CONF_RELEVANCE_FLOOR = 0.4       # damping applied when the finding barely
CONF_RELEVANCE_GAIN = 2.5        # relates to the question that was asked
CONF_CEILING = 95.0              # never claim certainty from documents alone
CONF_FLOOR = 5.0
HUMAN_REVIEW_THRESHOLD = 50.0

# Document kinds that can corroborate a log-anchored hypothesis. The logs
# themselves are the anchor, not corroboration — otherwise a single noisy
# component would corroborate itself.
CORROBORATING_KINDS = ("known_issues", "deployment", "runbook", "precedent")

SEVERITY_WEIGHT = {"FATAL": 1.0, "CRITICAL": 1.0, "ERROR": 1.0, "WARN": 0.6,
                   "WARNING": 0.6}

# Phrases that mark a source as disclaiming its own applicability.
HEDGE_MARKERS = (
    "may not apply", "unconfirmed", "unverified", "incomplete", "not currently",
    "no documented", "pending", "not instrumented", "outside this service",
    "worth noting", "whether this is actually",
)

# Phrases that mark an explicitly recorded *absence* of correlation.
ABSENCE_PATTERNS = (
    r"\bno (?:other )?deployment(?:s)? (?:touched|correlated|in)",
    r"\bthere is no deployment\b",
    r"\bno previous incident\b",
    r"\bno other runbook\b",
    r"\bfirst recorded report\b",
    r"\bno known issue\b",
    r"\bno matching\b",
)

# Vocabulary that appears in the *asks* of an incident query rather than in the
# symptom itself ("identify the probable root cause", "what is the mean time to
# recover"). It is identical across incidents, so it carries no discriminating
# signal and actively mis-ranks: "mean time to recover" makes every log line
# containing the word "time" look relevant.
QUERY_BOILERPLATE = frozenset("""
identify probable root cause supporting evidence impacted component components
recommended remediation mean time recover recovery systems system mttr what
reporting report customers users
""".split())

STOPWORDS = frozenset("""
a an and are as at be been being but by can could did do does for from had has
have how i if in into is it its may might must no not of on or our over should
so some such than that the their them then there these they this those to too
under up was we were what when where which while who why will with would you your
""".split())

# service-name shape: two or more lowercase hyphen-joined words
COMPONENT_RE = re.compile(r"\b[a-z][a-z0-9]*(?:-[a-z0-9]+)+\b")
# exception / error-type shape: CamelCase ending in a failure noun
EXCEPTION_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:Exception|Error|Timeout|Failure)\b")
LOG_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})\s+"
    r"(?P<sev>[A-Z]{3,8})\s+(?P<rest>.*)$"
)
MTTR_RE = re.compile(r"MTTR[^0-9\n]{0,40}?(\d{1,4})\s*minute", re.IGNORECASE)
VOLATILE_RE = re.compile(r"\b\w+_id=[\w-]+|\b\d+(?:\.\d+)?(?:ms|s|m)\b|\b\d{2,}\b")


# --------------------------------------------------------------------------- #
# Text utilities                                                               #
# --------------------------------------------------------------------------- #
def _tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens, stopwords removed, hyphenated service
    names kept whole *and* split so `notification-service` also matches a
    query that says "notification service"."""
    raw = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*|\d+", text.lower())
    tokens: List[str] = []
    for token in raw:
        token = token.strip("-_")
        if not token or token in STOPWORDS or len(token) < 2:
            continue
        tokens.append(token)
        if "-" in token:
            tokens.extend(p for p in token.split("-") if p and p not in STOPWORDS)
    return tokens


def _term_frequencies(tokens: Sequence[str]) -> Dict[str, float]:
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = float(len(tokens))
    return {term: count / total for term, count in counts.items()}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    shared = set(a) & set(b)
    if not shared:
        return 0.0
    dot = sum(a[t] * b[t] for t in shared)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return 0.0 if na == 0.0 or nb == 0.0 else dot / (na * nb)


def _covers(term: str, token: str, min_prefix: int = 5) -> bool:
    """Prefix-tolerant term match, standing in for a stemmer.

    `payments`/`payment`, `intermittently`/`intermittent` and `emails`/`email`
    should match; `late`/`latency` should not. Requiring a shared prefix of at
    least five characters draws that line without pulling in a stemming
    dependency.
    """
    if term == token:
        return True
    shorter, longer = (term, token) if len(term) <= len(token) else (token, term)
    return len(shorter) >= min_prefix and longer.startswith(shorter)


def _coverage(terms: Sequence[str], tokens: Iterable[str]) -> float:
    """Fraction of the symptom's distinctive terms present in a piece of text.

    Recall over the query rather than cosine similarity: a long, specific
    paragraph should not be penalised for its length the way cosine penalises
    it, and what matters here is how much of the reported symptom a document
    actually accounts for.
    """
    if not terms:
        return 0.0
    token_set = set(tokens)
    hits = sum(1 for term in terms if any(_covers(term, t) for t in token_set))
    return hits / float(len(terms))


def _symptom_terms(query: str) -> List[str]:
    """The distinctive terms of the reported symptom: first line of the query,
    boilerplate asks removed."""
    lines = [line.strip() for line in query.splitlines() if line.strip()]
    symptom = lines[0] if lines else query
    seen: List[str] = []
    for token in _tokenize(symptom):
        if token not in QUERY_BOILERPLATE and token not in seen:
            seen.append(token)
    return seen


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


def _normalize_signature(text: str) -> str:
    """Collapse ids, durations and counters so that repeated occurrences of the
    same failure collapse to one signature."""
    return VOLATILE_RE.sub("<v>", text).strip()


def _clean_excerpt(text: str, limit: int = 460) -> str:
    """Excerpts must stay verbatim substrings of their source document, so this
    only strips surrounding whitespace and truncates on a word boundary."""
    excerpt = text.strip()
    if len(excerpt) <= limit:
        return excerpt
    # Prefer a sentence boundary: an instruction cut mid-clause ("Compare the
    # current pool") reads as a defect in the citation, not a truncation.
    sentence_end = max(excerpt.rfind(". ", 0, limit), excerpt.rfind(".\n", 0, limit))
    if sentence_end > limit // 2:
        return excerpt[: sentence_end + 1].rstrip()
    cut = excerpt.rfind(" ", 0, limit)
    return excerpt[: cut if cut > 0 else limit].rstrip()


def _readable(text: str) -> str:
    """Render a markdown fragment as a plain sentence fragment: strip table
    pipes and emphasis so a cited change reads as prose inside a narrative."""
    stripped = text.strip()
    if stripped.startswith("|"):
        cells = [c.strip().strip("*").strip() for c in stripped.strip("|").split("|")]
        return " · ".join(c for c in cells if c)
    return " ".join(stripped.replace("**", "").split())


def _is_table_skeleton(text: str) -> bool:
    """True for a markdown table header/separator carrying no data of its own."""
    return "---|" in text or "--- |" in text or re.search(r"\|[\s\-:|]{6,}\|", text) is not None


def _has_hedge(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in HEDGE_MARKERS)


def _absence_statements(text: str) -> List[str]:
    lowered = text.lower()
    return [p for p in ABSENCE_PATTERNS if re.search(p, lowered)]


# --------------------------------------------------------------------------- #
# Stage 1 — ingest                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class Unit:
    """One retrievable fragment of the corpus."""

    source: str
    unit_id: str
    text: str
    kind: str                      # logs | deployment | known_issues | runbook |
                                   # precedent | architecture | api_spec | other
    tokens: List[str] = field(default_factory=list)
    tf: Dict[str, float] = field(default_factory=dict)
    components: List[str] = field(default_factory=list)
    severity: Optional[str] = None
    timestamp: Optional[str] = None
    signature: Optional[str] = None
    emitter: Optional[str] = None          # log lines: the component that logged
    meta: Dict[str, str] = field(default_factory=dict)   # CSV rows: column -> value
    section_text: str = ""                 # enclosing section, for hedge context

    @property
    def is_anomaly(self) -> bool:
        return bool(self.severity and self.severity in SEVERITY_WEIGHT)


def _classify_kind(filename: str, text: str) -> str:
    """Infer document kind from the filename, falling back to content so the
    pipeline does not depend on any particular naming convention."""
    name = filename.lower()
    table = (
        ("logs", ("log",)),
        ("deployment", ("deploy", "release", "change")),
        ("known_issues", ("known_issue", "known-issues", "issues", "catalog")),
        ("runbook", ("runbook", "playbook", "sop")),
        ("precedent", ("previous_incident", "past_incident", "postmortem", "history_incident")),
        ("architecture", ("architecture", "design", "topology")),
        ("api_spec", ("api", "spec", "contract")),
    )
    for kind, needles in table:
        if any(n in name for n in needles):
            return kind
    lowered = text[:2000].lower()
    if LOG_LINE_RE.search(text[:4000] or ""):
        return "logs"
    if "issue_id" in lowered and "signature" in lowered:
        return "known_issues"
    if "runbook" in lowered or "remediation" in lowered:
        return "runbook"
    if "mttr" in lowered and "root cause" in lowered:
        return "precedent"
    if "deployment" in lowered and "version" in lowered:
        return "deployment"
    return "other"


def _split_markdown_sections(text: str) -> List[Tuple[str, str]]:
    """Split on markdown headings, then on table rows and blank-line blocks, so
    a single deployment row or runbook entry can be cited on its own."""
    sections: List[Tuple[str, str]] = []
    current_title = "preamble"
    buffer: List[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((current_title, body))

    for line in text.splitlines():
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            flush()
            buffer = []
            current_title = heading.group(2).strip()
            continue
        buffer.append(line)
    flush()
    return sections


def _split_blocks(body: str) -> List[str]:
    """Split a section body into blocks: blank-line paragraphs, and top-level
    bullets (which are frequently not blank-line separated).

    Granularity matters for specificity: an architecture section that describes
    four components in four bullets should yield four units, so that the bullet
    describing the component in question can outrank the diagram that names them
    all. Every returned block is a contiguous substring of `body`, which keeps
    cited excerpts verbatim.
    """
    blocks: List[str] = []
    buffer: List[str] = []

    def flush() -> None:
        block = "\n".join(buffer).strip()
        if block:
            blocks.append(block)

    for line in body.splitlines():
        starts_bullet = re.match(r"^[-*]\s+\S", line)
        if (not line.strip() or starts_bullet) and buffer:
            flush()
            buffer = []
        if line.strip():
            buffer.append(line)
    flush()
    return blocks


def _ingest_corpus(corpus: Dict[str, str]) -> List[Unit]:
    """Turn `filename -> text` into a flat list of retrievable units.

    Granularity is chosen per document kind: log files split per line (a single
    line is the natural unit of evidence), CSV catalogs split per row, prose
    splits per heading and then per table row.
    """
    units: List[Unit] = []

    for source, text in sorted(corpus.items()):
        kind = _classify_kind(source, text)

        if kind == "known_issues" or source.lower().endswith(".csv"):
            units.extend(_ingest_csv(source, text, kind))
            continue

        if kind == "logs":
            units.extend(_ingest_logs(source, text))
            # Keep the prose commentary around a log file as its own unit.
            # Taking the segments *outside* the code fences keeps each one a
            # contiguous substring of the file, so it stays quotable verbatim.
            for seg_index, segment in enumerate(text.split("```")[::2]):
                if len(segment.strip()) > 40:
                    units.append(
                        _make_unit(source, f"{source}#notes{seg_index}",
                                   segment.strip(), "logs")
                    )
            continue

        for index, (title, body) in enumerate(_split_markdown_sections(text)):
            section = _make_unit(source, f"{source}#{title or index}", body, kind)
            section.section_text = body
            units.append(section)
            blocks = _split_blocks(body)
            if len(blocks) > 1:
                for b_index, block in enumerate(blocks):
                    unit = _make_unit(source, f"{source}#{title or index}/b{b_index}",
                                      block, kind)
                    # A block inherits its section's context: a runbook that
                    # disclaims itself two paragraphs below the symptom list is
                    # still a disclaimed runbook.
                    unit.section_text = body
                    units.append(unit)
            # table rows are individually citable evidence
            for row in body.splitlines():
                stripped = row.strip()
                if stripped.startswith("|") and stripped.count("|") >= 3 \
                        and not re.match(r"^\|[\s\-:|]+\|$", stripped):
                    row_unit = _make_unit(source, f"{source}#{title}:{stripped[:40]}",
                                          stripped, kind)
                    row_unit.section_text = body
                    units.append(row_unit)
    return units


def _make_unit(source: str, unit_id: str, text: str, kind: str) -> Unit:
    tokens = _tokenize(text)
    return Unit(
        source=source,
        unit_id=unit_id,
        text=text,
        kind=kind,
        tokens=tokens,
        tf=_term_frequencies(tokens),
        components=sorted(set(COMPONENT_RE.findall(text.lower()))),
    )


def _ingest_csv(source: str, text: str, kind: str) -> List[Unit]:
    """One unit per row, so an irrelevant catalog row cannot drag a relevant one
    into the ranking (or vice versa)."""
    units: List[Unit] = []
    try:
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            values = [str(v) for v in row.values() if v]
            if not values:
                continue
            row_id = str(next(iter(row.values())) or f"row{len(units)}")
            raw_line = ",".join(values)
            unit = _make_unit(source, f"{source}#{row_id}", raw_line, kind or "known_issues")
            unit.meta = {str(k): str(v) for k, v in row.items() if k and v}
            units.append(unit)
    except Exception:
        # Malformed CSV: fall back to line-level units rather than losing the file.
        for index, line in enumerate(text.splitlines()):
            if line.strip():
                units.append(_make_unit(source, f"{source}#L{index}", line.strip(), kind))
    return units


def _ingest_logs(source: str, text: str) -> List[Unit]:
    units: List[Unit] = []
    for index, raw in enumerate(text.splitlines()):
        line = raw.strip()
        if not line or line.startswith(("#", "```")):
            continue
        match = LOG_LINE_RE.match(line)
        if not match:
            continue
        rest = match.group("rest")
        unit = _make_unit(source, f"{source}#L{index}", line, "logs")
        emitter = rest.split(None, 1)[0] if rest.split() else ""
        unit.emitter = emitter if COMPONENT_RE.fullmatch(emitter.lower()) else None
        unit.severity = match.group("sev").upper()
        unit.timestamp = match.group("ts")
        unit.signature = _normalize_signature(rest)
        unit.meta["message"] = (
            rest[len(emitter):].strip() if emitter and rest.startswith(emitter) else rest
        )
        units.append(unit)
    return units


# --------------------------------------------------------------------------- #
# Stage 2 — retrieve                                                           #
# --------------------------------------------------------------------------- #
class _Index:
    """Minimal TF-IDF index over units, with cosine similarity."""

    def __init__(self, units: Sequence[Unit]) -> None:
        self.units = list(units)
        n = max(1, len(self.units))
        doc_freq: Counter = Counter()
        for unit in self.units:
            doc_freq.update(set(unit.tokens))
        self.idf = {
            term: math.log((1.0 + n) / (1.0 + df)) + 1.0
            for term, df in doc_freq.items()
        }
        self.vectors = [self._vector(unit.tf) for unit in self.units]

    def _vector(self, tf: Dict[str, float]) -> Dict[str, float]:
        return {t: w * self.idf.get(t, 1.0) for t, w in tf.items()}

    def query_vector(self, text: str) -> Dict[str, float]:
        return self._vector(_term_frequencies(_tokenize(text)))

    def focused_query_vector(self, query: str) -> Dict[str, float]:
        """Query vector with the standard asks stripped and the symptom
        sentence weighted above the rest.

        An incident query is a symptom statement followed by boilerplate
        ("identify the root cause", "what is the MTTR"). The boilerplate is the
        same for every incident, so it dilutes the vector and drags in any
        document that happens to share a common word with it.
        """
        lines = [line.strip() for line in query.splitlines() if line.strip()]
        symptom = lines[0] if lines else query
        remainder = " ".join(lines[1:])
        weighted: Dict[str, float] = defaultdict(float)
        for token in _tokenize(symptom):
            if token not in QUERY_BOILERPLATE:
                weighted[token] += 2.0
        for token in _tokenize(remainder):
            if token not in QUERY_BOILERPLATE:
                weighted[token] += 0.5
        total = sum(weighted.values()) or 1.0
        return self._vector({t: w / total for t, w in weighted.items()})

    def similarity(self, query_vec: Dict[str, float], index: int) -> float:
        return _cosine(query_vec, self.vectors[index])

    def rank_units(self, text: str) -> List[Tuple[Unit, float]]:
        qv = self.query_vector(text)
        scored = [(unit, self.similarity(qv, i)) for i, unit in enumerate(self.units)]
        return sorted(scored, key=lambda pair: -pair[1])


def _retrieve_relevant_documents(
    query: str, corpus: Dict[str, str]
) -> List[Tuple[str, float]]:
    """Rank corpus entries against `query`, most relevant first.

    Returns finer-grained ids where the document is row-oriented (e.g.
    `known_issues.csv#KI-101`) and plain filenames elsewhere, aggregating each
    source to its best-scoring unit.
    """
    units = _ingest_corpus(corpus)
    if not units:
        return []
    index = _Index(units)
    best: Dict[str, Tuple[str, float]] = {}
    for unit, score in index.rank_units(query):
        label = unit.unit_id if unit.kind == "known_issues" else unit.source
        if label not in best or score > best[label][1]:
            best[label] = (label, score)
    return sorted(best.values(), key=lambda pair: -pair[1])


# --------------------------------------------------------------------------- #
# Stage 3 — correlate                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class Corroboration:
    kind: str
    unit: Unit
    strength: float
    hedged: bool
    reason: str


@dataclass
class Hypothesis:
    component: str
    signature: str
    anchor_units: List[Unit]
    occurrences: int
    severity_weight: float
    alignment: float                       # 0-1, how well it answers the query
    corroborations: List[Corroboration] = field(default_factory=list)
    absences: List[Tuple[str, Unit]] = field(default_factory=list)
    temporal_support: Optional[Unit] = None

    @property
    def corroborating_kinds(self) -> List[str]:
        return sorted({c.kind for c in self.corroborations})

    @property
    def hedged_count(self) -> int:
        return sum(1 for c in self.corroborations if c.hedged)

    @property
    def strength(self) -> float:
        """Ranking score: corroboration dominates, relevance breaks ties."""
        return (
            3.0 * len(self.corroborating_kinds)
            + 4.0 * self.alignment
            + 0.5 * self.severity_weight
            + 0.25 * math.log1p(self.occurrences)
            + (0.5 if self.temporal_support else 0.0)
        )


DESCRIPTIVE_KINDS = ("architecture", "api_spec", "known_issues", "runbook",
                     "precedent", "deployment")


def _attested_components(units: Sequence[Unit]) -> set:
    """The set of names that are actually components of this system.

    Hyphenated-lowercase word shape alone is not enough: `order_id=ORD-88350`
    lowercases to `ord-88350`, and prose is full of `third-party` and
    `per-stage`. Two structural signals are used instead, both of which
    generalise to any log corpus:

      * the component field of a log line — whatever emitted the entry;
      * a column named like "component" in a CSV catalog.

    If neither is available the function degrades to word shape within
    descriptive documents rather than returning nothing.
    """
    attested: set = set()
    for unit in units:
        if unit.emitter:
            attested.add(unit.emitter.lower())
        for column, value in unit.meta.items():
            if "component" in column.lower():
                for name in COMPONENT_RE.findall(str(value).lower()):
                    attested.add(name)
    if attested:
        return attested
    for unit in units:
        if unit.kind in DESCRIPTIVE_KINDS:
            attested.update(unit.components)
    return attested


def _component_relevance(component: str, units: Sequence[Unit],
                         index: _Index, query: str,
                         attested: Optional[set] = None) -> float:
    """How relevant is this component to the question?

    Measured as the best query similarity of any unit that describes the
    component — so a component whose documented purpose matches the symptom
    ("sends order confirmation emails") outranks one that merely appears in a
    noisy log line.
    """
    qv = index.focused_query_vector(query)
    best = 0.0
    for i, unit in enumerate(index.units):
        describes = unit.kind in (
            "architecture", "api_spec", "runbook", "precedent", "known_issues"
        )
        if not (describes and component in unit.components):
            continue
        # Weight by specificity: a unit naming one component says something
        # about that component; an architecture diagram naming five says
        # something about the system. Without this every component in the
        # diagram inherits an identical relevance score.
        named = [c for c in unit.components if c in attested] if attested else unit.components
        specificity = 1.0 / float(max(1, len(named)))
        best = max(best, index.similarity(qv, i) * specificity)
    return best


def _component_coverage(component: str, index: _Index, query: str,
                        attested: Optional[set] = None) -> float:
    """Best symptom coverage achieved by any document that describes this
    component — how much of the reported symptom its documented purpose or
    known behaviour actually accounts for."""
    terms = _symptom_terms(query)
    best = 0.0
    for unit in index.units:
        describes = unit.kind in DESCRIPTIVE_KINDS
        if not (describes and component in unit.components):
            continue
        named = [c for c in unit.components if c in attested] if attested else unit.components
        specificity = 1.0 / float(max(1, len(named)))
        best = max(best, _coverage(terms, unit.tokens) * specificity)
    return best


def _build_hypotheses(query: str, units: Sequence[Unit], index: _Index) -> List[Hypothesis]:
    """Candidate root causes come from anomalous log lines, grouped by
    (component, normalised signature)."""
    attested = _attested_components(units)
    groups: Dict[Tuple[str, str], List[Unit]] = defaultdict(list)
    for unit in units:
        if unit.kind != "logs" or not unit.is_anomaly or not unit.signature:
            continue
        named = [c for c in unit.components if c in attested] or unit.components
        component = named[0] if named else "unknown-component"
        groups[(component, unit.signature)].append(unit)

    qv = index.focused_query_vector(query)
    query_tokens = set(_tokenize(query))
    hypotheses: List[Hypothesis] = []

    for (component, signature), anchors in groups.items():
        sig_tf = _term_frequencies(_tokenize(signature))
        sig_vec = {t: w * index.idf.get(t, 1.0) for t, w in sig_tf.items()}
        signature_similarity = _cosine(qv, sig_vec)
        component_similarity = _component_relevance(
            component, units, index, query, attested
        )
        # Direct lexical bridge: does the query name the component itself?
        names_component = _jaccard(query_tokens, set(component.split("-")))
        component_coverage = _component_coverage(component, index, query, attested)
        alignment = max(
            signature_similarity,
            0.85 * component_similarity,
            names_component,
            component_coverage,
            _coverage(_symptom_terms(query), _tokenize(signature)),
        )
        severity = max(SEVERITY_WEIGHT.get(u.severity or "", 0.0) for u in anchors)
        hypotheses.append(
            Hypothesis(
                component=component,
                signature=signature,
                anchor_units=sorted(anchors, key=lambda u: u.timestamp or ""),
                occurrences=len(anchors),
                severity_weight=severity,
                alignment=min(1.0, alignment),
            )
        )
    return hypotheses


def _corroborate(hypothesis: Hypothesis, units: Sequence[Unit]) -> None:
    """Ask every non-log document kind whether it independently supports the
    hypothesis. A match needs both the component and real signature overlap —
    naming the component alone is not corroboration, or every catalog row about
    a service would corroborate every failure of that service.
    """
    signature_tokens = {
        t for t in _tokenize(hypothesis.signature) if t not in hypothesis.component.split("-")
    }
    component_parts = set(hypothesis.component.split("-"))

    for unit in units:
        if unit.kind not in CORROBORATING_KINDS:
            continue
        mentions_component = (
            hypothesis.component in unit.components
            or _jaccard(component_parts, set(unit.tokens)) > 0.0
            and component_parts.issubset(set(unit.tokens))
        )
        overlap = _jaccard(signature_tokens, set(unit.tokens))
        shared = len(signature_tokens & set(unit.tokens))
        if not mentions_component or (overlap < 0.06 and shared < 2):
            continue
        hedged = _has_hedge(unit.section_text or unit.text)
        hypothesis.corroborations.append(
            Corroboration(
                kind=unit.kind,
                unit=unit,
                strength=overlap + 0.1 * shared,
                hedged=hedged,
                reason=(
                    f"{unit.kind} evidence naming {hypothesis.component} with "
                    f"{shared} shared signature term(s)"
                ),
            )
        )

    # Keep the single strongest corroboration per document kind: five runbook
    # sections about one component are one independent opinion, not five.
    best_by_kind: Dict[str, Corroboration] = {}
    for corroboration in hypothesis.corroborations:
        current = best_by_kind.get(corroboration.kind)
        if current is None or corroboration.strength > current.strength:
            best_by_kind[corroboration.kind] = corroboration
    hypothesis.corroborations = sorted(
        best_by_kind.values(), key=lambda c: -c.strength
    )

    # Temporal check: did a recorded change to this component precede the first
    # anomaly? That is what turns a deployment mention into a correlation.
    first_anomaly = hypothesis.anchor_units[0].timestamp if hypothesis.anchor_units else None
    for corroboration in hypothesis.corroborations:
        if corroboration.kind != "deployment":
            continue
        stamps = re.findall(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}", corroboration.unit.text)
        if first_anomaly and stamps and min(stamps) <= first_anomaly[:16]:
            hypothesis.temporal_support = corroboration.unit

    # Explicitly recorded absences of correlation. These only count against a
    # hypothesis where they explain a *missing* corroboration: a deployment log
    # stating "no deployment touched this component" matters precisely when
    # deployment evidence is absent. Where that kind already corroborates, the
    # same sentence is just bookkeeping ("no OTHER deployment touched it") and
    # must not be charged as a penalty.
    corroborated_kinds = set(hypothesis.corroborating_kinds)
    for unit in units:
        if unit.kind == "logs" or unit.kind in corroborated_kinds:
            continue
        if unit.kind not in CORROBORATING_KINDS:
            continue
        for pattern in _absence_statements(unit.text):
            hypothesis.absences.append((pattern, unit))
            break


def _correlate_evidence(
    query: str, corpus: Dict[str, str], ranked: List[Tuple[str, float]]
) -> Dict[str, object]:
    """Build hypotheses, corroborate each across independent document kinds, and
    return the winner together with everything needed to explain it."""
    units = _ingest_corpus(corpus)
    index = _Index(units)
    hypotheses = _build_hypotheses(query, units, index)
    for hypothesis in hypotheses:
        _corroborate(hypothesis, units)
    hypotheses.sort(key=lambda h: -h.strength)

    leading = hypotheses[0] if hypotheses else None
    return {
        "units": units,
        "index": index,
        "ranked": ranked,
        "hypotheses": hypotheses,
        "leading": leading,
        "query": query,
    }


# --------------------------------------------------------------------------- #
# Stage 4 — calibrate                                                          #
# --------------------------------------------------------------------------- #
def _score_hypothesis(leading: Optional[Hypothesis]) -> float:
    """Calibrated 0-100 confidence for one hypothesis.

    Exposed per hypothesis rather than only for the winner, so that competing
    explanations can be ranked on the same scale and presented honestly when no
    single one is well enough supported to stand alone.
    """
    if leading is None:
        return CONF_FLOOR

    score = CONF_BASE
    score += CONF_PER_CORROBORATION * len(leading.corroborating_kinds)
    score += CONF_TEMPORAL_BONUS if leading.temporal_support else 0.0
    score += CONF_ALIGNMENT_WEIGHT * leading.alignment
    score -= CONF_HEDGE_PENALTY * leading.hedged_count
    score -= CONF_ABSENCE_PENALTY * min(len(leading.absences), 2)

    # Corroboration says the finding is real; it does not say the finding
    # answers the question that was asked. A heavily corroborated incident in
    # the corpus should not be reported confidently as the cause of a symptom it
    # has no lexical or documented connection to. This damps — rather than
    # thresholds — the score by relevance, so it degrades smoothly instead of
    # flipping at a cutoff fitted to any particular incident.
    relevance = min(1.0, CONF_RELEVANCE_FLOOR + CONF_RELEVANCE_GAIN * leading.alignment)
    score *= relevance
    return round(max(CONF_FLOOR, min(CONF_CEILING, score)), 1)


def _calibrate_confidence(evidence: Dict[str, object]) -> float:
    """Confidence is a function of independent corroboration, not of how
    relevant the best-matching document felt."""
    leading: Optional[Hypothesis] = evidence.get("leading")  # type: ignore[assignment]
    return _score_hypothesis(leading)


# --------------------------------------------------------------------------- #
# Report assembly                                                              #
# --------------------------------------------------------------------------- #
def _impacted_systems(leading: Hypothesis, units: Sequence[Unit]) -> List[str]:
    """Components named in the corroborated evidence, plus components the
    architecture places in the direct path of the failing operation."""
    attested = _attested_components(units)
    systems: List[str] = [leading.component]

    def offer(component: str) -> None:
        if component in attested and component not in systems:
            systems.append(component)

    for anchor in leading.anchor_units:
        for component in anchor.components:
            offer(component)
    # Co-failing components: those logging an anomaly at the same instants as
    # the anchor — in a synchronous call path they fail together.
    stamps = {u.timestamp for u in leading.anchor_units if u.timestamp}
    for unit in units:
        if unit.kind == "logs" and unit.is_anomaly and unit.timestamp in stamps:
            for component in unit.components:
                offer(component)
    return systems


def _extract_mttr(leading: Hypothesis) -> Tuple[Optional[int], str]:
    """Adopt an MTTR only from a corroborating source that does not disclaim
    itself. Runbook figures are preferred (they are the operational planning
    number); a prior incident's actual MTTR is the fallback.

    Returns (value, provenance note).
    """
    candidates: List[Tuple[int, str, bool]] = []
    for corroboration in leading.corroborations:
        scope = corroboration.unit.section_text or corroboration.unit.text
        for match in MTTR_RE.finditer(scope):
            candidates.append(
                (int(match.group(1)), corroboration.kind, corroboration.hedged)
            )
    usable = [c for c in candidates if not c[2]]
    if not usable:
        if candidates:
            figures = ", ".join(f"{value} minutes" for value, _, _ in candidates)
            return None, (
                f"No time to recover is reported. The corpus does contain an MTTR "
                f"figure ({figures}), but its own source disclaims its applicability "
                f"to this incident, so adopting it would misrepresent a caveated "
                f"number as a measured one."
            )
        return None, "No MTTR figure is recorded for this failure mode."

    def note(kind: str, value: int) -> str:
        if kind == "runbook":
            return (
                f"Expected time to recover is {value} minutes — the runbook's typical "
                f"MTTR for this failure mode, not a measured recovery: the corpus "
                f"records no resolution timestamp for this occurrence."
            )
        if kind == "precedent":
            return (
                f"Expected time to recover is {value} minutes, the MTTR actually "
                f"observed the last time this failure mode occurred; the corpus records "
                f"no resolution timestamp for this occurrence."
            )
        return f"Expected time to recover is {value} minutes, per the corroborating {kind}."

    for preferred in ("runbook", "precedent"):
        for value, kind, _ in usable:
            if kind == preferred:
                return value, note(preferred, value)
    value, kind, _ = usable[0]
    return value, note(kind, value)


def _extract_remediation(leading: Hypothesis, confident: bool) -> str:
    """Prefer a remediation line from a corroborating runbook; otherwise state
    what would have to be established first."""
    for corroboration in leading.corroborations:
        if corroboration.kind != "runbook":
            continue
        match = re.search(
            r"\*{0,2}Remediation\*{0,2}\s*:?\s*(.+?)(?:\n\s*\n|\Z)",
            corroboration.unit.section_text or corroboration.unit.text,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            remediation = " ".join(match.group(1).split())
            if confident:
                return remediation
            return (
                f"Provisional, pending confirmation of the root cause: {remediation} "
                "Before acting, instrument the suspected stage so the hypothesis can "
                "be confirmed or ruled out."
            )
    if confident:
        return (
            f"Revert or reconfigure the most recent change to {leading.component}, "
            "then confirm recovery against the failure signature in the logs."
        )
    return (
        f"Do not remediate blind. Instrument {leading.component} to expose per-stage "
        "timings and confirm or eliminate the leading hypothesis before making any "
        "change."
    )


def _sole_change_attested(leading: Hypothesis, units: Sequence[Unit]) -> bool:
    """Does the corpus actually state that no *other* change touched this
    component in the window?

    Without this check the narrative would assert exclusivity the documents may
    never have claimed — exactly the plausible-sounding overclaim this pipeline
    is meant to avoid.
    """
    for unit in units:
        if unit.kind != "deployment":
            continue
        if leading.component not in unit.components:
            continue
        if _absence_statements(unit.text):
            return True
    return False


def _describe_root_cause(leading: Hypothesis, confident: bool,
                         mttr_note: str, sole_change: bool = False,
                         symptom: Optional[Unit] = None,
                         scenarios: str = "", uninstrumented: str = "") -> str:
    anchor = leading.anchor_units[0] if leading.anchor_units else None
    observed = (anchor.meta.get("message") if anchor else "") or " ".join(
        leading.signature.replace("<v>", "").split()
    )
    kinds = leading.corroborating_kinds
    window = ""
    if leading.anchor_units and leading.anchor_units[0].timestamp:
        first = leading.anchor_units[0].timestamp
        last = leading.anchor_units[-1].timestamp
        window = f" first seen at {first}" + (
            f" and recurring through {last}" if last and last != first else ""
        )

    if confident:
        parts: List[str] = []
        if leading.temporal_support:
            parts.append(
                f"A recorded change to {leading.component} — "
                f"{_readable(leading.temporal_support.text)} — precedes the onset of "
                f"the failure, which is{window}."
            )
            exclusivity = (
                "The deployment record states that no other change touched the "
                "component in the preceding window, which makes this the probable "
                "cause rather than a coincidence."
                if sole_change
                else "It is the closest recorded change preceding the failure."
            )
            parts.append(
                f"{leading.component} then fails with: {observed}. {exclusivity}"
            )
        else:
            parts.append(
                f"{leading.component} is failing with: {observed}{window}."
            )
        parts.append(
            f"The signature is corroborated independently by {len(kinds)} other kinds "
            f"of document ({', '.join(kinds)}), which is what raises this above a "
            f"single-source guess."
        )
        return " ".join(parts)
    missing = [k for k in CORROBORATING_KINDS if k not in kinds]
    hedged = "which disclaims its own applicability" if leading.hedged_count else ""
    supported = (
        f"corroborated only by the {', '.join(kinds)}"
        + (f" — {hedged}" if hedged else "")
        if kinds
        else "uncorroborated by any other document"
    )
    # The symptom being well documented and the cause being unestablished are
    # separate facts. Collapsing them would misdescribe the evidence in both
    # directions: it understates what is known and overstates what is guessed.
    symptom_clause = ""
    if symptom is not None:
        quoted = _readable(symptom.text)
        # cut at the end of the first sentence rather than mid-word
        stop = quoted.find(". ")
        quoted = quoted[: stop + 1] if 0 < stop < 260 else _clean_excerpt(quoted, 240)
        symptom_clause = (
            f"The symptom itself is well documented — {quoted} — so the failure is "
            f"real; what is missing is any evidence of its cause. "
        )
    parts = [
        "Undetermined — the cause cannot be established from the available evidence.",
        symptom_clause.strip(),
        f"The best-supported hypothesis points at {leading.component} ({observed}), "
        f"{supported}; the corpus records no corroborating "
        f"{', '.join(missing) if missing else 'evidence'} for it.",
    ]
    if scenarios:
        parts.append(scenarios)
    if uninstrumented:
        parts.append(
            f"The documents also state why these cannot be separated from the "
            f"record alone: {uninstrumented}"
        )
    parts.append(
        "This needs human review and targeted instrumentation rather than a "
        "remediation."
    )
    return " ".join(part for part in parts if part)


def _mechanism_evidence_all(leading: Hypothesis,
                            units: Sequence[Unit]) -> List[Unit]:
    """One design passage per document kind that explains the mechanism.

    The architecture answers "why does this component failing produce this
    symptom"; the API spec answers "why this timeout, why this error code".
    They are different explanations and both are worth citing.
    """
    # Exclude the component's own name from the overlap: otherwise any passage
    # that merely mentions the component scores as if it explained the failure,
    # and "Owned by: payment-service, delegates to payment-gateway-adapter"
    # outranks the paragraph that actually defines the timeout.
    signature_tokens = set(_tokenize(leading.signature)) - set(
        _tokenize(leading.component)
    )
    best: Dict[str, Tuple[int, Unit]] = {}
    for unit in units:
        if unit.kind not in ("architecture", "api_spec"):
            continue
        # A spec paragraph often says "the adapter" where its section heading
        # named the component in full, so the enclosing section counts too.
        in_scope = (
            leading.component in unit.components
            or leading.component in (unit.section_text or "").lower()
        )
        if not in_scope or _is_table_skeleton(unit.text):
            continue
        overlap = len(signature_tokens & set(unit.tokens))
        if overlap < 1:
            continue
        current = best.get(unit.kind)
        # prefer more overlap, then the more specific (shorter) passage
        if current is None or (overlap, -len(unit.text)) > (current[0], -len(current[1].text)):
            best[unit.kind] = (overlap, unit)
    return [unit for _, unit in best.values()]


def _symptom_evidence(query: str, leading: Hypothesis,
                      units: Sequence[Unit]) -> Optional[Unit]:
    """The passage that documents the reported symptom itself.

    Distinct from the anomaly the hypothesis is built on: an incident can have
    a thoroughly documented symptom and no established cause at all, and a
    report that conflates the two misdescribes its own evidence.
    """
    terms = _symptom_terms(query)
    anchor_ids = {unit.unit_id for unit in leading.anchor_units}
    best: Optional[Unit] = None
    best_score = 0.0
    for unit in units:
        if unit.kind != "logs" or unit.unit_id in anchor_ids:
            continue
        score = _coverage(terms, unit.tokens)
        # prefer the narrative summary of a log file over one more log line
        if unit.is_anomaly is False and unit.severity is None:
            score *= 1.5
        if score > best_score:
            best, best_score = unit, score
    return best if best_score > 0.0 else None


def _mechanism_evidence(leading: Hypothesis, units: Sequence[Unit]) -> Optional[Unit]:
    """The design document that explains *why* this failure produces this
    symptom — an architecture or API-spec passage describing the component and
    sharing vocabulary with the failure signature.

    It does not corroborate that the incident happened (so it earns no
    confidence), but it is what lets a reader understand the mechanism.
    """
    signature_tokens = set(_tokenize(leading.signature))
    best: Optional[Unit] = None
    best_overlap = 0
    for unit in units:
        if unit.kind not in ("architecture", "api_spec"):
            continue
        if leading.component not in unit.components:
            continue
        overlap = len(signature_tokens & set(unit.tokens))
        if overlap > best_overlap:
            best, best_overlap = unit, overlap
    return best if best_overlap >= 1 else None


def _exclusivity_evidence(leading: Hypothesis, units: Sequence[Unit]) -> Optional[Unit]:
    """The passage that licenses the "no other change touched it" statement.

    Prefers the most specific passage available: the enclosing section also
    matches, but it is mostly a table, and quoting a table to support a
    sentence is not evidence a reader can check at a glance.
    """
    candidates = [
        unit for unit in units
        if unit.kind == "deployment"
        and leading.component in unit.components
        and _absence_statements(unit.text)
        and not _is_table_skeleton(unit.text)
    ]
    return min(candidates, key=lambda u: len(u.text)) if candidates else None


def _discriminating_check(hypothesis: Hypothesis) -> str:
    """What evidence would actually settle this hypothesis, stated in terms of
    the corroboration it is missing."""
    missing = [k for k in CORROBORATING_KINDS if k not in hypothesis.corroborating_kinds]
    asks = {
        "deployment": "a deployment or config-change record for the component in the window",
        "known_issues": "a known-issue entry matching this signature",
        "runbook": "a runbook describing this symptom",
        "precedent": "a prior incident with the same signature",
    }
    wanted = [asks[k] for k in missing if k in asks]
    if not wanted:
        return "corroborated on every available axis; confirm against live metrics"
    if len(wanted) == 1:
        return f"would be settled by {wanted[0]}"
    return f"would be settled by {wanted[0]}, or failing that {wanted[-1]}"


def _hedged_sentence(text: str) -> str:
    """The specific sentence in a passage that carries the disclaimer."""
    for sentence in re.split(r"(?<=[.!?])\s+", " ".join(text.split())):
        if _has_hedge(sentence):
            return sentence.strip()
    return ""


def _competing_factors(leading: Hypothesis,
                       units: Sequence[Unit]) -> Tuple[List[str], str]:
    """Factors the design documents explicitly name as uninstrumented or outside
    the service's control.

    Where a corpus says "X and Y are both outside this service's control and are
    not currently instrumented", it is naming the competing explanations *and*
    telling you it cannot distinguish them. Those are the real alternatives for
    a thin incident — not other components that the same corpus documents as
    unrelated. Returns (factors, the sentence they came from).
    """
    for unit in _mechanism_evidence_all(leading, units):
        sentence = _hedged_sentence(unit.text)
        if not sentence:
            continue
        # the subject of such a sentence is the list of factors
        subject = re.split(r"\s+(?:are|is|were|was)\s+", sentence, maxsplit=1)[0]
        parts = [
            re.sub(r"^(?:the|a|an)\s+", "", part.strip(" ,"), flags=re.IGNORECASE)
            for part in re.split(r"\s+and\s+|,\s+", subject)
            if len(part.strip(" ,")) > 3
        ]
        if len(parts) >= 2:
            return parts, sentence
    return [], ""


def _uninstrumented_note(leading: Hypothesis, units: Sequence[Unit]) -> str:
    """Where the design documents say a stage is uninstrumented or outside the
    service's control, the corpus is telling you *why* the competing
    explanations cannot be separated. That is worth quoting rather than
    resolving by guesswork."""
    passages = sorted(
        _mechanism_evidence_all(leading, units),
        key=lambda u: 0 if u.kind == "architecture" else 1,
    )
    for unit in passages:
        sentence = _hedged_sentence(unit.text)
        if sentence:
            return _clean_excerpt(sentence, 300)
    return ""


def _rank_scenarios(hypotheses: Sequence[Hypothesis], units: Sequence[Unit],
                    leading: Optional[Hypothesis] = None,
                    limit: int = 3) -> str:
    """Ranked competing explanations with their own calibrated confidences.

    When no hypothesis is well enough supported to stand alone, presenting the
    single best one as though it were the answer is the overconfidence trap.
    The honest output is the shortlist, each with what it rests on and what
    would settle it.
    """
    # Only hypotheses that actually address the reported symptom belong on the
    # shortlist. A corpus documents plenty of unrelated anomalies; offering one
    # as a candidate explanation because nothing better is available is the
    # overconfidence trap wearing a humble face.
    floor = 0.6 * leading.alignment if leading else 0.0
    scored = [
        (h, _score_hypothesis(h))
        for h in hypotheses
        if _score_hypothesis(h) > CONF_FLOOR and h.alignment >= floor and h.alignment > 0
    ]
    scored.sort(key=lambda pair: -pair[1])

    factors, factor_sentence = _competing_factors(leading, units) if leading else ([], "")
    if len(scored) < 2 and len(factors) < 2:
        return ""

    lines: List[str] = [
        "Ranked candidate explanations on the evidence available "
        "(none is established; all require confirmation):"
    ]
    for position, (hypothesis, score) in enumerate(scored[:limit], start=1):
        anchor = hypothesis.anchor_units[0] if hypothesis.anchor_units else None
        observed = (anchor.meta.get("message") if anchor else "") or hypothesis.signature
        support = (
            f"corroborated by {', '.join(hypothesis.corroborating_kinds)}"
            + (" (which disclaims itself)" if hypothesis.hedged_count else "")
            if hypothesis.corroborating_kinds
            else "no corroborating document"
        )
        lines.append(
            f"({position}) {hypothesis.component} — {observed} — confidence "
            f"{score:.0f}/100: {support}; {_discriminating_check(hypothesis)}."
        )

    # Factor-level alternatives within the implicated component. These carry no
    # confidence figure precisely because the corpus states they are not
    # measured — inventing a number for them would be the error this whole
    # pipeline exists to avoid.
    if len(factors) >= 2:
        offset = len(lines) - 1
        for position, factor in enumerate(factors[:limit], start=offset + 1):
            lines.append(
                f"({position}) {factor} — unquantified: named by the design "
                f"documents as a contributor to this path but not instrumented, so "
                f"it can be neither ranked against the above nor excluded."
            )
    return " ".join(lines)


def _select_evidence(leading: Hypothesis, units: Sequence[Unit],
                     symptom: Optional[Unit] = None,
                     limit: int = 11) -> List[Dict[str, str]]:
    """Cite the anchor plus one excerpt per corroborating (or explicitly absent)
    source, so the evidence spans independent documents rather than restating
    the top hit."""
    evidence: List[Dict[str, str]] = []
    seen: set = set()

    def add(source: str, text: str) -> None:
        if _is_table_skeleton(text):
            return  # a table header carries no evidence of its own
        excerpt = _clean_excerpt(text)
        key = (source, excerpt[:80])
        if excerpt and key not in seen:
            seen.add(key)
            evidence.append({"source": source, "excerpt": excerpt})

    # 1. the anomaly itself — first and last occurrence bound the window
    anchors = leading.anchor_units
    for anchor in ([anchors[0], anchors[-1]] if len(anchors) > 1 else anchors):
        add(anchor.source, anchor.text)

    # 2. one excerpt per corroborating document kind
    for corroboration in leading.corroborations:
        add(corroboration.unit.source, corroboration.unit.text)

    # 2b. the downstream component failing in the same instant — the anchor
    #     shows the cause, this shows the customer-facing effect
    stamps = {u.timestamp for u in anchors if u.timestamp}
    for unit in units:
        if (unit.kind == "logs" and unit.is_anomaly and unit.timestamp in stamps
                and unit.unit_id not in {a.unit_id for a in anchors}):
            add(unit.source, unit.text)
            break

    # 2c. the corroborating runbook's actual remediation instruction, which is
    #     what the report's own recommendation rests on
    for corroboration in leading.corroborations:
        if corroboration.kind != "runbook":
            continue
        for block in _split_blocks(corroboration.unit.section_text or ""):
            if re.match(r"^\*{0,2}Remediation", block.strip(), re.IGNORECASE):
                add(corroboration.unit.source, block)
                break
        break

    # 3. the passage licensing any exclusivity claim in the narrative
    exclusivity = _exclusivity_evidence(leading, units)
    if exclusivity is not None:
        add(exclusivity.source, exclusivity.text)

    # 4. the design passages explaining the mechanism, where they exist
    for mechanism in _mechanism_evidence_all(leading, units):
        add(mechanism.source, mechanism.text)

    # 5. the passage documenting the reported symptom itself
    if symptom is not None:
        add(symptom.source, symptom.text)

    # 6. explicitly recorded absences are evidence too — they are why the
    #    confidence is low, and a reviewer needs to see them.
    for _, unit in leading.absences[:3]:
        add(unit.source, unit.text)

    return evidence[:limit]


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #
def investigate(query: str, corpus: Dict[str, str]) -> Dict[str, object]:
    """Investigate one incident and return a structured report.

    corpus: filename -> full document text.
    """
    ranked = _retrieve_relevant_documents(query, corpus)
    evidence = _correlate_evidence(query, corpus, ranked)
    confidence = _calibrate_confidence(evidence)
    leading: Optional[Hypothesis] = evidence.get("leading")  # type: ignore[assignment]
    units: List[Unit] = evidence.get("units", [])  # type: ignore[assignment]

    if leading is None:
        return {
            "root_cause": (
                "Undetermined — no anomalous log entry could be extracted from the "
                "supplied corpus, so no hypothesis is supported by evidence."
            ),
            "supporting_evidence": [],
            "impacted_systems": [],
            "mttr_minutes": None,
            "remediation": (
                "Escalate to a human on-call engineer: the corpus does not contain "
                "the signal needed to form a hypothesis."
            ),
            "confidence_score": CONF_FLOOR,
            "needs_human_review": True,
        }

    confident = confidence >= HUMAN_REVIEW_THRESHOLD
    mttr, mttr_note = _extract_mttr(leading)
    symptom = _symptom_evidence(query, leading, units)
    # Ranked alternatives are offered only where the leading hypothesis is not
    # established. Where it is, a shortlist would manufacture doubt rather than
    # report it.
    hypotheses: List[Hypothesis] = evidence.get("hypotheses", [])  # type: ignore[assignment]
    scenarios = "" if confident else _rank_scenarios(hypotheses, units, leading)
    uninstrumented = "" if confident else _uninstrumented_note(leading, units)
    report: Dict[str, object] = {
        "root_cause": _describe_root_cause(
            leading, confident, mttr_note,
            _sole_change_attested(leading, units), symptom,
            scenarios, uninstrumented,
        ),
        "supporting_evidence": _select_evidence(leading, units, symptom),
        "impacted_systems": _impacted_systems(leading, units),
        "mttr_minutes": mttr,
        "remediation": _extract_remediation(leading, confident),
        "confidence_score": float(confidence),
        "needs_human_review": bool(confidence < HUMAN_REVIEW_THRESHOLD),
    }
    # The MTTR provenance always travels with the report: an expected recovery
    # time drawn from a runbook is a different kind of claim from a measured
    # one, and a reader acting on it needs to know which they have.
    if mttr_note:
        report["remediation"] = f"{report['remediation']} {mttr_note}".strip()
    return report
