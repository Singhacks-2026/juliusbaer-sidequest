"""Offline incident investigation using retrieval and corroborated evidence.

No LLM, network calls, API keys, or third-party dependencies. Run this file
to regenerate answers.json; investigate(query, corpus) is the public API.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path


STOP_WORDS = set("""
a an the and or to of in on for from with by is are was were be been being
it its this that these those as at into no not any all each both then than
after before up out per if when under current recent see check against
identify probable root cause supporting evidence impacted component components
recommended remediation what mean time recover systems yesterday s customers
customer reporting report sometimes over hour purchase arriving
service adapter gateway log logs logging exception error warn info request
""".split())
ALIASES = {
    "payments": "payment", "charge": "payment", "charges": "payment",
    "emails": "email", "notification": "email", "confirmation": "email",
    "orders": "order", "failed": "failure", "failing": "failure",
    "failures": "failure", "fails": "failure", "fail": "failure",
    "intermittently": "intermittent", "delays": "delay", "delayed": "delay",
    "late": "delay", "latency": "delay", "connections": "connection",
    "pooled": "pool", "consumers": "consumer", "workers": "worker",
    "messages": "message", "warnings": "warning", "timeouts": "timeout",
    "reduced": "reduction", "reducing": "reduction", "deployments": "deployment",
    "configured": "configuration", "config": "configuration",
}
LOG_RE = re.compile(
    r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\s+"
    r"(TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL)\s+(\S+)\s+(.+)$", re.M
)
UNCERTAIN_RE = re.compile(
    r"unconfirmed|unverified|may not apply|incomplete|pending better instrumentation", re.I
)
SIGNATURE_RE = re.compile(
    r"\b[A-Za-z][A-Za-z0-9]*(?:Exception|Error)\b|\b[A-Z]+(?:_[A-Z]+)+\b"
)
CHANGE_WINDOW = timedelta(days=14)
OPERATIONS = {"payment", "email", "refund", "webhook", "order", "search", "login", "checkout"}


@dataclass
class Chunk:
    source: str
    kind: str
    excerpt: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Hypothesis:
    event: Chunk
    relevance: float
    issue: Chunk | None = None
    history: Chunk | None = None
    runbook: Chunk | None = None
    deployment: Chunk | None = None
    ambiguous: bool = False
    conflicts: list[Chunk] = field(default_factory=list)
    onset_event: Chunk | None = None


def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[*`#]", "", text)).strip()


def _tokens(text: str) -> list[str]:
    # Strip transaction identifiers/timestamps; split CamelCase signatures.
    text = re.sub(r"\b\w+_id=\S+", " ", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    words = re.findall(r"[a-z]+", text.lower())
    return [ALIASES.get(w, w) for w in words if w not in STOP_WORDS and len(w) > 1]


def _section(text: str, label: str) -> str:
    """Extract one labeled Markdown field without swallowing the next field."""
    pattern = rf"\*\*{re.escape(label)}\s*:?(?:\*\*)\s*:?\s*(.*?)(?=\n\s*\*\*|\n##|\Z)"
    match = re.search(pattern, text, re.S | re.I)
    return _plain(match.group(1)) if match else ""


def _document_kind(text: str) -> str:
    # Content-based classification keeps renaming a source from changing results.
    title = text.splitlines()[0].lower() if text.splitlines() else ""
    if "runbook" in title or "**Diagnostic steps" in text:
        return "runbook"
    if "previous incident" in title or "**Root cause**" in text:
        return "history"
    if "deployment" in title or "| Version | Timestamp" in text:
        return "deployment"
    if "architecture" in title:
        return "architecture"
    if "api spec" in title:
        return "api"
    return "document"


def _ingest_corpus(corpus: dict) -> list[Chunk]:
    chunks = []
    for source, text in sorted(corpus.items()):
        if not isinstance(source, str) or not isinstance(text, str):
            raise TypeError("corpus must map string filenames to document strings")
        if not text.strip():
            continue
        reader = csv.DictReader(io.StringIO(text))
        if {"issue_id", "signature", "affected_component"} <= set(reader.fieldnames or []):
            lines = text.splitlines(keepends=True)
            previous_line = reader.line_num
            for row in reader:
                # reader.line_num includes quoted multiline CSV records.
                excerpt = "".join(lines[previous_line:reader.line_num]).strip()
                previous_line = reader.line_num
                chunks.append(Chunk(source, "issue", excerpt, dict(row)))
            continue
        events = list(LOG_RE.finditer(text))
        if events:
            seen = set()
            for match in events:
                if match.group(0) in seen:
                    continue
                seen.add(match.group(0))
                stamp, level, component, message = match.groups()
                chunks.append(Chunk(source, "log", match.group(0), {
                    "time": datetime.fromisoformat(stamp), "level": level,
                    "component": component, "message": message,
                }))
            # Interpret observations from events, not the author's summary.
            continue
        kind = _document_kind(text)
        if kind == "deployment":
            for block in re.split(r"\n\s*\n", text):
                for line in block.splitlines():
                    cells = [_plain(c) for c in line.strip().strip("|").split("|")]
                    if len(cells) != 4 or not re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d", cells[1]):
                        continue
                    version, stamp, component, change = cells
                    chunks.append(Chunk(source, kind, line, {
                        "time": datetime.fromisoformat(stamp), "version": version,
                        "component": component, "change": change,
                    }))
                if "|" not in block and not block.startswith("#"):
                    chunks.append(Chunk(source, kind, block.strip()))
            continue
        # Keep each runbook / prior incident intact, including caveats and MTTR.
        for section in re.split(r"(?m)(?=^## )", text):
            blocks = [section] if section.startswith("## ") else re.split(r"\n\s*\n", section)
            for block in blocks:
                if block.strip() and not re.fullmatch(r"#[^\n]+", block.strip()):
                    chunks.append(Chunk(source, kind, block.strip()))
    return chunks


def _retrieve_relevant_documents(query: str, chunks: list[Chunk]) -> list[tuple[Chunk, float]]:
    """Sublinear TF-IDF cosine ranking, implemented with the standard library."""
    if not chunks:
        return []
    texts = []
    for chunk in chunks:
        if chunk.kind == "log":
            text = chunk.metadata["component"] + " " + chunk.metadata["message"]
        else:
            text = chunk.excerpt
        texts.append(Counter(_tokens(text)))
    frequencies = Counter(token for document in texts for token in document)
    idf = {token: 1 + math.log((len(texts) + 1) / (count + 1))
           for token, count in frequencies.items()}

    def vector(counts: Counter) -> dict:
        values = {t: (1 + math.log(n)) * idf[t] for t, n in counts.items() if t in idf}
        norm = math.sqrt(sum(v * v for v in values.values())) or 1
        return {t: v / norm for t, v in values.items()}

    target = vector(Counter(_tokens(query)))
    ranked = []
    for chunk, counts in zip(chunks, texts):
        score = sum(target.get(t, 0) * v for t, v in vector(counts).items())
        ranked.append((chunk, score))
    return sorted(ranked, key=lambda item: (-item[1], item[0].excerpt, item[0].source))


def _matches_symptom(event: Chunk, candidate: Chunk) -> bool:
    component = event.metadata["component"]
    if component not in candidate.excerpt:
        return False
    if candidate.kind == "issue" and candidate.metadata["affected_component"] != component:
        return False
    text = (candidate.metadata.get("signature", "") if candidate.kind == "issue"
            else _section(candidate.excerpt, "Symptoms") or _section(candidate.excerpt, "Summary"))
    if not text:
        return False  # A "no previous incident" paragraph is not a precedent.
    message = event.metadata["message"]
    signatures = set(SIGNATURE_RE.findall(message))
    if signatures:
        return any(_positive_mention(re.escape(signature), text)
                   for signature in signatures & set(SIGNATURE_RE.findall(text)))
    remove = set(_tokens(component)) | {"failure", "warning", "elevated"}
    observed = set(_tokens(message)) - remove
    documented = {token for match in re.finditer(r"[A-Za-z]+", text)
                  if not _negated_at(text, match.start(), match.end())
                  for token in _tokens(match[0])} - remove
    common = observed & documented
    return len(common) >= 2 and len(common) / max(1, len(observed)) >= 0.5


def _phenomena(text: str) -> set[str]:
    """Keep symptom compatibility separate from component-name similarity."""
    words = set(_tokens(text))
    result = set()
    if words & {"delay", "timeout", "lag", "backlog", "slow", "slowness"} or re.search(
            r"queue\s+(?:depth|age).*elevated", _plain(text), re.I):
        result.add("delay")
    if "failure" in words or re.search(r"\bERROR\b|\bFATAL\b|\w+Exception\b", text):
        result.add("failure")
    return result


def _operation_compatible(query: str, message: str) -> bool:
    wanted = set(_tokens(query)) & OPERATIONS
    observed = set(_tokens(message)) & OPERATIONS
    return not wanted or bool(wanted & observed)


def _infrastructure_link(event: Chunk, query: str, chunks: list[Chunk]) -> bool:
    """Require resource-level context for events without a business operation."""
    message_words = set(_tokens(event.metadata["message"]))
    if message_words & OPERATIONS:
        return False  # An explicitly different operation cannot use this bridge.
    wanted = set(_tokens(query)) & OPERATIONS
    component = event.metadata["component"]
    meaningful = message_words - {"timeout", "delay", "elevated", "warning", "ms"}
    for chunk in chunks:
        if chunk.kind != "architecture":
            continue
        for paragraph in re.split(r"\n\s*\n|\n(?=- )", chunk.excerpt):
            if component not in paragraph or re.search(r"independent|no evidence", paragraph, re.I):
                continue
            context = set(_tokens(paragraph))
            if (not wanted or wanted & context) and len(meaningful & context) >= 2:
                return True
    return False


def _negated_at(text: str, start: int, end: int) -> bool:
    prefix = re.split(r"[.;:!?()]", text[:start])[-1]
    previous_words = re.findall(r"[\w']+", prefix.lower())[-6:]
    if set(previous_words) & {"not", "no", "never", "without", "neither", "cannot",
                              "can't", "doesn't", "isn't", "wasn't"}:
        return True
    suffix = text[end:]
    return bool(re.match(r"\s+(?:(?:is|are|was|were)\s+)?(?:not|never)\s+"
                         r"(?:seen|observed|present|logged|detected)", suffix, re.I))


def _positive_mention(pattern: str, text: str) -> bool:
    text = _plain(text)
    return any(not _negated_at(text, match.start(), match.end())
               for match in re.finditer(pattern, text, re.I))


def _negative_claim(text: str) -> bool:
    return bool(re.search(
        r"\b(?:not|never)\s+(?:a\s+)?(?:known\s+)?signature\s+of\b|"
        r"\bnot\s+(?:caused by|due to)\b|\bno evidence\s+(?:of|for)\b|"
        r"\bruled out\b|\bunrelated to\b", _plain(text), re.I))


def _cause_text(document: Chunk | None) -> str:
    if document is None or UNCERTAIN_RE.search(document.excerpt):
        return ""
    text = (document.metadata.get("signature", "") if document.kind == "issue"
            else _section(document.excerpt, "Root cause"))
    if _negative_claim(text):
        return ""
    if document.kind == "issue":
        match = re.search(r"(?:known )?signature of\s+(.+?)(?:;|$)", text, re.I)
        return _plain(match[1]).rstrip(".") if match else ""
    return text.rstrip(".")


def _cause_mechanisms(text: str) -> set[str]:
    """A bounded vocabulary for comparing causes, not just shared resources."""
    result = set()
    patterns = {
        "capacity_shortage": r"undersized|too (?:low|small)|insufficient (?:capacity|pool|consumer)|"
                             r"(?:reduc|decreas)\w* (?:connection )?pool",
        "resource_leak": r"\bleak\w*\b|not released|never released",
        "throttling": r"\bthrottl\w*\b|rate limit",
    }
    for mechanism, pattern in patterns.items():
        if _positive_mention(pattern, text):
            result.add(mechanism)
    return result


def _causes_agree(first: str, second: str) -> bool:
    a, b = _cause_mechanisms(first), _cause_mechanisms(second)
    if a or b:
        return bool(a and b and a == b)
    # Shared component/resource nouns alone do not demonstrate cause agreement.
    context = {"connection", "pool", "size", "traffic", "configuration", "change", "payment"}
    left, right = set(_tokens(first)) - context, set(_tokens(second)) - context
    return len(left & right) >= 2 and len(left & right) / max(1, min(len(left), len(right))) >= 0.6


def _find_deployment(event: Chunk, chunks: list[Chunk], onset: datetime, cause: str) -> Chunk | None:
    event_words = set(_tokens(event.metadata["message"]))
    component = event.metadata["component"]
    candidates = [c for c in chunks if c.kind == "deployment" and c.metadata
                  and c.metadata["component"] == component
                  and timedelta(0) <= onset - c.metadata["time"] <= CHANGE_WINDOW]
    # This is a change log, not a sequence of complete configuration snapshots.
    # Ignore unrelated releases, but let a later change to the same resource
    # supersede an earlier one (including an explicit rollback or increase).
    candidates.sort(key=lambda c: (c.metadata["time"], c.excerpt), reverse=True)
    cause_words = set(_tokens(cause))
    resources = cause_words & {"pool", "consumer", "worker", "buffer", "timeout", "memory", "cpu"}
    if not resources:
        resources = cause_words & {"connection", "provider", "database"}
    relevant_changes = [candidate for candidate in candidates
                        if (any(_positive_mention(rf"\b{re.escape(resource)}\b", candidate.metadata["change"])
                                for resource in resources) if resources
                            else _causes_agree(cause, candidate.metadata["change"]))]
    if not relevant_changes:
        return None
    change = relevant_changes[0]
    if not cause or not _causes_agree(cause, change.metadata["change"]):
        return None
    if re.search(r"\b(?:fixed?|resolved?|reverted?|restored?|increased?|expanded?)\b",
                 change.metadata["change"], re.I):
        return None  # A mitigation is not evidence that the release caused it.
    words = set(_tokens(change.metadata["change"]))
    if len((event_words & words) - set(_tokens(component))) < 2:
        return None
    # A capacity increase is not corroboration for an undersized pool.
    if "pool" in event_words and "pool" in words:
        sizes = re.search(r"from\s+(\d+)\s+to\s+(\d+)", change.metadata["change"], re.I)
        if sizes and int(sizes[2]) >= int(sizes[1]):
            return None
        if not sizes and not re.search(r"reduc|decreas|lower|shrink", change.metadata["change"], re.I):
            return None
    return change


def _scope(query: str, chunks: list[Chunk]) -> tuple[set[str], list[Chunk]]:
    logs = [c for c in chunks if c.kind == "log"]
    # A deployment is temporal context, not the user-facing operation that
    # failed. Otherwise a deploy-agent INFO line can outrank actual symptoms.
    symptom = " ".join(t for t in _tokens(query) if t not in {"deployment", "deploy", "release"})
    ranked = _retrieve_relevant_documents(symptom, logs)
    if not ranked or ranked[0][1] <= 0:
        return set(), logs
    scores = {}
    for event, score in ranked:
        if re.search(r"customer report|user report|reported by|customer complaint", event.metadata["message"], re.I):
            continue
        if (not _operation_compatible(query, event.metadata["message"])
                and not _infrastructure_link(event, query, chunks)):
            continue
        component = event.metadata["component"]
        scores[component] = max(scores.get(component, 0), score)
    # Correlated queued/sent measurements are stronger localization evidence
    # than a single observer log paraphrasing the user's complaint.
    if "delay" in _phenomena(query):
        measured = set()
        for start, end, minutes in _delivery_pairs(logs, set(scores)):
            if minutes > 0 and _operation_compatible(query, start.metadata["message"]):
                measured.add(start.metadata["component"])
        for component in measured:
            scores[component] += 1
    if not scores or max(scores.values()) <= 0:
        return set(), logs
    primary = min(scores, key=lambda component: (-scores[component], component))
    components = {primary}
    # Follow a synchronous error into another component only if both the
    # timestamp and an architecture/API relationship support that connection.
    errors = [c for c in logs if c.metadata["level"] in {"ERROR", "FATAL"}]
    primary_errors = [c for c in errors if c.metadata["component"] == primary]
    for event in errors:
        peer = event.metadata["component"]
        correlated = any(abs(event.metadata["time"] - e.metadata["time"]) <= timedelta(seconds=1)
                         for e in primary_errors)
        connected = False
        for chunk in chunks:
            if chunk.kind not in {"architecture", "api"}:
                continue
            for paragraph in re.split(r"\n\s*\n|\n(?=- )", chunk.excerpt):
                if (primary in paragraph and peer in paragraph
                        and re.search(r"\bcalls?\b|delegates to|direct path", paragraph, re.I)
                        and not re.search(r"independent|does not call|no evidence", paragraph, re.I)):
                    connected = True
        if correlated and connected:
            components.add(peer)
    return components, logs


def _correlate_evidence(query: str, chunks: list[Chunk]) -> tuple[list[Hypothesis], set[str], list[Chunk]]:
    components, logs = _scope(query, chunks)
    relevant = [c for c in logs if c.metadata["component"] in components]
    abnormal = [c for c in relevant if c.metadata["level"] in {"WARN", "WARNING", "ERROR", "FATAL"}]
    ranking = _retrieve_relevant_documents(query, abnormal)
    unique = {}
    for event, score in ranking:
        wanted = _phenomena(query)
        observed = _phenomena(event.metadata["level"] + " " + event.metadata["message"])
        # A causally linked downstream component can use entirely different
        # vocabulary from the query; scope correlation supplies that bridge.
        compatible = (_operation_compatible(query, event.metadata["message"])
                      or _infrastructure_link(event, query, chunks))
        if (wanted and not wanted & observed) or not compatible:
            continue
        key = (event.metadata["component"], tuple(_tokens(event.metadata["message"])))
        unique.setdefault(key, []).append((event, score))
    hypotheses = []
    severity = {"FATAL": 3, "ERROR": 2, "WARN": 1, "WARNING": 1}
    for observations in unique.values():
        event, score = min(observations, key=lambda item: (
            -severity[item[0].metadata["level"]], item[0].metadata["time"], item[0].excerpt))
        onset_event = min((item[0] for item in observations), key=lambda c: (c.metadata["time"], c.excerpt))
        hypothesis = Hypothesis(event, score, onset_event=onset_event)
        # The second retrieval uses the observed mechanism, not just the query.
        expanded = query + " " + event.metadata["component"] + " " + event.metadata["message"]
        ranked = _retrieve_relevant_documents(expanded, chunks)
        for kind, attr in (("issue", "issue"), ("history", "history"), ("runbook", "runbook")):
            for candidate, relevance in ranked:
                if relevance <= 0 or candidate.kind != kind or not _matches_symptom(event, candidate):
                    continue
                if kind == "history":
                    dates = re.findall(r"\b\d{4}-\d\d-\d\d\b", candidate.excerpt.splitlines()[0])
                    if dates and datetime.fromisoformat(dates[0]) >= event.metadata["time"]:
                        continue
                if kind == "issue" and not _issue_applicable(candidate, onset_event.metadata["time"], chunks):
                    hypothesis.conflicts.append(candidate)
                    continue
                claim = (candidate.metadata.get("signature", "") if kind == "issue"
                         else _section(candidate.excerpt, "Root cause"))
                if _negative_claim(claim):
                    hypothesis.conflicts.append(candidate)
                    continue
                selected = getattr(hypothesis, attr)
                if selected is None:
                    setattr(hypothesis, attr, candidate)
                elif kind in {"issue", "history"}:
                    selected_cause, other_cause = _cause_text(selected), _cause_text(candidate)
                    if selected_cause and other_cause and not _causes_agree(selected_cause, other_cause):
                        hypothesis.conflicts.append(candidate)
        issue_cause, historical_cause = _cause_text(hypothesis.issue), _cause_text(hypothesis.history)
        if issue_cause and historical_cause and not _causes_agree(issue_cause, historical_cause):
            hypothesis.conflicts.append(hypothesis.history)
            hypothesis.history = None
        hypothesis.deployment = _find_deployment(event, chunks, onset_event.metadata["time"], _cause_description(hypothesis))
        hypotheses.append(hypothesis)
    hypotheses.sort(key=lambda h: (-_calibrate_confidence(h), -h.relevance, h.event.excerpt))
    if len(hypotheses) > 1:
        leading, alternate = hypotheses[:2]
        # Multiple equally supported mechanisms should not produce certainty.
        if (_calibrate_confidence(leading) >= 50 and
                _calibrate_confidence(leading) - _calibrate_confidence(alternate) < 10):
            leading.ambiguous = True
    return hypotheses, components, logs


def _issue_applicable(issue: Chunk, onset: datetime, chunks: list[Chunk]) -> bool:
    notes = issue.metadata.get("notes") or ""
    if re.search(r"no longer applicable|unrelated|(?:fixed|resolved).*before this incident", notes, re.I):
        return False
    fixed_version = re.search(r"(?:fixed|resolved)\s+in\s+(v[\w.-]+)", notes, re.I)
    if fixed_version:
        version = fixed_version[1].rstrip(".")
        if any(c.kind == "deployment" and c.metadata
               and c.metadata["component"] == issue.metadata["affected_component"]
               and c.metadata["version"] == version and c.metadata["time"] <= onset
               for c in chunks):
            return False
    fixed_date = re.search(r"(?:fixed|resolved)\s+(?:on\s+)?(\d{4}-\d\d-\d\d)", notes, re.I)
    return not (fixed_date and datetime.fromisoformat(fixed_date[1]) <= onset)


def _calibrate_confidence(hypothesis: Hypothesis | None) -> float:
    """An evidence-support heuristic, not a statistically calibrated probability.

    Each source type contributes at most once. Architecture explains impact;
    it does not independently corroborate a cause. Qualified runbooks do not
    substantiate a cause or supply an applicable MTTR.
    """
    if hypothesis is None:
        return 5.0
    direct = hypothesis.event.metadata["level"] in {"ERROR", "FATAL"}
    score = 5 + (20 if direct else 6)
    strong = 0
    for document, weight in ((hypothesis.issue, 20), (hypothesis.history, 20),
                             (hypothesis.deployment, 25)):
        if document and not UNCERTAIN_RE.search(document.excerpt):
            score += weight
            strong += 1
    if hypothesis.runbook:
        score += 2 if UNCERTAIN_RE.search(hypothesis.runbook.excerpt) else 10
    if not direct:
        score = min(score, 35)
    if strong < 2 or hypothesis.ambiguous or hypothesis.conflicts or not _cause_description(hypothesis):
        score = min(score, 45)
    return float(min(score, 95))


def _delivery_pairs(logs: list[Chunk], components: set[str]) -> list[tuple[Chunk, Chunk, float]]:
    queued = {}
    pairs = []
    for event in sorted(logs, key=lambda c: (c.metadata["time"], c.excerpt)):
        if event.metadata["component"] not in components:
            continue
        message = event.metadata["message"]
        identifier = re.search(r"\b(?:order|message|correlation)_id=(\S+)", message)
        if not identifier:
            continue
        key = (event.metadata["component"], identifier[0])
        if re.search(r"\bqueued\b", message, re.I):
            queued.setdefault(key, event)
        elif re.search(r"\bsent\b", message, re.I) and key in queued:
            start = queued.pop(key)
            minutes = (event.metadata["time"] - start.metadata["time"]).total_seconds() / 60
            pairs.append((start, event, minutes))
    return pairs


def _cause_description(hypothesis: Hypothesis) -> str:
    return _cause_text(hypothesis.issue) or _cause_text(hypothesis.history)


def _mttr(hypothesis: Hypothesis | None, confidence: float) -> tuple[int | None, str]:
    if not hypothesis or confidence < 50:
        return None, "Recovery time cannot be reliably estimated from the available evidence."
    for chunk, label in ((hypothesis.runbook, "matched runbook's typical recovery estimate"),
                         (hypothesis.history, "matched historical incident's recovery duration")):
        if chunk is None or UNCERTAIN_RE.search(chunk.excerpt):
            continue
        match = re.search(r"(?:Typical\s+)?MTTR\s*:\s*(\d+)\s*minutes", _plain(chunk.excerpt), re.I)
        if match:
            return int(match[1]), f"MTTR uses the {label} ({match[1]} minutes), not a measured recovery of this incident."
    return None, "No applicable recovery-time estimate was found."


def _build_report(query: str, chunks: list[Chunk], hypotheses: list[Hypothesis],
                  components: set[str], logs: list[Chunk]) -> dict:
    leading = hypotheses[0] if hypotheses else None
    confidence = _calibrate_confidence(leading)
    supporting = []

    def cite(chunk: Chunk | None, excerpt: str | None = None) -> None:
        if chunk is None:
            return
        item = {"source": chunk.source, "excerpt": excerpt if excerpt is not None else chunk.excerpt}
        if item not in supporting:
            supporting.append(item)

    pairs = _delivery_pairs(logs, components)
    if leading:
        cite(leading.event)
        cite(leading.onset_event)
        for document in (leading.deployment, leading.issue, leading.history, leading.runbook):
            cite(document)
        for document in leading.conflicts:
            cite(document)
        component = leading.event.metadata["component"]
        cause = _cause_description(leading)
        if confidence >= 50 and cause:
            root = f"Probable root cause: {cause} in {component}."
            if leading.deployment:
                meta = leading.deployment.metadata
                root += (f" Deployment {meta['version']} at {meta['time']:%Y-%m-%d %H:%M} "
                         f"changed this component: {meta['change']}. "
                         f"The first matching signal in the supplied logs occurred at "
                         f"{(leading.onset_event or leading.event).metadata['time']:%H:%M:%S}, after that change.")
            sources = [label for document, label in (
                (leading.issue, "known-issue signature"),
                (leading.history, "historical precedent"),
                (leading.runbook, "runbook"))
                if document and not UNCERTAIN_RE.search(document.excerpt)]
            if sources:
                root += " The matching " + ", ".join(sources) + " support this mechanism."
        else:
            root = f"Root cause remains unconfirmed for {component}."
            if cause:
                root += f" A candidate explanation is {cause}, but independent corroboration is insufficient."
            else:
                root += f" The observed warning/failure ({leading.event.metadata['message']}) does not establish its cause."
            if not leading.deployment:
                root += " No temporally and mechanistically correlated deployment was found in the supplied corpus."
            if not leading.issue:
                root += " No known-issue signature matched this observation."
            if not leading.history:
                root += " No matching historical precedent was found."
            if leading.conflicts:
                root += " Documents with the same symptom negate the proposed mechanism, describe a different cause, or are no longer applicable; they are conflicting evidence, not corroboration."
        remediation_parts = []
        has_action = False
        if leading.runbook:
            diagnostic = _section(leading.runbook.excerpt, "Diagnostic steps")
            action = _section(leading.runbook.excerpt, "Remediation")
            if diagnostic:
                remediation_parts.append("Investigate: " + diagnostic)
            if action:
                if UNCERTAIN_RE.search(action):
                    prefix = "Unverified suggestion (validate before acting): "
                else:
                    prefix = "Recommended remediation: " if confidence >= 50 else "Only after confirming the bottleneck, consider: "
                remediation_parts.append(prefix + action)
                has_action = True
        if confidence >= 50:
            if not has_action and leading.history:
                resolution = _section(leading.history.excerpt, "Resolution")
                if resolution:
                    prefix = ("Unverified suggestion (validate before acting): " if UNCERTAIN_RE.search(resolution)
                              else "Consider the matched historical resolution: ")
                    remediation_parts.append(prefix + resolution)
                    has_action = True
            if has_action:
                remediation_parts.append("After the change, verify that the relevant failure signature stops and successful requests resume without recurrence.")
            else:
                remediation_parts.append("No applicable corrective action was documented; confirm a remediation with the component owner.")
        else:
            remediation_parts.append("Escalate for human investigation; collect missing metrics before selecting a corrective action.")
    else:
        root = "Root cause undetermined: no sufficiently relevant abnormal log event was found in the supplied corpus."
        remediation_parts = ["Collect timestamped logs and metrics for the reported symptom, correlate affected requests across components, and request human investigation."]

    if pairs:
        durations = [p[2] for p in pairs]
        root += (f" {len(pairs)} correlated queued-to-sent deliveries took "
                 f"{min(durations):.1f}–{max(durations):.1f} minutes; these are delivery delays, not recovery times.")
        # Cite the minimum/maximum measurements using both original log events.
        for pair in (min(pairs, key=lambda p: p[2]), max(pairs, key=lambda p: p[2])):
            cite(pair[0])
            cite(pair[1])
        if confidence < 50:
            relevant_context = " ".join(c.excerpt for c in chunks if c.kind == "architecture"
                                        and any(component in c.excerpt for component in components))
            if "consumer" in relevant_context.lower() and "provider" in relevant_context.lower():
                root += " Consumer throughput and downstream provider latency remain competing, unverified explanations; queued-to-sent logs do not distinguish them."
                remediation_parts.append("Instrument queue age/depth, consumer throughput/count, and provider request latency with per-stage timestamps and correlation IDs.")

    # Cite symptoms in other affected components and a successful control event.
    if leading:
        stamp = leading.event.metadata["time"]
        matching = [c for c in logs if c.metadata["component"] == leading.event.metadata["component"]
                    and _tokens(c.metadata["message"]) == _tokens(leading.event.metadata["message"])]
        if matching:
            cite(max(matching, key=lambda c: c.metadata["time"]))
        for component in sorted(components):
            related = [c for c in logs if c.metadata["component"] == component]
            failures = [c for c in related if c.metadata["level"] in {"ERROR", "FATAL"}
                        and abs(c.metadata["time"] - stamp) <= timedelta(seconds=1)]
            if failures:
                cite(min(failures, key=lambda c: c.excerpt))
            successes = [c for c in related
                         if re.search(r"\bsucceeded\b", c.metadata["message"], re.I)]
            before = [c for c in successes if c.metadata["time"] < stamp]
            after = [c for c in successes if c.metadata["time"] > stamp]
            if before:
                cite(max(before, key=lambda c: c.metadata["time"]))
            if after:
                cite(min(after, key=lambda c: c.metadata["time"]))
                if any(c.metadata["time"] > after[0].metadata["time"] for c in matching):
                    root += " Successful requests are interleaved with repeated failures, indicating intermittent degradation rather than a total outage."

    for kind in ("architecture", "api"):
        candidates = [c for c in chunks if c.kind == kind
                      and any(component in c.excerpt for component in components)]
        ranked = _retrieve_relevant_documents(query, candidates)
        if ranked:
            cite(ranked[0][0])
    if confidence < 50:
        # Explicit absence/caveat statements support uncertainty, not a cause.
        for chunk in chunks:
            if (chunk.kind in {"deployment", "history"} and not chunk.metadata
                    and any(component in chunk.excerpt for component in components)
                    and re.search(r"\bno\b|unrelated", chunk.excerpt, re.I)):
                cite(chunk)
    mttr, qualification = _mttr(leading, confidence)
    remediation_parts.append(qualification)
    return {
        "root_cause": root,
        "supporting_evidence": supporting,
        "impacted_systems": sorted(components),
        "mttr_minutes": mttr,
        "remediation": " ".join(remediation_parts),
        "confidence_score": confidence,
        "needs_human_review": confidence < 50,
    }


def investigate(query: str, corpus: dict) -> dict:
    """Investigate one incident using only its supplied filename -> text corpus."""
    if not isinstance(query, str) or not isinstance(corpus, dict):
        raise TypeError("query must be a string and corpus must be a dict")
    # Stop words remove boilerplate without truncating instruction-first or
    # multi-paragraph descriptions of the actual symptom.
    symptom = query.strip()
    chunks = _ingest_corpus(corpus)
    hypotheses, components, logs = _correlate_evidence(symptom, chunks)
    report = _build_report(symptom, chunks, hypotheses, components, logs)
    for evidence in report["supporting_evidence"]:
        if evidence["excerpt"] not in corpus[evidence["source"]]:
            raise ValueError("Evidence excerpt must be verbatim source text")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("answers.json"))
    args = parser.parse_args()
    answers = {}
    for query_path in sorted(args.data_dir.glob("*/query.txt")):
        corpus = {path.name: path.read_text(encoding="utf-8")
                  for path in sorted(query_path.parent.iterdir()) if path.suffix in {".md", ".csv"}}
        answers[query_path.parent.name] = investigate(query_path.read_text(encoding="utf-8").strip(), corpus)
    if not answers:
        parser.error(f"No incident query.txt files found in {args.data_dir}")
    args.output.write_text(json.dumps(answers, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(answers)} incident reports to {args.output}")


if __name__ == "__main__":
    main()
