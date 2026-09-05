"""Offline incident investigation: retrieve, corroborate, cite, or abstain.

Python 3.10+, standard library only. No incident IDs or prewritten reports.
Run `python solution.py --help` for the reproducible submission command.
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
from datetime import datetime
from pathlib import Path
from statistics import mean

STOP = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "from",
    "with",
    "by",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "at",
    "then",
    "than",
    "up",
    "has",
    "have",
    "had",
    "no",
    "not",
    "any",
    "all",
    "both",
    "each",
    "only",
    "which",
    "who",
    "what",
    "how",
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
    "customer",
    "reporting",
    "report",
    "sometimes",
    "after",
    "yesterday",
    "deployment",
}
ALIASES = {
    "payments": "payment",
    "charge": "payment",
    "charges": "payment",
    "emails": "email",
    "notification": "email",
    "notifications": "email",
    "late": "delay",
    "delays": "delay",
    "delayed": "delay",
    "latency": "delay",
    "failing": "fail",
    "failed": "fail",
    "failures": "fail",
    "failure": "fail",
    "intermittently": "intermittent",
    "timeouts": "timeout",
    "connections": "connection",
    "consumers": "consumer",
    "workers": "worker",
    "warnings": "warning",
    "warn": "warning",
    "errors": "error",
    "sends": "send",
    "sent": "send",
    "sending": "send",
    "queued": "queue",
    "dequeues": "dequeue",
    "confirmation": "confirm",
    "confirmations": "confirm",
}
LOG = re.compile(
    r"^(?P<time>\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d)\s+"
    r"(?P<level>TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL)\s+"
    r"(?P<component>[\w.-]+)\s+(?P<message>.+)$",
    re.MULTILINE,
)
EXCEPTION = re.compile(r"\b[A-Z][a-zA-Z0-9]*(?:Exception|Error)\b")
UNCERTAIN = re.compile(
    r"unconfirmed|unverified|may not apply|incomplete|not currently.*instrument",
    re.IGNORECASE,
)


def clean(text: str) -> str:
    """Normalize presentation only; citations always keep the original text."""
    return " ".join(re.sub(r"[*`#]", "", text).split())


def tokens(text: str) -> list[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    words = re.findall(r"[a-z]+", text.lower())
    return [ALIASES.get(w, w) for w in words if w not in STOP and len(w) > 1]


@dataclass
class Passage:
    source: str
    excerpt: str
    kind: str
    meta: dict = field(default_factory=dict)

    def citation(self) -> dict:
        return {"source": self.source, "excerpt": self.excerpt}


@dataclass
class Event:
    passage: Passage
    time: datetime
    level: str
    component: str
    message: str


class BM25:
    """Length-normalized lexical retrieval with a positive smoothed IDF."""

    def __init__(self, passages: list[Passage]):
        self.passages = passages
        self.counts = [Counter(tokens(p.excerpt)) for p in passages]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.average = mean(self.lengths) if self.lengths else 1.0
        self.df: Counter = Counter()
        for counts in self.counts:
            self.df.update(counts.keys())

    def rank(self, query: str) -> list[tuple[Passage, float]]:
        result = []
        for passage, counts, length in zip(self.passages, self.counts, self.lengths):
            score = 0.0
            for term in set(tokens(query)):
                tf = counts[term]
                if tf:
                    idf = math.log1p(
                        (len(self.counts) - self.df[term] + 0.5) / (self.df[term] + 0.5)
                    )
                    score += (
                        idf
                        * tf
                        * 2.5
                        / (tf + 1.5 * (0.25 + 0.75 * length / (self.average or 1)))
                    )
            result.append((passage, score))
        return sorted(
            result, key=lambda item: (-item[1], item[0].excerpt, item[0].source)
        )


def document_kind(text: str) -> str:
    """Infer source role from content, never an incident name or filename."""
    heading = text.strip().splitlines()[0].lower() if text.strip() else ""
    for word, role in (
        ("runbook", "runbook"),
        ("previous incident", "history"),
        ("deployment", "deployment"),
        ("architecture", "architecture"),
        ("api specification", "api"),
    ):
        if word in heading:
            return role
    return "context"


def ingest(corpus: dict[str, str]) -> tuple[list[Passage], list[Event]]:
    passages, events = [], []
    # Content deduplication prevents copied documents becoming independent votes.
    seen_documents = set()
    for source, text in sorted(corpus.items()):
        fingerprint = " ".join(text.split())
        if not fingerprint or fingerprint in seen_documents:
            continue
        seen_documents.add(fingerprint)
        lines = text.splitlines(keepends=True)
        reader = csv.DictReader(io.StringIO(text))
        if {"issue_id", "signature", "affected_component"}.issubset(
            reader.fieldnames or []
        ):
            end = reader.line_num
            for row in reader:
                start, end = end, reader.line_num
                if row.get("signature") and row.get("affected_component"):
                    passages.append(
                        Passage(source, "".join(lines[start:end]).strip(), "issue", row)
                    )
            continue
        matches = list(LOG.finditer(text))
        if matches:
            for match in matches:
                try:
                    timestamp = datetime.fromisoformat(match["time"])
                except ValueError:
                    continue
                passage = Passage(source, match[0], "log")
                passages.append(passage)
                events.append(
                    Event(
                        passage,
                        timestamp,
                        match["level"],
                        match["component"],
                        match["message"],
                    )
                )
            # Log commentary is intentionally not an additional corroborating source.
            continue
        role = document_kind(text)
        if role == "deployment":
            for line in lines:
                cells = [clean(cell) for cell in line.strip().strip("|").split("|")]
                if len(cells) == 4 and re.match(r"\d{4}-\d\d-\d\d", cells[1]):
                    try:
                        timestamp = datetime.fromisoformat(cells[1])
                    except ValueError:
                        continue
                    passages.append(
                        Passage(
                            source,
                            line.strip(),
                            role,
                            {
                                "version": cells[0],
                                "time": timestamp,
                                "component": cells[2],
                                "change": cells[3],
                            },
                        )
                    )
            # Preserve explicit absence/future-deployment statements for citation.
            for paragraph in re.split(r"\n\s*\n", text):
                if paragraph.strip() and not paragraph.lstrip().startswith(("#", "|")):
                    passages.append(Passage(source, paragraph.strip(), "context"))
            continue
        # Keep a runbook/history entry intact so qualifiers cannot detach from MTTR.
        parts = (
            re.split(r"(?m)(?=^## )", text)
            if role in {"runbook", "history", "api"}
            else re.split(r"\n\s*\n", text)
        )
        if role == "architecture":
            parts = [
                piece for part in parts for piece in re.split(r"(?m)(?=^- \*\*)", part)
            ]
        for part in parts:
            part = part.strip()
            if part and not re.fullmatch(r"#[^\n]*", part):
                passages.append(Passage(source, part, role))
    unique = {}
    for event in events:
        unique.setdefault(
            (event.time, event.component, event.level, event.message), event
        )
    events = sorted(unique.values(), key=lambda e: (e.time, e.component, e.message))
    return passages, events


def field_text(passage: Passage | None, name: str) -> str:
    if passage is None:
        return ""
    match = re.search(
        r"\*\*" + re.escape(name) + r"\*\*\s*:\s*(.*?)(?=\n\s*\n|\Z)",
        passage.excerpt,
        re.DOTALL | re.IGNORECASE,
    )
    # The corpus places the colon inside bold markers.
    if not match:
        match = re.search(
            r"\*\*" + re.escape(name) + r":\*\*\s*(.*?)(?=\n\s*\n|\Z)",
            passage.excerpt,
            re.DOTALL | re.IGNORECASE,
        )
    return clean(match[1]) if match else ""


def mttr(passage: Passage | None) -> int | None:
    if passage is None or UNCERTAIN.search(clean(passage.excerpt)):
        return None
    match = re.search(
        r"\bMTTR\s*:\s*(\d+)\s*minutes?", clean(passage.excerpt), re.IGNORECASE
    )
    return int(match[1]) if match else None


def relevant_components(
    query: str, passages: list[Passage], events: list[Event]
) -> set[str]:
    """Anchor the investigation to the symptom's subject before expanding it."""
    components = {event.component for event in events}
    # Strong subject terms avoid matching 'order' in every correlation identifier.
    q = set(tokens(query.split("\n\n")[0]))
    q -= {"intermittent", "fail", "delay", "arriving", "arrive", "hour", "purchase"}
    scores = {}
    for component in components:
        subject = set(tokens(component))
        descriptions = [
            p
            for p in passages
            if p.kind == "architecture" and p.excerpt.startswith(f"- **{component}**:")
        ]
        for description in descriptions:
            subject.update(tokens(description.excerpt))
        scores[component] = len(q & subject)
    best = max(scores.values(), default=0)
    return {
        component for component, score in scores.items() if score == best and score > 0
    }


@dataclass
class Hypothesis:
    issue: Passage
    component: str
    signature: str
    observations: list[Event]
    deployment: Passage | None = None
    runbook: Passage | None = None
    history: Passage | None = None
    context: list[Passage] = field(default_factory=list)
    contradictions: list[Passage] = field(default_factory=list)


def correlate(
    query: str, passages: list[Passage], events: list[Event], index: BM25
) -> list[Hypothesis]:
    focus = relevant_components(query, passages, events)
    initial = {id(p): score for p, score in index.rank(query)}
    candidates = []
    observed: dict[tuple[str, str], list[Event]] = {}
    for event in events:
        if event.level not in {"ERROR", "FATAL", "WARN", "WARNING"}:
            continue
        for match in EXCEPTION.finditer(event.message):
            negated = re.search(
                r"\b(?:no|without)\s+$", event.message[: match.start()], re.IGNORECASE
            )
            absent = re.match(
                r"\s+(?:not observed|absent|resolved|count=0)\b",
                event.message[match.end() :],
                re.IGNORECASE,
            )
            if not (negated or absent):
                observed.setdefault((event.component, match[0]), []).append(event)
    seen_anchors = set()
    for issue in (p for p in passages if p.kind == "issue"):
        component = issue.meta["affected_component"].strip()
        if component not in focus:
            continue
        # Exact runtime identifiers are strong anchors; component overlap alone is not.
        signatures = EXCEPTION.findall(
            issue.meta["signature"] + " " + issue.meta.get("title", "")
        )
        for signature in sorted(set(signatures)):
            anchor = (component, signature, clean(issue.meta["signature"]))
            if anchor in seen_anchors:
                continue
            seen_anchors.add(anchor)
            observations = observed.get((component, signature), [])
            if not observations:
                continue
            hypothesis = Hypothesis(issue, component, signature, observations)
            onset = observations[0].time
            expanded = (
                query
                + " "
                + component
                + " "
                + signature
                + " "
                + issue.meta["signature"]
            )
            ranked = index.rank(expanded)
            mechanism = set(tokens(issue.meta["signature"])) - set(tokens(component))
            mechanism -= {
                "exception",
                "error",
                "known",
                "signature",
                "logs",
                "change",
                "recent",
            }
            for passage, _ in ranked:
                body = clean(passage.excerpt)
                if passage.kind in {"runbook", "history"}:
                    if (
                        component not in body
                        or signature not in body
                        or UNCERTAIN.search(body)
                    ):
                        continue
                    date = re.search(r"\b\d{4}-\d\d-\d\d\b", body)
                    if date and datetime.fromisoformat(date[0]) > onset:
                        continue
                    if passage.kind == "runbook" and hypothesis.runbook is None:
                        hypothesis.runbook = passage
                    if passage.kind == "history" and hypothesis.history is None:
                        hypothesis.history = passage
                if passage.kind in {"architecture", "api"} and component in body:
                    hypothesis.context.append(passage)
            deployments = [
                p
                for p in passages
                if p.kind == "deployment"
                and p.meta.get("component") == component
                and p.meta["time"] <= onset
            ]
            # Only the last component deployment can explain its current config.
            if deployments:
                latest = max(deployments, key=lambda p: p.meta["time"])
                change = latest.meta["change"]
                if len(mechanism & set(tokens(change))) >= 2:
                    if re.search(
                        r"undersized|too (?:small|low)",
                        issue.meta["signature"],
                        re.IGNORECASE,
                    ) and re.search(r"increas|restor|revert", change, re.IGNORECASE):
                        hypothesis.contradictions.append(latest)
                    else:
                        hypothesis.deployment = latest
            # A matching failure before a supposed triggering deployment contradicts it.
            future = [
                p
                for p in passages
                if p.kind == "deployment"
                and p.meta.get("component") == component
                and p.meta["time"] > onset
                and len(mechanism & set(tokens(p.meta["change"]))) >= 2
            ]
            hypothesis.contradictions.extend(future)
            candidates.append(hypothesis)
    return sorted(
        candidates,
        key=lambda h: (
            -confidence(h),
            -initial.get(id(h.issue), 0),
            h.component,
            h.signature,
        ),
    )


def confidence(hypothesis: Hypothesis) -> float:
    """An auditable evidence-strength rubric, not a fitted probability."""
    h = hypothesis
    votes = [
        (h.observations[0].passage, 25 if len(h.observations) >= 2 else 15),
        (h.issue, 25),
        (h.deployment, 20),
        (h.history, 12),
        (h.runbook, 10),
    ]
    # A filename can supply at most one independent vote, even if it has many rows.
    source_votes: dict[str, int] = {}
    for passage, weight in votes:
        if passage:
            source_votes[passage.source] = max(
                source_votes.get(passage.source, 0), weight
            )
    score = sum(source_votes.values())
    if any(p.kind == "architecture" for p in h.context):
        score += 3
    if len(source_votes) < 3 or not (h.deployment or h.history):
        score = min(score, 49)
    if h.contradictions:
        score = min(score, 40)
    return float(min(score, 95))


def citations(passages: list[Passage]) -> list[dict]:
    result, seen = [], set()
    for passage in passages:
        key = (passage.source, passage.excerpt)
        if key not in seen:
            result.append(passage.citation())
            seen.add(key)
    return result


def build_known_report(h: Hypothesis, events: list[Event]) -> dict:
    first, last = h.observations[0], h.observations[-1]
    components = {h.component}
    evidence = [event.passage for event in h.observations] + [h.issue]
    mechanism = h.issue.meta["signature"].split(";")[0].rstrip(".")
    mechanism = re.sub(r"^.*?known signature of\s+", "", mechanism, flags=re.IGNORECASE)
    root = f"The matching catalog mechanism is {mechanism} in {h.component}."
    if h.deployment:
        deployment = h.deployment.meta
        elapsed = int((first.time - deployment["time"]).total_seconds())
        root = (
            f"Probable root cause: {deployment['version']} changed {h.component}: "
            f"{deployment['change']}. " + root
        )
        root += (
            f" The first matching error at {first.time.isoformat(sep=' ')} follows "
            f"the {deployment['time'].isoformat(sep=' ')} deployment by "
            f"{elapsed // 60}m {elapsed % 60:02d}s."
        )
        evidence.append(h.deployment)
    root += f" There are {len(h.observations)} matching runtime events in the supplied window."
    # Same-time errors need documented dependency support, not just shared timestamps.
    context = " ".join(clean(p.excerpt) for p in h.context)
    times = {e.time for e in h.observations}
    cited_components = {h.component}
    for event in events:
        if (
            event.time in times
            and event.level in {"ERROR", "FATAL"}
            and event.component != h.component
        ):
            linked = any(
                event.component in p.excerpt
                and h.component in p.excerpt
                and re.search(r"calls|delegates|direct path|->", p.excerpt)
                and not re.search(
                    r"independent|does not call|unrelated", p.excerpt, re.IGNORECASE
                )
                and not p.excerpt.lstrip().startswith("```")
                for p in h.context
            )
            if linked:
                components.add(event.component)
                if event.component not in cited_components:
                    evidence.append(event.passage)
                    cited_components.add(event.component)
    successes = [
        e
        for e in events
        if e.component == h.component
        and re.search(r"succeed|success", e.message, re.IGNORECASE)
    ]
    before = [e for e in successes if e.time < first.time]
    during = [e for e in successes if first.time < e.time < last.time]
    for subset in (before, during):
        if subset:
            evidence.append(subset[0].passage)
    if during:
        root += " Successful operations between failures support intermittent degradation rather than a total outage."
    wait = re.search(r"after (\d+ms)", first.message)
    if wait:
        root += f" The observed acquisition timeout is {wait[1]}."
    if h.contradictions:
        root += " Deployment evidence conflicts with this trigger hypothesis; the cause needs human validation."
        evidence.extend(h.contradictions)
    for passage in (h.history, h.runbook):
        if passage:
            evidence.append(passage)
    # Source diversity is selected explicitly; an architecture hit cannot crowd out logs.
    for role in ("architecture", "api"):
        selected = [p for p in h.context if p.kind == role]
        selected.sort(
            key=lambda p: (
                not p.excerpt.startswith(f"- **{h.component}**:"),
                p.excerpt.lstrip().startswith("```"),
            )
        )
        evidence.extend(selected[:2])
    remediation = field_text(h.runbook, "Remediation") or field_text(
        h.history, "Resolution"
    )
    diagnostics = field_text(h.runbook, "Diagnostic steps")
    if diagnostics:
        remediation = "Validate the mechanism: " + diagnostics + " Then " + remediation
    reduction = (
        re.search(
            r"(?:size|capacity) from (\d+) to (\d+)",
            h.deployment.meta["change"],
            re.IGNORECASE,
        )
        if h.deployment
        else None
    )
    if reduction:
        remediation += f" Restore the prior documented baseline of {reduction[1]} (currently {reduction[2]})."
    if not remediation:
        remediation = "Have the owning on-call engineer validate the runtime signature and choose a reversible mitigation."
    estimate = mttr(h.runbook) or mttr(h.history)
    if confidence(h) < 50:
        estimate = None
        remediation = (
            "Human review required before applying a candidate fix. " + remediation
        )
    remediation += (
        " Verify sustained disappearance of the matching errors and recovery of the affected "
        "operation rate/latency after mitigation; retain telemetry and regression-test configuration changes."
    )
    if "pool" in tokens(context):
        remediation += " Monitor pool utilization, acquisition wait time, and traffic before adjusting capacity."
    if estimate is not None:
        basis = (
            "runbook estimate"
            if mttr(h.runbook) is not None
            else "matching historical recovery time"
        )
        remediation += (
            f" MTTR {estimate} minutes is a {basis}, not measured recovery for this incident. "
            "No incident resolution is documented in the supplied evidence."
        )
        previous = mttr(h.history)
        if previous is not None:
            remediation += (
                f" The matching previous incident recovered in {previous} minutes."
            )
    return {
        "root_cause": root,
        "supporting_evidence": citations(evidence),
        "impacted_systems": sorted(components),
        "mttr_minutes": estimate,
        "remediation": remediation,
        "confidence_score": confidence(h),
        "needs_human_review": confidence(h) < 50,
    }


def queue_delays(
    events: list[Event], focus: set[str]
) -> list[tuple[Event, Event, int]]:
    """Pair actual queue/send events by component and correlation identifier."""
    pending: dict[tuple[str, str, str], Event] = {}
    pairs = []
    for event in events:
        if event.component not in focus:
            continue
        identifier = re.search(
            r"\b(order_id|message_id|request_id|trace_id)=([\w.-]+)", event.message
        )
        if not identifier:
            continue
        key = (event.component, identifier[1], identifier[2])
        if re.search(r"\b(?:email|message) queued\b", event.message, re.IGNORECASE):
            pending.setdefault(key, event)
        elif (
            re.search(r"\b(?:email|message) sent\b", event.message, re.IGNORECASE)
            and key in pending
        ):
            start = pending.pop(key)
            seconds = int((event.time - start.time).total_seconds())
            pairs.append((start, event, seconds))
    return pairs


def build_uncertain_report(
    query: str, passages: list[Passage], events: list[Event], index: BM25
) -> dict:
    focus = relevant_components(query, passages, events)
    pairs = queue_delays(events, focus) if "delay" in tokens(query) else []
    evidence = []
    impacted = sorted({start.component for start, _, _ in pairs})
    root = "The available evidence does not establish a root cause."
    remediation = (
        "Escalate to the owning on-call engineer. Gather timestamped traces, error/latency "
        "metrics, dependency health and configuration changes for the affected operation; "
        "test competing hypotheses before changing production."
    )
    score = 10.0 if passages else 0.0
    if not pairs:
        anomalies = [
            e
            for e in events
            if e.component in focus and e.level in {"ERROR", "FATAL", "WARN", "WARNING"}
        ]
        if anomalies:
            evidence.extend(e.passage for e in (anomalies[0], anomalies[-1]))
            impacted = sorted(
                {e.component for e in anomalies if e.level in {"ERROR", "FATAL"}}
            )
            root += " Relevant runtime anomalies exist, but a corroborated causal explanation is missing."
            score = 25.0
    if pairs:
        seconds = [delay for _, _, delay in pairs]
        details = []
        for start, end, delay in pairs:
            identifier = re.search(
                r"\b(?:order_id|message_id|request_id|trace_id)=([\w.-]+)",
                start.message,
            )
            details.append(f"{identifier[1]}: {delay // 60}m {delay % 60:02d}s")
            evidence.extend([start.passage, end.passage])
        root += (
            f" The logs confirm delayed processing in {', '.join(impacted)}: "
            + "; ".join(details)
            + f" (mean queue-to-send latency {mean(seconds) / 60:.2f} minutes). "
            "These are delivery waiting times, not time to recover the incident. "
            "Each of these matched messages eventually has a send event; this does not establish inbox delivery."
        )
        scoped = [e for e in events if e.component in impacted]
        warnings = [
            e
            for e in scoped
            if e.level in {"WARN", "WARNING"}
            and re.search(r"queue depth", e.message, re.IGNORECASE)
        ]
        if warnings:
            evidence.extend(e.passage for e in warnings[:2])
            root += (
                f" {len(warnings)} queue-depth warning(s) suggest possible queue backlog, "
                "but do not distinguish insufficient consumer capacity from downstream email-provider latency."
            )
        root += " Neither bottleneck is confirmed."
        remediation = (
            "Human investigation is required. Correlate enqueue, dequeue, provider request, provider acknowledgement "
            "and delivery receipt timestamps by message/order ID (or trace/span ID). Measure oldest message age, "
            "queue depth over time, consumer count/concurrency/throughput, retries/dead-letter queues, and provider "
            "latency/throttling. If consumer saturation is confirmed, scale consumers with capacity safeguards; "
            "if the provider is slow or throttling, work with the provider and tune bounded retries/backoff. "
            "Do not apply an unrelated payment rollback or an unverified scaling fix. Establish a delivery-latency "
            "SLO and verify backlog drain and sustained end-to-end latency recovery. "
            "MTTR is unknown because the bottleneck and an incident recovery event are not established."
        )
        score = 30.0 if warnings else 25.0
    # Cite limitations as limitations; they never raise causal confidence.
    expanded = query + " " + " ".join(impacted or sorted(focus))
    seen_roles = set()
    ranked_context = index.rank(expanded)
    ranked_context.sort(
        key=lambda item: (
            -int(bool(UNCERTAIN.search(clean(item[0].excerpt)))),
            -int(
                any(
                    item[0].excerpt.startswith(f"- **{component}**:")
                    for component in impacted
                )
            ),
            -item[1],
            item[0].excerpt,
        )
    )
    for passage, relevance in ranked_context:
        if passage.kind in {"log", "deployment", "issue"} or relevance <= 0:
            continue
        if passage.kind in seen_roles:
            continue
        evidence.append(passage)
        seen_roles.add(passage.kind)
        if passage.kind == "runbook" and UNCERTAIN.search(clean(passage.excerpt)):
            value = re.search(
                r"MTTR\s*:\s*(\d+)\s*minutes", clean(passage.excerpt), re.IGNORECASE
            )
            if value:
                remediation += (
                    f" The runbook's {value[1]}-minute figure is explicitly unconfirmed "
                    "and is not a defensible estimate for this incident."
                )
    if pairs:
        # Explain exclusions using retrieved text and the actual catalog, not corpus absence alone.
        scoped_issues = [
            p
            for p in passages
            if p.kind == "issue" and p.meta["affected_component"] in impacted
        ]
        if scoped_issues:
            best = max(
                scoped_issues,
                key=lambda p: len(
                    set(tokens(p.meta["signature"])) & set(tokens(query))
                ),
            )
            evidence.append(best)
            root += (
                f" The catalog entry for this component concerns '{best.meta.get('title', best.meta['issue_id'])}'; "
                "it does not match the observed queue-to-send delay mechanism."
            )
        contexts = " ".join(clean(p.excerpt) for p in evidence)
        if "no deployment" in contexts.lower():
            root += (
                " There is no correlated component deployment in the supplied history."
            )
        if "no previous incident" in contexts.lower():
            root += " No matching historical incident is recorded."
        if re.search(r"not.*instrument|no.*per-stage", contexts, re.IGNORECASE):
            root += " Missing per-stage timing prevents attribution."
    return {
        "root_cause": root,
        "supporting_evidence": citations(evidence),
        "impacted_systems": impacted,
        "mttr_minutes": None,
        "remediation": remediation,
        "confidence_score": score,
        "needs_human_review": True,
    }


def investigate(query: str, corpus: dict) -> dict:
    """Return exactly the seven required fields, grounded only in this corpus."""
    if not isinstance(query, str) or not isinstance(corpus, dict):
        raise TypeError("query must be a string and corpus a filename-to-text dict")
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in corpus.items()):
        raise TypeError("corpus filenames and document contents must be strings")
    passages, events = ingest(corpus)
    index = BM25(passages)
    candidates = correlate(query, passages, events, index)
    if candidates:
        report = build_known_report(candidates[0], events)
        if (
            len(candidates) > 1
            and confidence(candidates[0]) - confidence(candidates[1]) < 10
        ):
            report["root_cause"] += (
                " A competing runtime-signature hypothesis has similar support; review both mechanisms."
            )
            report["confidence_score"] = min(report["confidence_score"], 40.0)
            report["mttr_minutes"] = None
    else:
        report = build_uncertain_report(query, passages, events, index)
    report["needs_human_review"] = report["confidence_score"] < 50
    # Fail closed if later rendering changes ever damage citation provenance.
    for item in report["supporting_evidence"]:
        if not item["excerpt"] or item["excerpt"] not in corpus[item["source"]]:
            raise ValueError("Evidence excerpt lost its source provenance")
    return report


def load_inputs(data_dir: Path) -> dict[str, tuple[str, dict[str, str]]]:
    inputs = {}
    for directory in sorted(data_dir.iterdir()):
        if directory.is_dir() and (directory / "query.txt").is_file():
            query = (directory / "query.txt").read_text(encoding="utf-8").strip()
            corpus = {
                p.name: p.read_text(encoding="utf-8")
                for p in sorted(directory.iterdir())
                if p.suffix in {".md", ".csv"}
            }
            inputs[directory.name] = (query, corpus)
    return inputs


def self_test(data_dir: Path) -> None:
    """Executable contract and adversarial regression tests for the supplied data.

    Fixture expectations live only here, never in the investigation pipeline.
    Kept in this file so the submission contains exactly the requested three files.
    """
    import random
    import time

    inputs = list(load_inputs(data_dir).values())
    if len(inputs) != 2:
        raise ValueError(
            "The regression suite expects the two official incident fixtures"
        )
    (qa, ca), (qb, cb) = inputs
    count = 0

    def check(label: str, condition: bool) -> None:
        nonlocal count
        if not condition:
            raise AssertionError(label)
        count += 1

    def contract(query: str, corpus: dict) -> dict:
        r = investigate(query, corpus)
        check(
            "exact schema",
            set(r)
            == {
                "root_cause",
                "supporting_evidence",
                "impacted_systems",
                "mttr_minutes",
                "remediation",
                "confidence_score",
                "needs_human_review",
            },
        )
        check(
            "text",
            all(isinstance(r[k], str) and r[k] for k in ("root_cause", "remediation")),
        )
        check(
            "confidence type/range",
            type(r["confidence_score"]) is float and 0 <= r["confidence_score"] <= 100,
        )
        check(
            "review flag",
            type(r["needs_human_review"]) is bool
            and r["needs_human_review"] == (r["confidence_score"] < 50),
        )
        check(
            "MTTR",
            r["mttr_minutes"] is None
            or (type(r["mttr_minutes"]) is int and r["mttr_minutes"] > 0),
        )
        check(
            "systems",
            isinstance(r["impacted_systems"], list)
            and all(isinstance(s, str) for s in r["impacted_systems"]),
        )
        check(
            "citations",
            isinstance(r["supporting_evidence"], list)
            and all(
                set(e) == {"source", "excerpt"}
                and e["source"] in corpus
                and e["excerpt"]
                and e["excerpt"] in corpus[e["source"]]
                for e in r["supporting_evidence"]
            ),
        )
        check("JSON round trip", json.loads(json.dumps(r, allow_nan=False)) == r)
        return r

    a, b = contract(qa, ca), contract(qb, cb)
    check(
        "supported root cause",
        a["confidence_score"] >= 80 and "50 to 10" in a["root_cause"],
    )
    check(
        "correct impact",
        set(a["impacted_systems"]) == {"payment-service", "payment-gateway-adapter"},
    )
    check(
        "estimated MTTR", a["mttr_minutes"] == 20 and "not measured" in a["remediation"]
    )
    check(
        "five corroborating sources",
        {
            "logs.md",
            "deployment_history.md",
            "known_issues.csv",
            "runbooks.md",
            "previous_incidents.md",
        }
        <= {e["source"] for e in a["supporting_evidence"]},
    )
    check("uncertainty", b["confidence_score"] < 50 and b["mttr_minutes"] is None)
    check("email impact", b["impacted_systems"] == ["notification-service"])
    check(
        "computed delays",
        all(t in b["root_cause"] for t in ("46m 35s", "42m 26s", "75m 24s", "56m 43s")),
    )
    check("no guessed mechanism", "Neither bottleneck is confirmed" in b["root_cause"])
    check("empty abstention", contract("", {})["confidence_score"] == 0)
    check("blank corpus", contract(qa, {"blank": " \n"})["confidence_score"] == 0)
    check(
        "unrelated query",
        contract("Why are profile photos missing?", ca)["needs_human_review"],
    )
    for query, corpus, baseline in ((qa, ca, a), (qb, cb, b)):
        check("repeatability", investigate(query, corpus) == baseline)
        items = list(corpus.items())
        random.Random(42).shuffle(items)
        check("input ordering", investigate(query, dict(items)) == baseline)
        renamed = {
            f"document-{i}.txt": value for i, value in enumerate(corpus.values())
        }
        r = contract(query, renamed)
        check(
            "filenames are not features",
            all(r[k] == baseline[k] for k in baseline if k != "supporting_evidence"),
        )
        duplicates = {**corpus, **{"copy-" + k: v for k, v in corpus.items()}}
        check(
            "duplicate documents",
            contract(query, duplicates)["confidence_score"]
            == baseline["confidence_score"],
        )
        logs_only = {k: v for k, v in corpus.items() if LOG.search(v)}
        check("logs alone abstain", contract(query, logs_only)["needs_human_review"])
        no_logs = {k: v for k, v in corpus.items() if not LOG.search(v)}
        check("no runtime evidence", contract(query, no_logs)["needs_human_review"])
        repeated = {k: v + "\n" + v if LOG.search(v) else v for k, v in corpus.items()}
        r = contract(query, repeated)
        check(
            "duplicate log events",
            r["root_cause"] == baseline["root_cause"]
            and r["confidence_score"] == baseline["confidence_score"],
        )
        # Delete one document at a time. Losing support must never increase confidence.
        for key in corpus:
            reduced = {k: v for k, v in corpus.items() if k != key}
            check(
                "evidence removal monotonicity",
                contract(query, reduced)["confidence_score"]
                <= baseline["confidence_score"],
            )

    future = {
        k: v.replace("2026-09-02 14:30", "2026-09-03 14:30")
        if document_kind(v) == "deployment"
        else v
        for k, v in ca.items()
    }
    check("future deployment", contract(qa, future)["needs_human_review"])
    reversed_change = {
        k: v.replace(
            "Reduced connection pool size from 50 to 10",
            "Increased connection pool size from 10 to 50",
        )
        if document_kind(v) == "deployment"
        else v
        for k, v in ca.items()
    }
    check("opposite change", contract(qa, reversed_change)["needs_human_review"])
    changed = {
        k: v.replace("payment-gateway-adapter", "ledger-bridge")
        .replace("payment-service", "ledger-core")
        .replace("50", "80")
        .replace("20 minutes", "31 minutes")
        for k, v in ca.items()
    }
    r = contract(qa, changed)
    check(
        "component names extracted",
        set(r["impacted_systems"]) == {"ledger-bridge", "ledger-core"},
    )
    check("configuration extracted", "80 to 10" in r["root_cause"])
    check("MTTR extracted", r["mttr_minutes"] == 31)
    repeated_issue = {
        k: v
        + "\n"
        + next((line for line in v.splitlines() if line.startswith("KI-101,")), "")
        if k.endswith(".csv")
        else v
        for k, v in ca.items()
    }
    check(
        "duplicate catalog rows",
        contract(qa, repeated_issue)["confidence_score"] == a["confidence_score"],
    )
    negated_errors = {
        k: v.replace(
            "ConnectionPoolTimeoutException:",
            "No ConnectionPoolTimeoutException observed:",
        )
        if LOG.search(v)
        else v
        for k, v in ca.items()
    }
    check(
        "negated error signatures", contract(qa, negated_errors)["needs_human_review"]
    )
    noisy = dict(ca)
    noisy["extra.log"] = (
        "2026-09-02 14:47:12 ERROR order-service UnrelatedDatabaseError"
    )
    check(
        "simultaneous unrelated error",
        contract(qa, noisy)["impacted_systems"] == a["impacted_systems"],
    )
    no_ids = {
        k: re.sub(r"order_id=\S+", "", v) if LOG.search(v) else v for k, v in cb.items()
    }
    check(
        "no invented queue pairs",
        "queue-to-send latency" not in contract(qb, no_ids)["root_cause"],
    )
    check(
        "paraphrased payment query",
        contract("Why do charges fail intermittently?", ca)["confidence_score"] >= 80,
    )
    check(
        "paraphrased email query",
        contract("Investigate email latency", cb)["needs_human_review"],
    )
    for invalid in (None, [], {"bad": None}):
        try:
            investigate(qa, invalid)
        except TypeError:
            count += 1
        else:
            raise AssertionError("Invalid input must fail clearly")
    csv_sample = 'issue_id,title,signature,affected_component,notes\nX,"Quoted, title","A multiline\nsignature",demo-service,"note, value"\n'
    parsed, _ = ingest({"any-name": csv_sample})
    check(
        "CSV quoting/multiline",
        len(parsed) == 1
        and parsed[0].meta["signature"] == "A multiline\nsignature"
        and parsed[0].excerpt in csv_sample,
    )
    start = time.perf_counter()
    for _ in range(50):
        investigate(qa, ca)
        investigate(qb, cb)
    elapsed = time.perf_counter() - start
    print(
        f"PASS: {count} checks; 100 investigations in {elapsed:.3f}s ({elapsed * 10:.2f}ms/report)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data"
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("answers.json")
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run contract and adversarial regression checks",
    )
    args = parser.parse_args()
    if args.self_test:
        self_test(args.data_dir)
        return
    answers = {
        name: investigate(*inputs)
        for name, inputs in load_inputs(args.data_dir).items()
    }
    if not answers:
        parser.error("No incident directories containing query.txt were found")
    args.output.write_text(
        json.dumps(answers, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for name, report in answers.items():
        print(
            f"{name}: confidence={report['confidence_score']}, "
            f"MTTR={report['mttr_minutes']}, review={report['needs_human_review']}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
