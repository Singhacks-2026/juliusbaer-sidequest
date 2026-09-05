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
from datetime import datetime, timezone
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
    r"^(?P<time>\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:?\d\d)?)\s+"
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
        if matches and document_kind(text) == "context":
            for match in matches:
                try:
                    timestamp = utc_time(match["time"])
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
                        timestamp = utc_time(cells[1])
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
    # Keep wrapped lines but stop at the next field, even without a blank line.
    match = re.search(
        r"(?im)^\s*(?:\*\*)?"
        + re.escape(name)
        + r"(?:\*\*)?\s*:(?:\*\*)?\s*(.*?)"
        + r"(?=\n\s*\n|\n\s*(?:\*\*)?[A-Z][A-Za-z ]+?(?:\*\*)?\s*:|\Z)",
        passage.excerpt,
        re.DOTALL,
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
    components = {event.component for event in events}
    subject = query.strip().split("\n\n")[0]
    q = set(tokens(subject)) - {
        "intermittent",
        "fail",
        "delay",
        "arriving",
        "arrive",
        "hour",
        "purchase",
        "investigate",
        "why",
        "do",
    }
    explicit = {
        c
        for c in components
        if re.search(r"(?<![\w-])" + re.escape(c) + r"(?![\w-])", subject)
    }
    if explicit:
        return explicit
    scores = {}
    for component in components:
        words = set(tokens(component))
        for p in passages:
            if p.kind == "architecture" and p.excerpt.startswith(f"- **{component}**:"):
                # An 'independent of' clause is not evidence of ownership.
                description = re.split(
                    r"independent of|unrelated to|does not call",
                    p.excerpt,
                    flags=re.IGNORECASE,
                )[0]
                words.update(tokens(description))
        lexical = len(q & words)
        event_overlap = max(
            (
                len(q & set(tokens(e.message)))
                for e in events
                if e.component == component
            ),
            default=0,
        )
        scores[component] = max(lexical, event_overlap)
    best = max(scores.values(), default=0)
    return {c for c, score in scores.items() if score == best and score > 0}


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
    followup_changes: list[Passage] = field(default_factory=list)
    query: str = ""


def correlate(
    query: str, passages: list[Passage], events: list[Event], index: BM25
) -> list[Hypothesis]:
    focus = relevant_components(query, passages, events)
    initial = {id(p): score for p, score in index.rank(query)}
    candidates, seen = [], set()
    for issue in (p for p in passages if p.kind == "issue"):
        component = issue.meta["affected_component"].strip()
        if component not in focus:
            continue
        for signature in signature_anchors(issue):
            identity = (component, signature, clean(issue.meta["signature"]))
            if identity in seen:
                continue
            seen.add(identity)
            observed = [
                e
                for e in events
                if e.component == component
                and affirmative(e, signature)
                and anchor_matches(signature, e.message, component)
            ]
            if not observed or not operation_matches(query, issue, observed):
                continue
            h = Hypothesis(issue, component, signature, observed, query=query)
            onset = observed[0].time
            for passage, _ in index.rank(
                query + " " + component + " " + issue.meta["signature"]
            ):
                body = clean(passage.excerpt)
                if passage.kind in {"architecture", "api"} and component in body:
                    h.context.append(passage)
                if passage.kind not in {"runbook", "history"} or component not in body:
                    continue
                if not anchor_matches(signature, body, component) or UNCERTAIN.search(
                    body
                ):
                    continue
                if re.search(
                    r"not (?:the|a) (?:cause|match)|unrelated|ruled out",
                    body,
                    re.IGNORECASE,
                ):
                    continue
                date = re.search(r"\b\d{4}-\d\d-\d\d\b", body)
                if date:
                    try:
                        if utc_time(date[0]) > onset:
                            continue
                    except ValueError:
                        continue
                if not reference_matches(h, passage):
                    continue
                if passage.kind == "history":
                    cause = field_text(passage, "Root cause")
                    if (
                        cause
                        and capacity_deficit(issue.meta["signature"])
                        and not capacity_deficit(cause)
                    ):
                        continue
                    if h.history is None:
                        h.history = passage
                elif h.runbook is None:
                    h.runbook = passage
            changes = sorted(
                (
                    p
                    for p in passages
                    if p.kind == "deployment"
                    and p.meta.get("component") == component
                    and reference_scope(query, p.meta["change"], component)
                ),
                key=lambda p: (p.meta["time"], p.excerpt),
            )
            relevant = [(p, change_status(p, issue)) for p in changes]
            relevant = [(p, status) for p, status in relevant if status != "unrelated"]
            past = [(p, status) for p, status in relevant if p.meta["time"] <= onset]
            if past:
                latest, status = past[-1]
                if status == "support":
                    h.deployment = latest
                else:
                    h.contradictions.append(latest)
            elif relevant:
                # A future change cannot explain already observed failures.
                h.contradictions.extend(p for p, _ in relevant)
            # Once there is an earlier trigger, later changes are mitigation context,
            # not evidence refuting the historical cause. They cannot establish recovery.
            h.followup_changes = (
                [p for p, _ in relevant if p.meta["time"] > onset]
                if h.deployment
                else []
            )
            candidates.append(h)
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
    if len(source_votes) < 3 or not h.deployment:
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


def build_known_report(
    h: Hypothesis, events: list[Event], competing: bool = False
) -> dict:
    first, last = h.observations[0], h.observations[-1]
    score = min(confidence(h), 40.0) if competing else confidence(h)
    decisive = score >= 50
    components = {h.component}
    evidence = [e.passage for e in h.observations] + [h.issue]
    mechanism = h.issue.meta["signature"].split(";")[0].rstrip(".")
    mechanism = re.sub(r"^.*?known signature of\s+", "", mechanism, flags=re.IGNORECASE)
    root = (
        "Probable mechanism" if decisive else "Unconfirmed hypothesis"
    ) + f" in {h.component}: {mechanism}."
    if h.deployment:
        d = h.deployment.meta
        elapsed = int((first.time - d["time"]).total_seconds())
        qualifier = (
            "Probable contributing trigger"
            if decisive
            else "A candidate trigger requiring validation"
        )
        root += f" {qualifier}: {d['version']} changed {h.component}: {d['change']}."
        root += (
            f" The first matching error at {first.time.isoformat(sep=' ')} follows the "
            f"{d['time'].isoformat(sep=' ')} deployment by {elapsed // 60}m {elapsed % 60:02d}s."
        )
        evidence.append(h.deployment)
    root += f" There are {len(h.observations)} affirmative matching runtime events in the supplied window."
    cited = set()
    for e in events:
        if linked_failure(h, e):
            components.add(e.component)
            if e.component not in cited:
                evidence.append(e.passage)
                cited.add(e.component)
    successes = [
        e for e in events if e.component == h.component and successful(e, h.query)
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
        root += " Conflicting or inapplicable change evidence prevents a reliable trigger attribution."
        evidence.extend(h.contradictions)
    if competing:
        root += " Another mechanism has comparable support; human review is needed to resolve the competing hypotheses."
    if not decisive:
        root += " The available evidence is insufficient to establish this as the root cause."
    for p in (h.history, h.runbook):
        if p:
            evidence.append(p)
    for role in ("architecture", "api"):
        matches = [p for p in h.context if p.kind == role]
        matches.sort(
            key=lambda p: (
                not p.excerpt.startswith(f"- **{h.component}**:"),
                p.excerpt.lstrip().startswith("```"),
            )
        )
        evidence.extend(matches[:2])
    estimate = (mttr(h.runbook) or mttr(h.history)) if decisive else None
    diagnostics = field_text(h.runbook, "Diagnostic steps")
    remedy = field_text(h.runbook, "Remediation") or field_text(h.history, "Resolution")
    if decisive:
        remediation = (
            ("Validate the mechanism: " + diagnostics + " ") if diagnostics else ""
        )
        remediation += (
            ("After validation, apply the matching documented mitigation: " + remedy)
            if remedy
            else "Have the owning on-call engineer validate a reversible mitigation."
        )
        if h.deployment:
            delta = configuration_delta(h.deployment.meta["change"])
            if delta and capacity_deficit(h.issue.meta["signature"]):
                remediation += f" The suspected trigger changed the baseline from {delta[0]} to {delta[1]}; confirm current configuration before restoring capacity."
    else:
        remediation = (
            "Human review required. Gather current configuration and utilization metrics and test competing mechanisms. "
            "Do not apply a rollback or the candidate procedure until the operation, resource, and cause are validated."
        )
        if diagnostics:
            remediation += " Relevant diagnostic checks: " + diagnostics
    remediation += " Verify sustained recovery of error rate and latency after any mitigation; retain evidence and regression-test configuration changes."
    if h.followup_changes:
        evidence.extend(h.followup_changes[-2:])
        root += " Later related changes are recorded; they cannot explain onset and do not alone prove recovery."
    if estimate is not None:
        basis = (
            "runbook estimate"
            if mttr(h.runbook) is not None
            else "matching historical recovery time"
        )
        remediation += f" MTTR {estimate} minutes is a {basis}, not measured recovery for this incident."
        prior = mttr(h.history)
        if prior is not None:
            remediation += (
                f" The matching previous incident recovered in {prior} minutes."
            )
        remediation += " The supplied evidence does not establish a complete incident recovery interval."
    return {
        "root_cause": root,
        "supporting_evidence": citations(evidence),
        "impacted_systems": sorted(components),
        "mttr_minutes": estimate,
        "remediation": remediation,
        "confidence_score": score,
        "needs_human_review": score < 50,
    }


def queue_delays(
    events: list[Event], focus: set[str]
) -> list[tuple[Event, Event, int]]:
    """Pair successful terminal events by the strongest shared identifier.

    Reused ambiguous keys are discarded instead of silently choosing a FIFO match.
    Duplicate events have already been removed during ingestion.
    """
    grouped: dict[tuple[str, str, str], dict[str, list[Event]]] = {}
    for event in events:
        if event.component not in focus or event.level != "INFO":
            continue
        if re.search(
            r"(?:sent|queued)\s*[:=]\s*(?:false|0)|\b(?:failed|failure|not sent|not queued|attempt|retry)\b|\b(?:no|never|not)\s+(?:email|message)\s+(?:sent|queued)\b",
            event.message,
            re.IGNORECASE,
        ):
            continue
        stage = (
            "start"
            if re.search(r"\b(?:email|message) queued\b", event.message, re.IGNORECASE)
            else "end"
            if re.search(r"\b(?:email|message) sent\b", event.message, re.IGNORECASE)
            else None
        )
        if stage is None:
            continue
        ids = correlation_ids(event.message)
        for name in ("message_id", "request_id", "trace_id", "order_id"):
            if name in ids:
                key = (event.component, name, ids[name])
                grouped.setdefault(key, {"start": [], "end": []})[stage].append(event)
                break
    pairs = []
    for group in grouped.values():
        if len(group["start"]) != 1 or len(group["end"]) != 1:
            continue
        start, end = group["start"][0], group["end"][0]
        start_ids, end_ids = (
            correlation_ids(start.message),
            correlation_ids(end.message),
        )
        if any(
            start_ids[key] != end_ids[key] for key in start_ids.keys() & end_ids.keys()
        ):
            continue
        if end.time >= start.time:
            pairs.append((start, end, int((end.time - start.time).total_seconds())))
    return sorted(pairs, key=lambda p: (p[0].time, p[0].message))


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
            if e.component in focus
            and affirmative(e, next(iter(EXCEPTION.findall(e.message)), ""))
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
            f" The logs measure queue-to-send processing times in {', '.join(impacted)}: "
            + "; ".join(details)
            + f" (mean queue-to-send latency {mean(seconds) / 60:.2f} minutes). "
            "These are delivery waiting times, not time to recover the incident. Without a documented latency target or baseline, the samples alone do not establish an SLA breach. "
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


def utc_time(value: str) -> datetime:
    """Compare every timestamp in UTC; unzoned challenge timestamps mean UTC."""
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def affirmative(event: Event, anchor: str = "") -> bool:
    """Reject explicit absent/zero/failed observations without negating 'no capacity'."""
    if event.level not in {"ERROR", "FATAL", "WARN", "WARNING"}:
        return False
    message = event.message
    if re.search(
        r"\b(?:count|occurrences|errors)\s*[:=]\s*0\b|\b(?:zero|0) (?:occurrences|errors)\b|"
        r"\b(?:not observed|not detected|not thrown|did not occur|never occurred|"
        r"false alarm|absent|resolved|cleared|suppressed|simulation|dry.run)\b",
        message,
        re.IGNORECASE,
    ):
        return False
    if anchor:
        match = re.search(re.escape(anchor), message, re.IGNORECASE)
        if match and re.search(
            r"\b(?:no|without|never|not)\s+(?:any\s+)?$",
            message[: match.start()],
            re.IGNORECASE,
        ):
            return False
    return True


def correlation_ids(message: str) -> dict[str, str]:
    return dict(
        re.findall(r"\b(message_id|request_id|trace_id|order_id)=([\w.-]+)", message)
    )


def signature_anchors(issue: Passage) -> list[str]:
    text = issue.meta["signature"] + " " + (issue.meta.get("title") or "")
    identifiers = EXCEPTION.findall(text) + re.findall(
        r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b", text
    )
    if identifiers:
        return sorted(set(identifiers))
    phrase = re.search(
        r"^(?:A|An|The)\s+(.+?)\s+in\s+"
        + re.escape(issue.meta["affected_component"])
        + r"\b",
        issue.meta["signature"],
        re.IGNORECASE,
    )
    return [phrase[1] if phrase else issue.meta["signature"].split(";")[0]]


def anchor_matches(anchor: str, text: str, component: str) -> bool:
    if re.search(r"(?<!\w)" + re.escape(anchor) + r"(?!\w)", text, re.IGNORECASE):
        return True
    if EXCEPTION.fullmatch(anchor) or re.fullmatch(
        r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+", anchor
    ):
        return False
    words = (
        set(tokens(anchor))
        - set(tokens(component))
        - {"log", "logs", "known", "signature"}
    )
    shared = words & set(tokens(text))
    return len(words) >= 3 and len(shared) >= 3 and len(shared) / len(words) >= 0.7


def operation_terms(text: str) -> set[str]:
    """Extract operation subjects as well as common domain synonyms.

    Grammar-derived subjects allow unseen operations such as invoice/report to be
    distinguished without adding a service-specific branch for each new case.
    """
    normalized = clean(text)
    terms = set(tokens(normalized)) & {
        "payment",
        "refund",
        "webhook",
        "email",
        "search",
        "checkout",
        "login",
        "order",
    }
    patterns = (
        r"\b([A-Za-z][\w-]*)\s+(?:requests?|processing|fail(?:ed|ing|ures?|s)?|succeeded|queued|sent)\b",
        r"\b(?:while|when)\s+processing\s+([A-Za-z][\w-]*)\b",
        r"\b([A-Za-z][\w-]*)\s+(?:are|is)\s+(?:intermittently\s+)?(?:failing|delayed|late)\b",
    )
    excluded = STOP | {
        "request",
        "intermittent",
        "intermittently",
        "total",
        "any",
        "pool",
        "connection",
        "gateway",
        "http",
        "api",
    }
    for pattern in patterns:
        for match in re.finditer(pattern, normalized, re.IGNORECASE):
            terms.update(w for w in tokens(match[1]) if w not in excluded)
    return terms


def operation_matches(query: str, issue: Passage, observations: list[Event]) -> bool:
    wanted = operation_terms(query.split("\n\n")[0]) - {"order"}
    if not wanted:
        return True
    # Runtime operation takes precedence over a possibly misleading catalog title.
    actual = operation_terms(" ".join(e.message for e in observations)) - {"order"}
    if actual and not (wanted & actual):
        return False
    body = issue.meta["signature"] + "; " + (issue.meta.get("notes") or "")
    return reference_scope(query, body, issue.meta["affected_component"])


def mechanism_terms(issue: Passage) -> set[str]:
    words = set(tokens(issue.meta["signature"])) - set(
        tokens(issue.meta["affected_component"])
    )
    return words - {
        "exception",
        "error",
        "known",
        "signature",
        "logs",
        "change",
        "recent",
        "reference",
        "cross",
        "current",
    }


def capacity_deficit(text: str) -> bool:
    return bool(
        re.search(
            r"undersized|too (?:small|low)|insufficient|under.provision|capacity.*(?:reduc|low)|pool size.*(?:reduc|low)",
            text,
            re.IGNORECASE,
        )
    )


def configuration_delta(text: str) -> tuple[int, int] | None:
    """Read numbers attached to capacity, excluding unrelated retries/timeouts.

    Multiple distinct capacity changes are ambiguous without a structured
    resource identifier, so omit a numeric baseline rather than choose one.
    """
    matches = re.findall(
        r"\b(?:pool(?:\s+(?:size|capacity))?|"
        r"(?:worker|thread|consumer|process)\s+(?:count|capacity))"
        r"\s+from\s+(\d+)\s+to\s+(\d+)\b",
        text,
        re.IGNORECASE,
    )
    deltas = {(int(before), int(after)) for before, after in matches}
    return next(iter(deltas)) if len(deltas) == 1 else None


def change_status(passage: Passage, issue: Passage) -> str:
    change = passage.meta["change"]
    expected, actual = resource_terms(issue.meta["signature"]), resource_terms(change)
    if expected and actual and not expected & actual:
        return "unrelated"
    if len(mechanism_terms(issue) & set(tokens(change))) < 2:
        return "unrelated"
    if re.search(
        r"not (?:applied|deployed|implemented|changed)|did not|unchanged|remains|"
        r"proposed|planned|scheduled|documentation only|dry.run|no (?:change|reduction)",
        change,
        re.IGNORECASE,
    ):
        return "unapplied"
    delta = configuration_delta(change)
    if delta and delta[0] == delta[1]:
        return "unapplied"
    if capacity_deficit(issue.meta["signature"]):
        if delta:
            return "support" if delta[1] < delta[0] else "opposite"
        if re.search(r"increas|restor|revert|rollback", change, re.IGNORECASE):
            return "opposite"
    return "support"


def resource_terms(text: str) -> set[str]:
    """Use explicit resource qualifiers; 'pool' alone cannot join unlike pools."""
    ignored = {
        "connection",
        "the",
        "a",
        "an",
        "bounded",
        "undersized",
        "exhausted",
        "current",
        "new",
        "prior",
        "old",
        "static",
        "insufficient",
        "reduced",
        "increased",
        "configured",
        "available",
        "normal",
        "same",
        "small",
        "large",
    }
    aliases = {
        "https": "http",
        "db": "database",
        "sql": "database",
        "postgres": "database",
        "postgresql": "database",
    }
    qualifiers = re.findall(
        r"\b([A-Za-z][\w-]*)\s+connection\s+pool\b", clean(text), re.IGNORECASE
    )
    qualifiers += re.findall(
        r"\b(worker|thread|consumer|process)\s+pool\b", clean(text), re.IGNORECASE
    )
    words = {word.lower() for word in qualifiers if word.lower() not in ignored}
    return {aliases.get(word, word) for word in words}


def reference_scope(query: str, text: str, component: str) -> bool:
    wanted = operation_terms(query.split("\n\n")[0]) - {"order"}
    body = clean(text).replace(component, "")
    if not wanted:
        return True
    for clause in re.split(r"[.;\n]", body):
        mentioned = operation_terms(clause) - {"order"}
        if wanted & mentioned and re.search(
            r"unrelated|independent|does not affect|not related|remains? healthy|not part of",
            clause,
            re.IGNORECASE,
        ):
            return False
        if (
            re.search(
                r"scope\s*:|applies? only to|while processing|for .*processing",
                clause,
                re.IGNORECASE,
            )
            and mentioned
            and not (wanted & mentioned)
        ):
            return False
    return True


def reference_matches(h: Hypothesis, passage: Passage) -> bool:
    if not reference_scope(h.query, passage.excerpt, h.component):
        return False
    for name in ("Summary", "Symptoms"):
        value = field_text(passage, name).replace(h.component, "")
        operations = operation_terms(value) - {"order"}
        wanted = operation_terms(h.query) - {"order"}
        if operations and wanted and not (operations & wanted):
            return False
    cause = field_text(passage, "Root cause")
    if cause:
        known_resources, resources = (
            resource_terms(h.issue.meta["signature"]),
            resource_terms(cause),
        )
        if known_resources and resources and not (known_resources & resources):
            return False
        if capacity_deficit(h.issue.meta["signature"]) and not capacity_deficit(cause):
            return False
    return True


def direct_dependency(caller: str, target: str, context: list[Passage]) -> bool:
    for passage in context:
        if passage.kind not in {
            "architecture",
            "api",
        } or passage.excerpt.lstrip().startswith("```"):
            continue
        for sentence in re.split(r"[.!?]\s+", clean(passage.excerpt)):
            if re.search(
                r"independent|does not call|unrelated", sentence, re.IGNORECASE
            ):
                continue
            if re.search(
                re.escape(caller)
                + r"\b[^.;]*?\b(?:calls|delegates to|depends on)\s+"
                + re.escape(target)
                + r"\b",
                sentence,
                re.IGNORECASE,
            ):
                return True
    return False


def linked_failure(h: Hypothesis, event: Event) -> bool:
    if event.component == h.component or not affirmative(event):
        return False
    if not direct_dependency(event.component, h.component, h.context):
        return False
    wanted = operation_terms(h.query) - {"order"}
    operation = operation_terms(event.message) - {"order"}
    if wanted and operation and not wanted & operation:
        return False
    ids = correlation_ids(event.message)
    for observation in h.observations:
        other = correlation_ids(observation.message)
        common = ids.keys() & other.keys()
        if any(ids[key] != other[key] for key in common):
            continue
        delta = (event.time - observation.time).total_seconds()
        if common and 0 <= delta <= 30:
            return True
        # Legacy records lack shared IDs: require exact timing and the same
        # documented symptom family, in addition to a direct dependency.
        if (
            not common
            and delta == 0
            and set(tokens(event.message)) & set(tokens(h.signature))
        ):
            return True
    return False


def successful(event: Event, query: str) -> bool:
    if event.level != "INFO" or not re.search(
        r"\b(?:succeeded|successful|success)\b", event.message, re.IGNORECASE
    ):
        return False
    if re.search(
        r"\b(?:no|not|never|failed|failure|unsuccessful)\b|success\s*[:=]\s*(?:false|0)",
        event.message,
        re.IGNORECASE,
    ):
        return False
    wanted, actual = (
        operation_terms(query) - {"order"},
        operation_terms(event.message) - {"order"},
    )
    return not wanted or not actual or bool(wanted & actual)


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
        competing = (
            len(candidates) > 1
            and confidence(candidates[0]) - confidence(candidates[1]) < 10
        )
        report = build_known_report(candidates[0], events, competing)
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
    # Reproduced final-review defects: numeric and operation scope in deployments.
    pool_change = "Reduced connection pool size from 50 to 10"
    for retry_change in (
        "Reduced retry count from 5 to 2",
        "Increased retry count from 2 to 5",
    ):
        mixed_changes = {
            source: text.replace(
                pool_change,
                retry_change + " and reduced connection pool size from 50 to 10",
            )
            if document_kind(text) == "deployment"
            else text
            for source, text in ca.items()
        }
        report = contract(qa, mixed_changes)
        check(
            "capacity numbers belong to the pool, not retry count",
            report["confidence_score"] >= 80
            and "baseline from 50 to 10" in report["remediation"]
            and "baseline from 5 to 2" not in report["remediation"]
            and "baseline from 2 to 5" not in report["remediation"],
        )
    for operation, applies in (("refund", False), ("charge", True)):
        scoped_change = {
            source: text.replace(
                pool_change,
                pool_change + f" for {operation} processing only",
            )
            if document_kind(text) == "deployment"
            else text
            for source, text in ca.items()
        }
        report = contract(qa, scoped_change)
        check(
            "deployment must apply to the investigated operation",
            ("contributing trigger: v2.4.1" in report["root_cause"]) == applies,
        )
    start = time.perf_counter()
    for _ in range(50):
        investigate(qa, ca)
        investigate(qb, cb)
    elapsed = time.perf_counter() - start
    print(
        f"PASS: {count} checks; 100 investigations in {elapsed:.3f}s ({elapsed * 10:.2f}ms/report)."
    )


def run_evaluation(investigator, data_dir, split="all"):
    """Run the frozen 36-case independent challenge evaluation.

    This function owns all test fixtures and labels; production investigation
    never consumes them. ``data_dir`` contains the two public incident folders.
    Counts represent distinct end-to-end cases, not repeated schema asserts.
    Development has 26 cases, held-out has 10. Once inspected, held-out cases
    are regression cases and must not be described as a fresh blind test.
    Labels are public-task and synthetic facts, not private answer-key labels.
    Lexical checks do not establish general accuracy or probabilistic calibration.
    """
    import copy
    import csv
    import hashlib
    import inspect
    import io
    import json
    import math
    import re
    import time
    from pathlib import Path
    from types import SimpleNamespace

    if split not in {"dev", "holdout", "all"}:
        raise ValueError("split must be dev, holdout, or all")

    ROOT = Path(data_dir).resolve().parent
    KEYS = {
        "root_cause",
        "supporting_evidence",
        "impacted_systems",
        "mttr_minutes",
        "remediation",
        "confidence_score",
        "needs_human_review",
    }
    UNKNOWN = re.compile(
        r"not establish|unknown|undetermined|uncertain|insufficient|cannot.{0,30}(?:determin|confirm|establish)|unconfirmed|not confirmed|inconclusive|missing|needs human",
        re.IGNORECASE,
    )

    def official(name):
        directory = ROOT / "data" / name
        return (directory / "query.txt").read_text().strip(), {
            p.name: p.read_text()
            for p in directory.iterdir()
            if p.is_file() and p.name != "query.txt"
        }

    def csv_rows(rows):
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(
            ["issue_id", "title", "signature", "affected_component", "notes"]
        )
        writer.writerows(rows)
        return out.getvalue()

    def fixture(
        service="ledger-adapter",
        caller="billing-service",
        operation="Charge",
        signature="ConnectionPoolTimeoutException",
        version="v9.1.0",
        resource="HTTP connection pool",
        baseline=80,
        current=8,
        recovery=18,
    ):
        """Explicit causal fixture: normal service, config decrease, repeated errors,
        exact catalog mechanism, matching prior incident, and matching runbook."""
        mechanism = f"undersized {resource} relative to current traffic"
        query = f"{operation} requests are intermittently failing in {service} after deployment. What caused this and how should we recover?"
        corpus = {
            "events.log": "# Application logs\n```\n"
            + "\n".join(
                [
                    f"2026-07-18 09:00:00 INFO  {service}  {operation} succeeded request_id=R0",
                    f"2026-07-18 10:00:00 INFO  deploy-agent  Deployment {version} completed on {service}",
                    f"2026-07-18 10:12:00 ERROR {service}  {operation} failed {signature}: no available {resource} connection after 5000ms request_id=R1",
                    f"2026-07-18 10:12:00 ERROR {caller}  {operation} failed reason=GATEWAY_TIMEOUT request_id=R1",
                    f"2026-07-18 10:15:00 INFO  {service}  {operation} succeeded request_id=R2",
                    f"2026-07-18 10:18:00 ERROR {service}  {operation} failed {signature}: no available {resource} connection after 5000ms request_id=R3",
                    f"2026-07-18 10:18:00 ERROR {caller}  {operation} failed reason=GATEWAY_TIMEOUT request_id=R3",
                ]
            )
            + "\n```\n",
            "changes.md": f"# Deployment history\n\n| Version | Timestamp (UTC) | Component | Change |\n|---|---|---|---|\n| {version} | 2026-07-18 10:00 | {service} | Reduced {resource} size from {baseline} to {current} for {operation.lower()} processing |\n",
            "catalog.csv": csv_rows(
                [
                    [
                        "KI-701",
                        f"{operation} {resource} exhausted",
                        f"A {signature} in {service} logs while processing {operation.lower()} is a known signature of an {mechanism}; check recent pool size changes",
                        service,
                        "Verified in matching previous incident",
                    ]
                ]
            ),
            "history.md": f"# Previous incidents\n\n## INC-71 (2026-05-01)\n\n**Summary**: {service} {operation.lower()} failures with {signature}.\n\n**Root cause**: {resource} size was too low for traffic following a pool size reduction.\n\n**Resolution**: restored the {resource} size to {baseline} and redeployed {service}.\n\n**MTTR**: {recovery + 2} minutes.\n",
            "operations.md": f"# Runbooks\n\n## RB-71: {operation} {resource} timeout\n\n**Symptoms**: {service} logs {signature} while processing {operation.lower()} requests.\n\n**Diagnostic steps**: Check {resource} utilization and {operation.lower()} failures.\n\n**Remediation**: Restore the {resource} size to {baseline} and redeploy {service}.\n\n**Typical MTTR: {recovery} minutes.**\n",
            "topology.md": f"# Architecture overview\n\n## Components\n\n- **{caller}**: calls {service} synchronously for every {operation.lower()} request.\n- **{service}**: owns the bounded {resource} used for {operation.lower()} processing.\n- **search-service**: independent search indexing.\n",
            "api.md": f"# API specification\n\n## POST /api/{operation.lower()}\n\nOwned by {caller}, delegates to {service}.\n\n**Timeout**: 5000ms to acquire an {resource} connection for {operation.lower()} processing.\n",
        }
        return query, corpus

    def queue_fixture(service="mail-worker", lines=None):
        query = f"Order confirmation email delivery from {service} is delayed. Investigate the delay."
        lines = lines or [
            f"2026-07-18 10:00:00 INFO  {service}  Email queued message_id=M1",
            f"2026-07-18 10:30:00 INFO  {service}  Email sent message_id=M1",
        ]
        return query, {
            "events.log": "# Application logs\n```\n" + "\n".join(lines) + "\n```\n",
            "topology.md": f"# Architecture overview\n\n## Components\n\n- **{service}**: sends order confirmation emails from a queue to an external email provider.\n- **order-service**: independent of the email delivery path.\n",
            "changes.md": f"# Deployment history\n\nNo deployment touched {service} before this incident.\n",
            "catalog.csv": csv_rows(
                [
                    [
                        "KI-99",
                        "Email rendering issue",
                        "Broken HTML in older mail clients",
                        service,
                        "Cosmetic only; does not affect email delivery timing",
                    ]
                ]
            ),
            "history.md": "# Previous incidents\n\nNo previous incident explains this email delay.\n",
            "operations.md": f"# Runbooks\n\n## RB-9: Email queue warning\n\n**Symptoms**: elevated queue depth in {service}.\n\n**Remediation**: Investigate provider latency and consumer throughput. Scaling consumers is unverified.\n\n**Typical MTTR**: 15 minutes, unconfirmed and may not apply.\n",
        }

    def case(
        name,
        split,
        query,
        corpus,
        *,
        answerable=False,
        root_groups=(),
        low=False,
        minimum=0,
        impacts=(),
        excludes=(),
        mttr=None,
        sources=0,
        forbidden_trigger=None,
        delay_seconds=(),
        forbid_latency=False,
        note="",
    ):
        return {
            "name": name,
            "split": split,
            "query": query,
            "corpus": corpus,
            "expected": {
                "answerable": answerable,
                "root_groups": root_groups,
                "low": low,
                "minimum": minimum,
                "impacts": impacts,
                "excludes": excludes,
                "mttr": mttr,
                "sources": sources,
                "forbidden_trigger": forbidden_trigger,
                "delay_seconds": delay_seconds,
                "forbid_latency": forbid_latency,
            },
            "label_note": note,
        }

    def positives(name, split="dev", **kwargs):
        q, c = fixture(**kwargs)
        service = kwargs.get("service", "ledger-adapter")
        caller = kwargs.get("caller", "billing-service")
        recovery = kwargs.get("recovery", 18)
        return case(
            name,
            split,
            q,
            c,
            answerable=True,
            root_groups=[
                ["pool"],
                ["undersized", "too low", "too small", "reduced", "reduction"],
            ],
            minimum=50,
            impacts=[service, caller],
            excludes=["search-service"],
            mttr=recovery,
            sources=4,
            note="Matching observed failure mechanism, prior trigger, precedent and runbook make the mechanism answerable. MTTR is an estimate, not observed recovery.",
        )

    def build_cases():
        cases = []
        q, c = official("incident_a_pool_exhaustion")
        cases.append(
            case(
                "official_a",
                "dev",
                q,
                c,
                answerable=True,
                root_groups=[["pool"], ["50"], ["10"], ["payment-gateway-adapter"]],
                minimum=75,
                impacts=["payment-gateway-adapter", "payment-service"],
                excludes=[
                    "order-service",
                    "notification-service",
                    "search-service",
                    "web-frontend",
                    "auth-service",
                ],
                mttr=20,
                sources=5,
                note="Public README specifies high confidence, payment adapter, approximately 20-minute estimate, and cross-document evidence.",
            )
        )
        q, c = official("incident_b_ambiguous_delay")
        cases.append(
            case(
                "official_b",
                "dev",
                q,
                c,
                low=True,
                impacts=["notification-service"],
                excludes=[
                    "payment-gateway-adapter",
                    "payment-service",
                    "order-service",
                    "search-service",
                    "web-frontend",
                    "auth-service",
                ],
                mttr=None,
                sources=3,
                note="Public README explicitly requires low confidence and a human flag; no defensible recovery estimate is established.",
            )
        )
        cases.append(positives("renamed_pool"))
        cases.append(
            positives(
                "alternate_exception",
                service="vault-proxy",
                caller="transfer-service",
                signature="PoolAcquireTimeoutError",
                version="v12.7.3",
                recovery=26,
            )
        )
        cases.append(
            positives("nonexception_literal", signature="POOL_CAPACITY_EXHAUSTED")
        )
        cases.append(
            positives(
                "nonexception_phrase",
                service="invoice-connector",
                caller="invoice-service",
                operation="Invoice",
                signature="pool acquire wait exceeded",
                recovery=32,
            )
        )
        x = positives("filenames_reordered")
        x["corpus"] = {
            f"evidence_{i}.txt": v
            for i, v in enumerate(reversed(list(x["corpus"].values())))
        }
        cases.append(x)
        x = positives("csv_quoted_multiline")
        rows = list(csv.reader(io.StringIO(x["corpus"]["catalog.csv"])))
        rows[1][4] = (
            "Verified previously, with two data points.\nSee the prior incident for details."
        )
        out = io.StringIO()
        csv.writer(out).writerows(rows)
        x["corpus"]["catalog.csv"] = out.getvalue()
        cases.append(x)
        x = positives("no_runbook_history_estimate")
        del x["corpus"]["operations.md"]
        x["expected"]["mttr"] = 20
        cases.append(x)
        x = positives("no_recovery_basis")
        del x["corpus"]["operations.md"]
        del x["corpus"]["history.md"]
        x["expected"]["mttr"] = None
        x["expected"]["sources"] = 3
        cases.append(x)
        x = positives("duplicated_sources")
        x["corpus"].update({f"duplicate_{k}": v for k, v in list(x["corpus"].items())})
        cases.append(x)
        q, c = fixture()
        c = {"events.log": c["events.log"]}
        cases.append(
            case(
                "logs_only",
                "dev",
                q,
                c,
                low=True,
                impacts=["ledger-adapter"],
                mttr=None,
                note="Runtime symptoms alone do not establish the catalog mechanism or recovery duration.",
            )
        )
        q, c = fixture()
        c.pop("events.log")
        cases.append(
            case(
                "no_runtime_observation",
                "dev",
                q,
                c,
                low=True,
                mttr=None,
                note="Documents describe a possible issue, but no observation establishes it happened in this incident.",
            )
        )
        cases.append(
            case(
                "empty_corpus",
                "dev",
                "Requests are failing. Why?",
                {},
                low=True,
                mttr=None,
                note="No evidence is available.",
            )
        )
        q, c = fixture()
        c["events.log"] = c["events.log"].replace(
            "Charge failed ConnectionPoolTimeoutException:",
            "No ConnectionPoolTimeoutException observed; Charge healthy:",
        )
        cases.append(
            case(
                "negated_exception",
                "dev",
                q,
                c,
                low=True,
                mttr=None,
                note="Text names the exception while explicitly denying its observation.",
            )
        )
        q, c = fixture()
        c["changes.md"] = c["changes.md"].replace(
            "2026-07-18 10:00", "2026-07-18 11:00"
        )
        cases.append(
            case(
                "deployment_after_errors",
                "dev",
                q,
                c,
                low=True,
                impacts=["ledger-adapter"],
                mttr=None,
                forbidden_trigger="v9.1.0",
                note="The proposed configuration trigger postdates all observed errors.",
            )
        )
        q, c = fixture()
        c["changes.md"] = c["changes.md"].replace(
            "Reduced HTTP connection pool size from 80 to 8",
            "Increased HTTP connection pool size from 8 to 80",
        )
        cases.append(
            case(
                "deployment_opposite_direction",
                "dev",
                q,
                c,
                low=True,
                impacts=["ledger-adapter"],
                mttr=None,
                forbidden_trigger="v9.1.0",
                note="The asserted trigger is contradicted by the documented capacity increase.",
            )
        )
        q, c = fixture()
        c["changes.md"] = c["changes.md"].replace(
            "HTTP connection pool", "database connection pool"
        )
        cases.append(
            case(
                "wrong_resource_deployment",
                "dev",
                q,
                c,
                root_groups=[["HTTP", "pool"]],
                impacts=["ledger-adapter"],
                mttr=[None, 18],
                forbidden_trigger="v9.1.0",
                note="An HTTP acquisition failure does not establish a database pool change as its trigger; the known symptom mechanism may still be reported.",
            )
        )
        q, c = fixture()
        c["events.log"] = c["events.log"].replace(
            "Charge failed ConnectionPoolTimeoutException",
            "Refund failed ConnectionPoolTimeoutException",
        )
        c["events.log"] += (
            "\n2026-07-18 10:12:00 INFO ledger-adapter Charge succeeded request_id=C1\n"
        )
        c["catalog.csv"] = c["catalog.csv"].replace(
            "processing charge", "processing refund"
        )
        c["operations.md"] = c["operations.md"].replace("charge", "refund")
        c["history.md"] = c["history.md"].replace("charge", "refund")
        c["changes.md"] = c["changes.md"].replace("charge", "refund")
        cases.append(
            case(
                "same_service_wrong_operation",
                "dev",
                q,
                c,
                low=True,
                mttr=None,
                note="The only matching service exception is on refund processing, while the question is about charge failures.",
            )
        )
        x = positives("unrelated_same_time_errors")
        x["corpus"]["events.log"] += (
            "\n2026-07-18 10:12:00 ERROR search-service ReindexException: index unavailable\n"
        )
        cases.append(x)
        x = positives("same_service_decoy_issue")
        c = x["corpus"]
        c["catalog.csv"] += (
            "KI-799,Refund callback slowness,Refund webhook delivery delayed,ledger-adapter,Separate from charge processing and does not affect charge failure\n"
        )
        c["events.log"] += (
            "\n2026-07-18 10:10:00 WARN ledger-adapter Refund webhook delivery delayed 480s merchant_id=MER1\n"
        )
        cases.append(x)
        q, c = queue_fixture()
        cases.append(
            case(
                "queue_message_id",
                "dev",
                q,
                c,
                low=True,
                impacts=["mail-worker"],
                excludes=["order-service"],
                mttr=None,
                delay_seconds=[1800],
                note="A same-message queue/send pair establishes a 30-minute observation, not the bottleneck or MTTR.",
            )
        )
        q, c = queue_fixture(
            lines=[
                "2026-07-18 10:00:00 INFO mail-worker Email queued order_id=O1 message_id=M1",
                "2026-07-18 10:05:00 INFO mail-worker Email queued order_id=O1 message_id=M2",
                "2026-07-18 10:25:00 INFO mail-worker Email sent order_id=O1 message_id=M2",
                "2026-07-18 10:30:00 INFO mail-worker Email sent order_id=O1 message_id=M1",
            ]
        )
        cases.append(
            case(
                "queue_shared_order_distinct_messages",
                "dev",
                q,
                c,
                low=True,
                impacts=["mail-worker"],
                mttr=None,
                delay_seconds=[1200, 1800],
                note="Message IDs disambiguate two notifications of one order; order ID alone would invent a 25-minute pair and lose another.",
            )
        )
        q, c = queue_fixture(
            lines=[
                "2026-07-18 10:30:00 INFO mail-worker Email sent message_id=M1",
                "2026-07-18 10:00:00 INFO mail-worker Email queued message_id=M1",
                "2026-07-18 10:00:00 INFO mail-worker Email queued message_id=M1",
                "2026-07-18 10:30:00 INFO mail-worker Email sent message_id=M1",
            ]
        )
        cases.append(
            case(
                "queue_shuffled_duplicates",
                "dev",
                q,
                c,
                low=True,
                impacts=["mail-worker"],
                mttr=None,
                delay_seconds=[1800],
                note="Chronological sorting and replay deduplication preserve one 30-minute pair.",
            )
        )
        q, c = queue_fixture(
            lines=[
                "2026-07-18 10:00:00 INFO mail-worker Email queued message_id=M1",
                "2026-07-18 10:30:00 INFO mail-worker Email sent message_id=M2",
            ]
        )
        cases.append(
            case(
                "queue_mismatched_identifiers",
                "dev",
                q,
                c,
                low=True,
                mttr=None,
                forbid_latency=True,
                note="Different messages cannot be joined into a measured delay.",
            )
        )
        q, c = queue_fixture(
            lines=[
                "2026-07-18 10:00:00 INFO mail-worker Email queued order_id=X1",
                "2026-07-18 10:30:00 INFO mail-worker Email sent message_id=X1",
            ]
        )
        cases.append(
            case(
                "queue_identifier_type_collision",
                "dev",
                q,
                c,
                low=True,
                mttr=None,
                forbid_latency=True,
                note="Equal identifier values in different namespaces are not the same message.",
            )
        )
        # Held-out variations are generated here but are not disclosed before release.
        cases.append(
            positives(
                "holdout_01",
                "holdout",
                service="settlement-bridge",
                caller="settlement-api",
                operation="Settlement",
                signature="ConnectionLeaseExpiredError",
                version="v44.2.8",
                resource="outbound connection pool",
                baseline=120,
                current=12,
                recovery=24,
            )
        )
        cases.append(
            positives(
                "holdout_02",
                "holdout",
                service="receipt-adapter",
                caller="receipt-service",
                operation="Receipt",
                signature="NO_FREE_CONNECTIONS",
                version="v8.6.1",
                resource="provider connection pool",
                recovery=12,
            )
        )
        x = positives(
            "holdout_03",
            "holdout",
            service="refund-bridge",
            caller="refund-api",
            operation="Refund",
            signature="acquisition capacity exhausted",
            resource="refund connection pool",
            recovery=16,
        )
        x["corpus"] = {f"proof-{i}.txt": v for i, v in enumerate(x["corpus"].values())}
        cases.append(x)
        q, c = fixture(
            service="invoice-adapter", caller="invoice-api", operation="Invoice"
        )
        c["changes.md"] = c["changes.md"].replace(
            "2026-07-18 10:00", "2026-07-19 09:00"
        )
        cases.append(
            case(
                "holdout_04",
                "holdout",
                q,
                c,
                low=True,
                mttr=None,
                forbidden_trigger="v9.1.0",
                note="Future deployment cannot cause earlier errors.",
            )
        )
        q, c = fixture()
        c["events.log"] = c["events.log"].replace(
            "ConnectionPoolTimeoutException:",
            "ConnectionPoolTimeoutException count=0; status healthy:",
        )
        cases.append(
            case(
                "holdout_05",
                "holdout",
                q,
                c,
                low=True,
                mttr=None,
                note="A zero counter is not a positive exception observation.",
            )
        )
        q, c = fixture(resource="provider connection pool")
        c["changes.md"] = c["changes.md"].replace(
            "provider connection pool", "database connection pool"
        )
        cases.append(
            case(
                "holdout_06",
                "holdout",
                q,
                c,
                root_groups=[["pool"]],
                mttr=[None, 18],
                forbidden_trigger="v9.1.0",
                note="Different connection-pool resources cannot be conflated.",
            )
        )
        q, c = queue_fixture(
            service="receipt-worker",
            lines=[
                "2026-07-18 12:42:00 INFO receipt-worker Email sent trace_id=T1 message_id=MB",
                "2026-07-18 12:05:00 INFO receipt-worker Email queued trace_id=T1 message_id=MB",
                "2026-07-18 12:53:00 INFO receipt-worker Email sent trace_id=T1 message_id=MA",
                "2026-07-18 12:00:00 INFO receipt-worker Email queued trace_id=T1 message_id=MA",
            ],
        )
        cases.append(
            case(
                "holdout_07",
                "holdout",
                q,
                c,
                low=True,
                impacts=["receipt-worker"],
                mttr=None,
                delay_seconds=[2220, 3180],
                note="Message identity outranks shared trace identity, independent of log order.",
            )
        )
        q, c = queue_fixture(
            service="delivery-worker",
            lines=[
                "2026-07-18 10:00:00 INFO delivery-worker Email queued request_id=R7",
                "2026-07-18 10:11:00 INFO unrelated-worker Email sent request_id=R7",
            ],
        )
        cases.append(
            case(
                "holdout_08",
                "holdout",
                q,
                c,
                low=True,
                mttr=None,
                forbid_latency=True,
                excludes=["unrelated-worker"],
                note="Events from separate components cannot establish this worker queue latency.",
            )
        )
        x = positives(
            "holdout_09",
            "holdout",
            service="clearing-adapter",
            caller="clearing-api",
            operation="Clearing",
            baseline=150,
            current=15,
            recovery=28,
        )
        x["corpus"]["operations.md"] = x["corpus"]["operations.md"].replace(
            "28 minutes.", "28 minutes, unconfirmed and may not apply."
        )
        del x["corpus"]["history.md"]
        x["expected"]["mttr"] = None
        x["expected"]["sources"] = 3
        cases.append(x)
        q, c = fixture(
            service="merchant-bridge", caller="merchant-api", operation="Charge"
        )
        c["events.log"] = c["events.log"].replace(
            "Charge failed ConnectionPoolTimeoutException",
            "Report failed ConnectionPoolTimeoutException",
        )
        c["catalog.csv"] = c["catalog.csv"].replace(
            "processing charge", "processing report"
        )
        c["operations.md"] = c["operations.md"].replace("charge", "report")
        c["history.md"] = c["history.md"].replace("charge", "report")
        c["changes.md"] = c["changes.md"].replace("charge", "report")
        cases.append(
            case(
                "holdout_10",
                "holdout",
                q,
                c,
                low=True,
                mttr=None,
                note="Reporting failure on the same component does not explain charge failures.",
            )
        )
        return cases

    def schema_ok(r):
        if not isinstance(r, dict) or set(r) != KEYS:
            return False
        n = r["confidence_score"]
        return (
            isinstance(r["root_cause"], str)
            and bool(r["root_cause"].strip())
            and isinstance(r["remediation"], str)
            and bool(r["remediation"].strip())
            and isinstance(n, float)
            and math.isfinite(n)
            and 0 <= n <= 100
            and type(r["needs_human_review"]) is bool
            and r["needs_human_review"] == (n < 50)
            and (
                r["mttr_minutes"] is None
                or type(r["mttr_minutes"]) is int
                and r["mttr_minutes"] >= 0
            )
            and isinstance(r["impacted_systems"], list)
            and all(isinstance(s, str) for s in r["impacted_systems"])
            and isinstance(r["supporting_evidence"], list)
            and all(
                isinstance(e, dict)
                and set(e) == {"source", "excerpt"}
                and all(isinstance(v, str) for v in e.values())
                for e in r["supporting_evidence"]
            )
        )

    def causal_trigger(root, version):
        # Inspect the sentence segment before/after the version, preserving dots in versions.
        text = root.replace(version, "VERSIONTOKEN")
        for segment in re.split(r"(?<=[.!?])\s+", text):
            if (
                "VERSIONTOKEN" in segment
                and re.search(
                    r"cause|caus|trigger|changed|responsib", segment, re.IGNORECASE
                )
                and not re.search(
                    r"not|contradict|post.?dat|after.*(?:error|onset)|unrelated|reject|unsupported|cannot|no evidence|conflict",
                    segment,
                    re.IGNORECASE,
                )
            ):
                return True
        return False

    def delay_mentioned(root, seconds):
        minutes, remaining = divmod(seconds, 60)
        patterns = [
            rf"\b{minutes}\s*m\s*{remaining:02d}\s*s\b",
            rf"\b{minutes}\s*min(?:utes)?\s*(?:and\s*)?{remaining}\s*(?:sec|s)",
            rf"\b{seconds}\s*(?:seconds|s)\b",
        ]
        if remaining == 0:
            patterns.append(rf"\b{minutes}(?:\.0+)?\s*(?:minutes|min)\b")
        return any(re.search(p, root, re.IGNORECASE) for p in patterns)

    def evaluate_case(module, c):
        began = time.perf_counter()
        try:
            report = module.investigate(c["query"], copy.deepcopy(c["corpus"]))
            json.dumps(report, allow_nan=False)
        except Exception as exc:  # noqa: BLE001 - Record arbitrary failures of the implementation under evaluation.
            return {
                "name": c["name"],
                "split": c["split"],
                "crash": f"{type(exc).__name__}: {exc}",
                "passed": False,
                "dimensions": {},
                "elapsed_ms": round((time.perf_counter() - began) * 1000, 3),
                "expected": c["expected"],
                "label_note": c["label_note"],
            }
        ex = c["expected"]
        root = str(report.get("root_cause", ""))
        score = report.get("confidence_score", 0)
        score = score if isinstance(score, (int, float)) else 0
        evidence = report.get("supporting_evidence", [])
        evidence = evidence if isinstance(evidence, list) else []
        provenance = all(
            isinstance(e, dict)
            and isinstance(e.get("excerpt"), str)
            and bool(e["excerpt"])
            and e.get("source") in c["corpus"]
            and e["excerpt"] in c["corpus"][e["source"]]
            for e in evidence
        )
        source_count = len({e.get("source") for e in evidence if isinstance(e, dict)})
        cause = all(
            any(term.lower() in root.lower() for term in group)
            for group in ex["root_groups"]
        )
        if ex["low"]:
            cause = cause and bool(UNKNOWN.search(root))
        if ex["forbidden_trigger"]:
            cause = cause and not causal_trigger(root, ex["forbidden_trigger"])
        impacts = set(report.get("impacted_systems", []))
        impact = set(ex["impacts"]) <= impacts and not (set(ex["excludes"]) & impacts)
        mttr = (
            report.get("mttr_minutes") in ex["mttr"]
            if isinstance(ex["mttr"], list)
            else report.get("mttr_minutes") == ex["mttr"]
        )
        calibration = score >= ex["minimum"] and (not ex["low"] or score < 50)
        delay = all(delay_mentioned(root, seconds) for seconds in ex["delay_seconds"])
        if ex["forbid_latency"]:
            delay = delay and not bool(
                re.search(
                    r"confirm(?:ed|s)? delayed processing|mean queue.to.send latency|\b\d+m \d+s",
                    root,
                    re.IGNORECASE,
                )
            )
        dims = {
            "schema": schema_ok(report),
            "citation_provenance": provenance,
            "cause": cause,
            "impact": impact,
            "mttr": mttr,
            "confidence": calibration,
            "source_diversity": source_count >= ex["sources"],
            "queue_observation": delay,
        }
        return {
            "name": c["name"],
            "split": c["split"],
            "crash": None,
            "passed": all(dims.values()),
            "dimensions": dims,
            "high_confidence_false_claim": score >= 70
            and (
                not cause
                or not impact
                or not mttr
                or not calibration
                or not provenance
                or not delay
            ),
            "answerable": ex["answerable"],
            "answered": score >= 50 and cause,
            "citation_count": len(evidence),
            "elapsed_ms": round((time.perf_counter() - began) * 1000, 3),
            "report": report,
            "expected": ex,
            "label_note": c["label_note"],
        }

    def summarize(results):
        total = len(results)
        crashes = sum(bool(r["crash"]) for r in results)
        dimensions = {
            k: sum(r.get("dimensions", {}).get(k, False) for r in results)
            for k in [
                "schema",
                "citation_provenance",
                "cause",
                "impact",
                "mttr",
                "confidence",
                "source_diversity",
                "queue_observation",
            ]
        }
        answerable = [r for r in results if r.get("answerable")]
        return {
            "distinct_cases": total,
            "passed_cases": sum(r["passed"] for r in results),
            "crashes": crashes,
            "crash_rate": round(crashes / total, 4),
            "dimension_pass_counts": dimensions,
            "high_confidence_false_claims": sum(
                r.get("high_confidence_false_claim", False) for r in results
            ),
            "answerable_cases": len(answerable),
            "correct_answers": sum(r["answered"] for r in answerable),
            "answer_coverage": round(
                sum(r["answered"] for r in answerable) / len(answerable), 4
            )
            if answerable
            else None,
            "total_citations": sum(r.get("citation_count", 0) for r in results),
            "elapsed_ms": round(sum(r["elapsed_ms"] for r in results), 3),
        }

    all_cases = build_cases()
    cases = [c for c in all_cases if split == "all" or c["split"] == split]
    fixture_hash = hashlib.sha256(
        json.dumps(all_cases, sort_keys=True).encode()
    ).hexdigest()
    module = SimpleNamespace(investigate=investigator)
    results = [evaluate_case(module, c) for c in cases]
    source_file = inspect.getsourcefile(investigator)
    solution_hash = (
        hashlib.sha256(Path(source_file).read_bytes()).hexdigest()
        if source_file
        else None
    )
    return {
        "evaluation_version": 1,
        "fixture_sha256": fixture_hash,
        "solution_sha256": solution_hash,
        "split": split,
        "summary": summarize(results),
        "results": results,
    }


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
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run 36 independently authored semantic scenarios",
    )
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="all")
    parser.add_argument(
        "--evaluation-output",
        type=Path,
        help="Save per-scenario reports and metrics as JSON",
    )
    parser.add_argument(
        "--evaluate-against",
        type=Path,
        help="Evaluate another trusted solution.py for comparison",
    )
    args = parser.parse_args()
    if args.evaluate:
        implementation = investigate
        if args.evaluate_against:
            import importlib.util
            import sys

            spec = importlib.util.spec_from_file_location(
                "evaluated_submission", args.evaluate_against
            )
            if spec is None or spec.loader is None:
                parser.error("Cannot load comparison implementation")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            implementation = module.investigate
        evaluation = run_evaluation(implementation, args.data_dir, args.split)
        if args.evaluation_output:
            args.evaluation_output.write_text(
                json.dumps(evaluation, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(evaluation["summary"], indent=2))
        if (
            evaluation["summary"]["passed_cases"]
            != evaluation["summary"]["distinct_cases"]
        ):
            raise SystemExit(1)
        return

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
