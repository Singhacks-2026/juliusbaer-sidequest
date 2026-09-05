"""Adversarial LLM review of the deterministic investigator — a QA harness.

NOT part of the graded pipeline. `solution.py` has no network dependency and
produces `answers.json` on its own; this script never writes it. Its only job is
to attack that output hard enough to find defects a self-review would miss.

The design principle is the same one that governs `solution.py`: a language
model is not a source of truth. So the model is used as a *critic*, never as an
author, and every criticism it returns is then verified deterministically
against the corpus before a human is allowed to see it:

  * a "missed evidence" excerpt is discarded unless it is a verbatim substring
    of the file the model attributes it to — this is what catches a fabricated
    quote;
  * an "unsupported claim" is discarded unless the claim text it quotes really
    does appear in the report under review — this is what catches the model
    inventing a claim to object to.

Findings that survive both gates are the ones worth acting on. Findings that do
not are reported too, as a measure of how much the critic confabulated.

Usage:
    export GEMINI_API_KEY=...        # never commit this
    python3 submissions/victor-gaya/llm_review.py

Run from the use-case directory (the one containing `data/`).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
USE_CASE_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
for path in (USE_CASE_DIR, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from data.loader import load_incident  # noqa: E402
import solution  # noqa: E402

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
INCIDENTS = ("incident_a_pool_exhaustion", "incident_b_ambiguous_delay")
TIMEOUT_SECONDS = 90

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "unsupported_claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "quoted_claim": {"type": "string"},
                    "why_unsupported": {"type": "string"},
                },
                "required": ["quoted_claim", "why_unsupported"],
            },
        },
        "missed_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "excerpt": {"type": "string"},
                    "why_it_matters": {"type": "string"},
                },
                "required": ["source", "excerpt", "why_it_matters"],
            },
        },
        "confidence_assessment": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["too_high", "about_right", "too_low"],
                },
                "reasoning": {"type": "string"},
            },
            "required": ["verdict", "reasoning"],
        },
        "worst_flaw": {"type": "string"},
    },
    "required": [
        "unsupported_claims",
        "missed_evidence",
        "confidence_assessment",
        "worst_flaw",
    ],
}

PROMPT = """You are a hostile reviewer of an automated incident-investigation \
report. Your job is to find what is WRONG with it. A reviewer who says the \
report looks fine has failed.

You are given the complete document corpus for one production incident, the \
plain-English query, and the structured report a deterministic pipeline \
produced from them.

Attack it on four fronts:

1. UNSUPPORTED CLAIMS. Quote, verbatim from the report, any statement that the \
corpus does not actually support — an asserted cause the documents never link, \
an exclusivity claim ("the only change") the documents never make, a component \
blamed without basis, a number that does not appear in any document.

2. MISSED EVIDENCE. Identify passages in the corpus that a competent \
investigator would have cited and this report did not. Quote each one VERBATIM \
from the file it appears in and name that file exactly. Do not paraphrase: an \
excerpt that is not a character-for-character substring of the named file will \
be discarded and counted against you.

3. CONFIDENCE. Is the confidence score defensible given the evidence actually \
present? Consider whether independent sources genuinely corroborate each other, \
whether any corroborating source disclaims itself, and whether the timeline \
supports the causal story. Over-confidence and under-confidence are both errors.

4. WORST FLAW. In one or two sentences: the single most damaging problem with \
this report.

Report only defects you can point at in the supplied text. If a category is \
genuinely empty, return an empty list for it rather than inventing something.

=== QUERY ===
{query}

=== CORPUS ===
{corpus}

=== REPORT UNDER REVIEW ===
{report}
"""


def _render_corpus(corpus: Dict[str, str]) -> str:
    return "\n\n".join(
        f"----- FILE: {name} -----\n{text}" for name, text in sorted(corpus.items())
    )


def call_gemini(prompt: str, api_key: str, attempts: int = 4) -> Dict[str, Any]:
    """One structured-output call, retried on transient upstream failures.

    503s and 429s are routine on a shared endpoint and say nothing about the
    review; a transport failure should not be mistaken for "the critic found
    nothing".
    """
    delay = 4.0
    last: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            return _call_once(prompt, api_key)
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise
            print(f"    upstream {exc.code}; retrying in {delay:.0f}s "
                  f"({attempt + 1}/{attempts - 1})")
            time.sleep(delay)
            delay *= 2
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"exhausted retries: {last}")


def _call_once(prompt: str, api_key: str) -> Dict[str, Any]:
    """One structured-output call. Raises on transport or protocol failure."""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json",
            "responseSchema": REVIEW_SCHEMA,
        },
    }
    request = urllib.request.Request(
        f"{ENDPOINT.format(model=MODEL)}?key={api_key}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        body = json.loads(response.read().decode("utf-8"))

    candidates = body.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"no candidates returned: {json.dumps(body)[:400]}")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts).strip()
    if not text:
        raise RuntimeError(
            f"empty completion (finishReason={candidates[0].get('finishReason')})"
        )
    return json.loads(text)


# --------------------------------------------------------------------------- #
# Deterministic verification of the critic                                     #
# --------------------------------------------------------------------------- #
def _normalize(text: str) -> str:
    """Whitespace-insensitive form, so a quote that only differs by line
    wrapping still counts as verbatim."""
    return " ".join(text.split())


def verify_missed_evidence(
    items: List[Dict[str, str]], corpus: Dict[str, str]
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Keep only excerpts that genuinely occur in the file they are attributed
    to. Everything else is a fabricated quote."""
    accepted, rejected = [], []
    normalized = {name: _normalize(text) for name, text in corpus.items()}
    for item in items:
        source = item.get("source", "")
        excerpt = _normalize(item.get("excerpt", ""))
        if source in normalized and excerpt and excerpt in normalized[source]:
            accepted.append(item)
        else:
            reason = (
                "unknown source file" if source not in normalized
                else "excerpt is not a verbatim substring of the cited file"
            )
            rejected.append({**item, "rejected_because": reason})
    return accepted, rejected


def verify_unsupported_claims(
    items: List[Dict[str, str]], report: Dict[str, Any]
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Keep only objections to text the report actually contains, so the critic
    cannot object to a claim it invented."""
    accepted, rejected = [], []
    haystack = _normalize(json.dumps(report, ensure_ascii=False)).lower()
    for item in items:
        claim = _normalize(item.get("quoted_claim", "")).lower()
        # allow a partial quote: a long claim rarely round-trips exactly
        probe = claim[:120]
        if probe and probe in haystack:
            accepted.append(item)
        else:
            rejected.append(
                {**item, "rejected_because": "quoted claim does not appear in the report"}
            )
    return accepted, rejected


def review_incident(name: str, api_key: str) -> Dict[str, Any]:
    query, corpus = load_incident(name)
    report = solution.investigate(query, corpus)
    raw = call_gemini(
        PROMPT.format(
            query=query,
            corpus=_render_corpus(corpus),
            report=json.dumps(report, indent=2),
        ),
        api_key,
    )
    good_evidence, bad_evidence = verify_missed_evidence(
        raw.get("missed_evidence", []), corpus
    )
    good_claims, bad_claims = verify_unsupported_claims(
        raw.get("unsupported_claims", []), report
    )
    return {
        "incident": name,
        "report_confidence": report["confidence_score"],
        "verified_unsupported_claims": good_claims,
        "discarded_unsupported_claims": bad_claims,
        "verified_missed_evidence": good_evidence,
        "discarded_missed_evidence": bad_evidence,
        "confidence_assessment": raw.get("confidence_assessment", {}),
        "worst_flaw": raw.get("worst_flaw", ""),
    }


def main() -> int:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("GEMINI_API_KEY is not set. This harness is optional; "
              "solution.py does not need it.", file=sys.stderr)
        return 2

    results = []
    for name in INCIDENTS:
        print(f"\n{'=' * 74}\nADVERSARIAL REVIEW — {name}\n{'=' * 74}")
        try:
            result = review_incident(name, api_key)
        except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError,
                json.JSONDecodeError) as exc:
            print(f"  review unavailable ({type(exc).__name__}: {exc})")
            continue
        results.append(result)

        print(f"  deterministic confidence: {result['report_confidence']}")
        assessment = result["confidence_assessment"]
        print(f"  critic's verdict on it:   {assessment.get('verdict')}")
        print(f"    {assessment.get('reasoning', '')}")
        print(f"\n  WORST FLAW (critic): {result['worst_flaw']}")

        print(f"\n  VERIFIED unsupported claims: {len(result['verified_unsupported_claims'])}"
              f"  (discarded as fabricated: {len(result['discarded_unsupported_claims'])})")
        for item in result["verified_unsupported_claims"]:
            print(f"    - \"{item['quoted_claim'][:150]}\"")
            print(f"      -> {item['why_unsupported'][:300]}")

        print(f"\n  VERIFIED missed evidence: {len(result['verified_missed_evidence'])}"
              f"  (discarded as fabricated: {len(result['discarded_missed_evidence'])})")
        for item in result["verified_missed_evidence"]:
            print(f"    - [{item['source']}] {item['excerpt'][:150]}")
            print(f"      -> {item['why_it_matters'][:300]}")
        for item in result["discarded_missed_evidence"]:
            print(f"    x [{item.get('source')}] REJECTED: {item['rejected_because']}")
            print(f"      claimed: {item.get('excerpt','')[:120]}")

    out = os.path.join(HERE, "llm_review_findings.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
