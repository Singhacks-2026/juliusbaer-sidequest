"""Run investigate() against both incidents and write answers.json.

Usage (from usecase-2-production-incident-investigator/):

    python run_answers.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUBMISSION = ROOT / "submissions" / "sumit"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SUBMISSION))

from data.loader import load_incident  # noqa: E402
import solution  # noqa: E402

INCIDENTS = [
    "incident_a_pool_exhaustion",
    "incident_b_ambiguous_delay",
]


def main() -> None:
    answers = {}
    for name in INCIDENTS:
        query, corpus = load_incident(name)
        answers[name] = solution.investigate(query, corpus)

    out = SUBMISSION / "answers.json"
    out.write_text(json.dumps(answers, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
