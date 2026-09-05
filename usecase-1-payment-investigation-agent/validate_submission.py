"""Validate submission.json before handoff to the organizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED = {"question_id", "payment_id", "answer", "citations", "facts", "tools_used"}


def validate(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 10:
        raise ValueError("Submission must be a JSON list containing exactly 10 results")
    expected_ids = {f"Q{number:02d}" for number in range(1, 11)}
    actual_ids = {item.get("question_id") for item in data if isinstance(item, dict)}
    if actual_ids != expected_ids:
        raise ValueError(f"Question IDs differ: expected {sorted(expected_ids)}, got {sorted(actual_ids)}")
    for position, item in enumerate(data, start=1):
        missing = REQUIRED - item.keys()
        if missing:
            raise ValueError(f"Result {position} is missing: {sorted(missing)}")
        if not isinstance(item["answer"], str) or not item["answer"].strip():
            raise ValueError(f"Result {position} has an empty answer")
        if not isinstance(item["citations"], list) or any(
            str(source).startswith("decoy_") for source in item["citations"]
        ):
            raise ValueError(f"Result {position} has invalid citations")
        if not isinstance(item["facts"], dict) or not isinstance(item["tools_used"], list):
            raise ValueError(f"Result {position} has invalid facts/tools_used types")
    print(f"Valid submission: {path} (10 complete results, no decoy citations)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="submission.json")
    validate(Path(parser.parse_args().path))
