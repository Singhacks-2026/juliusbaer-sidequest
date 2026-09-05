"""Validate submission structure, source facts, and available execution traces."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from agent.agent import TOOLS
from tools.client_tools import get_client_profile
from tools.payment_tools import get_payment

BASE = Path(__file__).resolve().parent


def validate(results, questions, trace_directory=None) -> list[str]:
    errors = []
    if not isinstance(results, list):
        return ["The output must be a JSON array."]
    if len(results) != len(questions):
        errors.append(f"Expected {len(questions)} results; found {len(results)}.")
    expected = {item["question_id"]: item for item in questions}
    seen = set()
    required = {"question_id", "payment_id", "answer", "citations", "facts", "tools_used"}
    for index, result in enumerate(results):
        label = f"Result {index + 1}"
        if not isinstance(result, dict) or not required <= result.keys():
            errors.append(f"{label}: missing required fields.")
            continue
        qid = result["question_id"]
        if not isinstance(qid, str) or qid not in expected or qid in seen:
            errors.append(f"{label}: unknown or duplicate question_id.")
            continue
        seen.add(qid)
        question = expected[qid]
        if result["payment_id"] != question["payment_id"]:
            errors.append(f"{qid}: payment_id does not match the official question.")
        if "question" in result and result["question"] != question["question"]:
            errors.append(f"{qid}: question text was changed.")
        if not isinstance(result["answer"], str) or not result["answer"].strip():
            errors.append(f"{qid}: answer must be a nonempty string.")
        for name in ("citations", "tools_used"):
            values = result[name]
            if not isinstance(values, list) or not values or not all(isinstance(v, str) for v in values):
                errors.append(f"{qid}: {name} must be a nonempty list of strings.")
                continue
            if name == "citations":
                for source in values:
                    if Path(source).name != source or not (BASE / "data/policies" / source).is_file():
                        errors.append(f"{qid}: unknown citation {source}.")
            elif set(values) - set(TOOLS):
                errors.append(f"{qid}: tools_used names an unknown tool.")
        facts = result["facts"]
        if not isinstance(facts, dict):
            errors.append(f"{qid}: facts must be an object.")
            continue
        payment = get_payment(question["payment_id"])
        client = get_client_profile(payment["client_id"])
        for key in ("amount", "currency", "client_id", "beneficiary_country_code"):
            if facts.get(key) != payment[key]:
                errors.append(f"{qid}: facts.{key} differs from the source CSV.")
        if facts.get("client_country") != client["country"]:
            errors.append(f"{qid}: client_country differs from the client CSV.")
        if trace_directory:
            digest = hashlib.sha256((question["payment_id"] + question["question"]).encode()).hexdigest()[:16]
            path = Path(trace_directory) / f"{digest}.json"
            if not path.is_file():
                errors.append(f"{qid}: execution trace is missing.")
            else:
                trace = json.loads(path.read_text(encoding="utf-8"))
                invoked = list(dict.fromkeys(call["tool"] for call in trace["tool_calls"] if call["invoked"]))
                if invoked != result["tools_used"]:
                    errors.append(f"{qid}: tools_used does not match the actual trace.")
                for key in ("answer", "citations", "facts", "tools_used"):
                    if result[key] != (trace.get("result") or {}).get(key):
                        errors.append(f"{qid}: {key} differs from the traced result.")
    if set(expected) - seen:
        errors.append("Missing questions: " + ", ".join(sorted(set(expected) - seen)))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", default="submission.json")
    parser.add_argument("--questions", default=str(BASE / "questions/questions.json"))
    parser.add_argument("--traces", action="store_true", help="Also verify every result against local execution traces.")
    args = parser.parse_args()
    try:
        results = json.loads(Path(args.submission).read_text(encoding="utf-8"))
        questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
        directory = None
        if args.traces:
            from dotenv import load_dotenv
            load_dotenv(BASE / ".env")
            directory = Path(os.getenv("LLM_TRACE_DIR", "artifacts/traces"))
            if not directory.is_absolute():
                directory = BASE / directory
        errors = validate(results, questions, directory)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Could not validate the submission: {error}") from None
    if errors:
        raise SystemExit("Validation failed:\n" + "\n".join(errors))
    print(f"Validated {len(results)} answers: required fields, official IDs, source facts" +
          (", and execution traces." if args.traces else "."))
    print("Review answer wording against the policies; this is not the organiser's private scoring system.")


if __name__ == "__main__":
    main()
