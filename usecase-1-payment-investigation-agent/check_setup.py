"""Check local setup; optionally investigate the first real question."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
from urllib.parse import urlsplit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Make real LLM calls for the first official question.")
    args = parser.parse_args()
    missing = [name for name in ("dotenv", "jsonschema", "openai", "anthropic")
               if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit("Missing packages: " + ", ".join(missing) +
                         ". Run: python -m pip install -r requirements.txt")
    from agent.providers import ConfigurationError, read_settings
    try:
        settings = read_settings()
    except ConfigurationError as error:
        raise SystemExit(str(error)) from None
    print(f"Provider: {settings.provider}; API: {settings.api}; model: {settings.model}")
    endpoint = os.getenv("OPENAI_BASE_URL") if settings.provider == "openai" else None
    if endpoint:
        print(f"Custom endpoint host: {urlsplit(endpoint).hostname}")
    from tools.data_access import read_rows
    from tools.policy_tools import search_policy
    print(f"Data: {len(read_rows('clients.csv'))} clients, {len(read_rows('payments.csv'))} payments.")
    print(f"Retrieval: {len(search_policy('global payment threshold'))} relevant passages found.")
    if args.live:
        from agent.agent import run_agent
        base = Path(__file__).resolve().parent
        question = json.loads((base / "questions/questions.json").read_text(encoding="utf-8"))[0]
        result = run_agent(question["question"], question["payment_id"])
        print("Live investigation succeeded.")
        print(result["answer"])
    else:
        print("Local configuration is ready. This check did not contact the LLM provider.")


if __name__ == "__main__":
    main()
