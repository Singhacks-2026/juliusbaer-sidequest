"""
LLM adapter.

Targets the OpenAI chat-completions API with tool calling, defaulting to
``gpt-4o``.  Configuration comes from the environment, so the same code runs
against OpenAI, Azure-style deployments, or any OpenAI-compatible endpoint via
``OPENAI_BASE_URL``.

Two behaviours matter for how this submission is evaluated.  The organizer
re-runs ``main.py`` in a fresh environment, possibly with their own key and
model, or with no key at all; and crashing on any official question is a
disqualifier.  So ``is_configured()`` is checked before any network call and
every request is retried then allowed to fail soft, letting the agent fall back
to deterministic synthesis instead of aborting the run.
"""

import json
import os
import time

DEFAULT_MODEL = "gpt-4o"
_MAX_ATTEMPTS = 3
_TIMEOUT_SECONDS = 60.0

_client = None
_client_initialised = False


def load_dotenv(path: str = ".env") -> None:
    """
    Load ``KEY=VALUE`` pairs from a .env file without adding a dependency.

    Existing environment variables win, so an organizer's exported key is never
    overridden by a committed file.
    """
    candidates = [
        path,
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path
        ),
    ]

    for candidate in candidates:
        if not os.path.isfile(candidate):
            continue

        with open(candidate, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")

                if key and key not in os.environ:
                    os.environ[key] = value
        return


load_dotenv()


def model_name() -> str:
    return os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)


def is_configured() -> bool:
    """Whether an API key and a usable SDK are both present."""
    if not os.environ.get("OPENAI_API_KEY"):
        return False
    return _get_client() is not None


def _get_client():
    global _client, _client_initialised

    if _client_initialised:
        return _client

    _client_initialised = True

    try:
        from openai import OpenAI
    except ImportError:
        _client = None
        return _client

    kwargs = {
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "timeout": _TIMEOUT_SECONDS,
    }
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url

    try:
        _client = OpenAI(**kwargs)
    except Exception:  # noqa: BLE001 - misconfiguration must not abort the run
        _client = None

    return _client


def chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float = 0.0,
) -> dict | None:
    """
    Send one chat-completion request.

    Returns a plain dict with ``content`` and ``tool_calls``, or ``None`` if the
    call could not be completed.  The message is rebuilt by hand rather than
    dumped from the SDK model so the shape stays stable across SDK versions.
    """
    client = _get_client()
    if client is None:
        return None

    request = {
        "model": model_name(),
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        request["tools"] = tools
        request["tool_choice"] = "auto"

    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = client.chat.completions.create(**request)
        except Exception as error:  # noqa: BLE001 - retry, then fail soft
            if not _is_retryable(error):
                print(f"  [llm] not retryable, falling back: {_summarise(error)}")
                return None

            if attempt == _MAX_ATTEMPTS - 1:
                print(f"  [llm] giving up after {_MAX_ATTEMPTS} attempts: {_summarise(error)}")
                return None

            time.sleep(2**attempt)
            continue

        return _normalise(response.choices[0].message)

    return None


def _is_retryable(error: Exception) -> bool:
    """
    Whether retrying could plausibly succeed.

    A bad key, a revoked key or an unknown model will fail identically every
    time, so retrying only delays the fallback.  Rate limits, timeouts and
    server errors are worth another attempt.
    """
    status = getattr(error, "status_code", None)
    if status is None:
        return True  # connection error or timeout
    return status not in (400, 401, 403, 404, 422)


def _summarise(error: Exception) -> str:
    status = getattr(error, "status_code", None)
    prefix = f"HTTP {status} " if status else ""
    return f"{prefix}{type(error).__name__}: {str(error)[:200]}"


def _normalise(message) -> dict:
    """Convert an SDK message into the dict shape the agent loop passes around."""
    tool_calls = []

    for call in getattr(message, "tool_calls", None) or []:
        try:
            arguments = json.loads(call.function.arguments or "{}")
        except (json.JSONDecodeError, TypeError):
            arguments = {}

        tool_calls.append(
            {
                "id": call.id,
                "name": call.function.name,
                "arguments": arguments,
                "raw_arguments": call.function.arguments or "{}",
            }
        )

    return {"content": message.content or "", "tool_calls": tool_calls}


def assistant_turn(message: dict) -> dict:
    """Rebuild the assistant message for the conversation history."""
    turn: dict = {"role": "assistant", "content": message["content"] or None}

    if message["tool_calls"]:
        turn["tool_calls"] = [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": call["raw_arguments"],
                },
            }
            for call in message["tool_calls"]
        ]

    return turn


def tool_turn(call_id: str, result, max_chars: int = 4000) -> dict:
    """Build a tool-result message, truncating oversized payloads."""
    content = json.dumps(result, default=str)
    if len(content) > max_chars:
        content = content[:max_chars] + '... "truncated": true}'

    return {"role": "tool", "tool_call_id": call_id, "content": content}
