"""Provider adapters; the investigation loop remains provider-independent."""

import os
import time
from dataclasses import dataclass
from pathlib import Path

FINAL_ANSWER_FORMAT = {
    "type": "json_schema", "name": "investigation_answer", "strict": True,
    "schema": {"type": "object", "additionalProperties": False,
               "properties": {"answer": {"type": "string"},
                              "citations": {"type": "array", "items": {"type": "string"}}},
               "required": ["answer", "citations"]},
}


class ConfigurationError(ValueError):
    pass


@dataclass
class Settings:
    provider: str
    model: str
    api: str
    max_rounds: int = 14
    max_output_tokens: int = 4096
    timeout: float = 45
    min_request_interval: float = 0


def read_settings() -> Settings:
    try:
        from dotenv import load_dotenv
    except ImportError as error:
        raise ConfigurationError("Install dependencies: python -m pip install -r requirements.txt") from error
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if not provider or provider == "auto":
        configured = [name for name, key in (("openai", "OPENAI_API_KEY"),
                      ("anthropic", "ANTHROPIC_API_KEY"), ("azure", "AZURE_OPENAI_API_KEY"))
                      if os.getenv(key)]
        if len(configured) > 1:
            raise ConfigurationError("Multiple providers are configured. Set LLM_PROVIDER in .env.")
        provider = configured[0] if configured else "openai"
    if provider in {"compatible", "openai-compatible"}:
        provider = "openai"
    if provider not in {"openai", "anthropic", "azure"}:
        raise ConfigurationError("LLM_PROVIDER must be openai, anthropic, azure, or auto.")
    model_key = {"openai": "OPENAI_MODEL", "anthropic": "ANTHROPIC_MODEL",
                 "azure": "AZURE_OPENAI_DEPLOYMENT"}[provider]
    key_name = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
                "azure": "AZURE_OPENAI_API_KEY"}[provider]
    for name in (key_name, model_key):
        value = os.getenv(name, "").strip()
        if not value or value in {"...", "sk-...", "sk-ant-...", "your-api-key", "your-model-name"}:
            raise ConfigurationError(f"Set {name} in the Side Quest 1 .env file. Do not put keys in source code.")
    if provider == "azure":
        for name in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION"):
            if not os.getenv(name):
                raise ConfigurationError(f"Set {name} in .env for Azure.")
    default_api = "chat_completions" if provider == "azure" or os.getenv("OPENAI_BASE_URL") else "responses"
    api = os.getenv("LLM_API", default_api).strip().lower()
    if provider == "anthropic":
        api = "messages"
    elif api not in {"responses", "chat_completions"}:
        raise ConfigurationError("LLM_API must be responses or chat_completions.")
    try:
        rounds = int(os.getenv("LLM_MAX_ROUNDS", "14"))
        tokens = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "4096"))
        timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
        interval = float(os.getenv("LLM_MIN_REQUEST_INTERVAL_SECONDS", "0"))
        if not 2 <= rounds <= 40 or not 256 <= tokens <= 32768 or not 1 <= timeout <= 300 or not 0 <= interval <= 60:
            raise ValueError
    except ValueError as error:
        raise ConfigurationError("Use LLM_MAX_ROUNDS 2-40, LLM_MAX_OUTPUT_TOKENS 256-32768, LLM_TIMEOUT_SECONDS 1-300, and LLM_MIN_REQUEST_INTERVAL_SECONDS 0-60.") from error
    return Settings(provider, os.environ[model_key].strip(), api, rounds, tokens, timeout, interval)


def make_client(settings: Settings):
    try:
        if settings.provider == "anthropic":
            from anthropic import Anthropic
            return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"],
                             timeout=settings.timeout, max_retries=2)
        from openai import AzureOpenAI, OpenAI
        if settings.provider == "azure":
            return AzureOpenAI(api_key=os.environ["AZURE_OPENAI_API_KEY"],
                               azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                               api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                               timeout=settings.timeout, max_retries=2)
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                      base_url=os.getenv("OPENAI_BASE_URL") or None,
                      timeout=settings.timeout, max_retries=2)
    except ImportError as error:
        raise ConfigurationError("Install dependencies: python -m pip install -r requirements.txt") from error


class Conversation:
    _last_request_at = 0.0

    def __init__(self, client, settings: Settings, system: str, prompt: str):
        self.client, self.settings, self.system = client, settings, system
        self.messages = [{"role": "user", "content": prompt}]
        if settings.api == "chat_completions":
            self.messages.insert(0, {"role": "system", "content": system})

    def request(self, schemas: list[dict]) -> tuple[str, list[dict]]:
        delay = self.settings.min_request_interval - (time.monotonic() - Conversation._last_request_at)
        if delay > 0:
            time.sleep(delay)
        Conversation._last_request_at = time.monotonic()
        common = {"model": self.settings.model}
        if self.settings.api == "responses":
            # Native OpenAI supports schema-constrained final answers alongside
            # function calling. Custom endpoints retain their existing API shape.
            output_format = ({"text": {"format": FINAL_ANSWER_FORMAT}}
                             if self.settings.provider == "openai" and not os.getenv("OPENAI_BASE_URL") else {})
            response = self.client.responses.create(
                **common, **output_format, instructions=self.system, input=self.messages,
                tools=[{"type": "function", **schema} for schema in schemas],
                max_output_tokens=self.settings.max_output_tokens,
                store=False, include=["reasoning.encrypted_content"],
            )
            self.messages.extend(item.model_dump(exclude_none=True) for item in response.output)
            calls = [{"id": item.call_id, "name": item.name, "arguments": item.arguments}
                     for item in response.output if item.type == "function_call"]
            return response.output_text, calls
        if self.settings.api == "messages":
            response = self.client.messages.create(
                **common, system=self.system, messages=self.messages,
                max_tokens=self.settings.max_output_tokens,
                tools=[{"name": schema["name"], "description": schema["description"],
                        "input_schema": schema["parameters"]} for schema in schemas],
            )
            self.messages.append({"role": "assistant", "content": [
                block.model_dump(exclude_none=True) for block in response.content]})
            calls = [{"id": block.id, "name": block.name, "arguments": block.input}
                     for block in response.content if block.type == "tool_use"]
            text = "\n".join(block.text for block in response.content if block.type == "text")
            return text, calls
        token_field = os.getenv("LLM_CHAT_TOKEN_FIELD", "max_tokens" if os.getenv("OPENAI_BASE_URL") else "max_completion_tokens")
        if token_field not in {"max_tokens", "max_completion_tokens"}:
            raise ConfigurationError("LLM_CHAT_TOKEN_FIELD must be max_tokens or max_completion_tokens.")
        response = self.client.chat.completions.create(
            **common, **{token_field: self.settings.max_output_tokens}, messages=self.messages,
            tools=[{"type": "function", "function": schema} for schema in schemas],
        )
        message = response.choices[0].message
        # Preserve provider-specific fields (e.g. a reasoning_content field) if returned.
        self.messages.append(message.model_dump(exclude_none=True))
        calls = [{"id": call.id, "name": call.function.name, "arguments": call.function.arguments}
                 for call in (message.tool_calls or [])]
        return message.content or "", calls

    def add_results(self, results: list[dict]) -> None:
        import json
        if self.settings.api == "messages":
            self.messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": result["id"],
                 "content": json.dumps(result["result"], allow_nan=False),
                 "is_error": isinstance(result["result"], dict) and "error" in result["result"]}
                for result in results]})
        else:
            for result in results:
                output = json.dumps(result["result"], allow_nan=False)
                self.messages.append(
                    {"type": "function_call_output", "call_id": result["id"], "output": output}
                    if self.settings.api == "responses" else
                    {"role": "tool", "tool_call_id": result["id"], "content": output})

    def correct(self, message: str) -> None:
        self.messages.append({"role": "user", "content": message})


def public_api_error(error: Exception) -> str:
    """Do not echo provider exception bodies, which may contain credentials."""
    name = type(error).__name__
    status = getattr(error, "status_code", None)
    if status in {401, 403}:
        return "The provider rejected access. Check the API key and model permissions in .env."
    if status == 429:
        return "The provider reported a rate or quota limit. Check the account, then retry."
    if status in {500, 502, 503, 504}:
        return f"The LLM provider is temporarily unavailable (HTTP {status}). Retry later; your configuration has been preserved."
    if "Connection" in name or "Timeout" in name:
        return "Cannot reach the configured LLM. Check its server, OPENAI_BASE_URL, and network connection."
    return f"LLM request failed ({name}{', HTTP ' + str(status) if status else ''}). Check model name, tool support, and LLM_API."
