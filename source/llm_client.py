import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMConfig:
    backend: str = "offline"
    base_url: str = "http://127.0.0.1:8317/api/provider/codex/v1"
    model: str = "gpt-5.5"
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str | None = None
    timeout_s: int = 60
    max_retries: int = 2

    def safe_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "base_url": self.base_url,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "api_key_present": bool(self.resolved_api_key()),
        }

    def resolved_api_key(self) -> str | None:
        return self.api_key or os.environ.get(self.api_key_env)


class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        api_key = config.resolved_api_key()
        if not api_key:
            raise RuntimeError(f"API key missing. Set {config.api_key_env} or pass --api-key.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI package is required for --backend openai-compatible.") from exc
        self.client = OpenAI(api_key=api_key, base_url=config.base_url, timeout=config.timeout_s)

    def chat_text(self, system: str, prompt: str) -> str:
        last_error = None
        for _ in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"LLM request failed: {_redact(str(last_error))}")

    def chat_json(self, system: str, prompt: str) -> dict[str, Any]:
        last_error = None
        for _ in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                )
                return _parse_json(response.choices[0].message.content or "{}")
            except Exception as exc:
                last_error = exc
                try:
                    return _parse_json(self.chat_text(system, prompt))
                except Exception:
                    continue
        raise RuntimeError(f"LLM JSON request failed: {_redact(str(last_error))}")


def make_client(config: LLMConfig) -> OpenAICompatibleClient | None:
    if config.backend == "offline":
        return None
    return OpenAICompatibleClient(config)


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start:end + 1]
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object")
    return parsed


def _redact(text: str) -> str:
    for value in os.environ.values():
        if value and len(value) > 8 and value in text:
            text = text.replace(value, "<redacted>")
    return text
