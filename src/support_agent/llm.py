"""One chat client for every model (small LoRA on Ollama, large model on OpenRouter, judge), with:

- usage and cost per call, so the graph can log latency/tokens/cost per node;
- record / replay, so CI can rerun the evaluation deterministically and for free from saved answers;
- a `mock` provider that answers offline with simple rules (tests, demos without a key).

The mode comes from LLM_MODE: `live` (default), `record` (call and save), `replay` (saved answers only).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from support_agent.config import RECORDINGS_DIR, ModelSpec, settings


@dataclass
class LLMResult:
    text: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: float
    replayed: bool = False


class ReplayMiss(RuntimeError):
    """Replay mode, and this exact request was never recorded."""


def parse_json(text: str) -> dict:
    """The first JSON object in a model reply (models sometimes wrap it in prose or ``` fences)."""
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in: {text[:200]!r}")
    return json.loads(match.group(0))


class _Recordings:
    """One JSONL file per logical model; loaded once, appended to in record mode."""

    _lock = threading.Lock()

    def __init__(self, directory: Path, name: str):
        self.path = directory / f"{name}.jsonl"
        self.items: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.items[row["key"]] = row

    def get(self, key: str) -> dict | None:
        return self.items.get(key)

    def put(self, key: str, row: dict) -> None:
        with self._lock:
            self.items[key] = row
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, **row}, ensure_ascii=False) + "\n")


class LLM:
    def __init__(self, spec: ModelSpec, mode: str | None = None, recordings_dir: Path | None = None):
        self.spec = spec
        self.mode = mode or settings().llm_mode
        self.recordings = _Recordings(recordings_dir or RECORDINGS_DIR, spec.name)
        self._client = None

    # -- public -------------------------------------------------------------------------------------------------
    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 800,
             json_mode: bool = True, sample: int = 0) -> LLMResult:
        key = self._key(messages, temperature, sample)
        if self.mode in ("replay", "record") and self.spec.provider != "mock":
            saved = self.recordings.get(key)
            if saved:
                return LLMResult(saved["text"], self.spec.model, saved["tokens_in"], saved["tokens_out"],
                                 self._cost(saved["tokens_in"], saved["tokens_out"]), 0.0, replayed=True)
            if self.mode == "replay":
                raise ReplayMiss(f"{self.spec.name}: request not recorded (key {key}). Run once with LLM_MODE=record.")

        start = time.perf_counter()
        if self.spec.provider == "mock":
            from support_agent.mock_llm import mock_reply
            text = mock_reply(messages, sample=sample)
            tokens_in = sum(len(m["content"] or "") for m in messages) // 4
            tokens_out = len(text) // 4
        else:
            text, tokens_in, tokens_out = self._call_api(messages, temperature, max_tokens, json_mode, sample)
        latency = (time.perf_counter() - start) * 1000

        if self.mode == "record" and self.spec.provider != "mock":
            self.recordings.put(key, {"model": self.spec.model, "text": text,
                                      "tokens_in": tokens_in, "tokens_out": tokens_out})
        return LLMResult(text, self.spec.model, tokens_in, tokens_out, self._cost(tokens_in, tokens_out), latency)

    def chat_json(self, messages: list[dict], **kwargs) -> tuple[dict, LLMResult]:
        result = self.chat(messages, **kwargs)
        return parse_json(result.text), result

    # -- internals ----------------------------------------------------------------------------------------------
    def _key(self, messages, temperature, sample) -> str:
        raw = json.dumps([self.spec.model, messages, temperature, sample], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def _cost(self, tokens_in: int, tokens_out: int) -> float:
        if self.spec.provider == "mock":
            return 0.0
        return (tokens_in * self.spec.price_in + tokens_out * self.spec.price_out) / 1e6

    def _call_api(self, messages, temperature, max_tokens, json_mode, sample):
        if self._client is None:
            from openai import OpenAI
            key = os.getenv(self.spec.api_key_env) if self.spec.api_key_env else "not-needed"
            if self.spec.api_key_env and not key:
                raise RuntimeError(f"{self.spec.api_key_env} is not set (model '{self.spec.name}'). See .env.example.")
            self._client = OpenAI(base_url=self.spec.base_url, api_key=key, max_retries=3, timeout=120)
        kwargs = dict(model=self.spec.model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        if sample:
            kwargs["seed"] = sample
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        from openai import BadRequestError, UnprocessableEntityError
        try:
            response = self._client.chat.completions.create(**kwargs)
        except (BadRequestError, UnprocessableEntityError) as err:
            # Some providers reject response_format or seed: retry once without them. Only on a 400/422: a 429 or a
            # timeout was already retried by the client (max_retries) and must not cost one more call.
            if not json_mode and not sample:
                raise
            kwargs.pop("response_format", None)
            kwargs.pop("seed", None)
            try:
                response = self._client.chat.completions.create(**kwargs)
            except Exception:
                raise err
        usage = response.usage
        text = response.choices[0].message.content or ""
        return text, getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0


_instances: dict[str, LLM] = {}


def get_llm(name: str) -> LLM:
    """Shared clients for 'small', 'large' and 'judge' (recordings are loaded once)."""
    s = settings()
    spec = {"small": s.small, "large": s.large, "judge": s.judge}[name]
    cache_key = f"{name}|{spec.provider}|{spec.model}|{s.llm_mode}"
    if cache_key not in _instances:
        _instances[cache_key] = LLM(spec, mode=s.llm_mode)
    return _instances[cache_key]


def reset_clients() -> None:
    _instances.clear()
