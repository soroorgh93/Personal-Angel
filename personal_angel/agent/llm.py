"""OpenAI-compatible chat client (Ollama on the PC, vLLM on a GPU workstation) with
streaming, image inputs and exposed reasoning; plus a deterministic fixture.

The same code path talks to:
  * Ollama          http://127.0.0.1:11434/v1   (qwen3.5:9b / :4b)
  * vLLM          http://127.0.0.1:8000/v1
"""
from __future__ import annotations

import base64
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import httpx

log = logging.getLogger(__name__)
StreamFn = Callable[[str, str], None]

@dataclass
class LLMResponse:
    content: str
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    model: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

def image_to_data_url(path: str | Path, max_side: int = 768) -> str:
    """Downscale evidence before sending: vision tokens dominate prefill cost."""
    try:
        import cv2

        img = cv2.imread(str(path))
        if img is not None:
            h, w = img.shape[:2]
            scale = min(1.0, max_side / max(h, w))
            if scale < 1.0:
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        pass
    with open(path, "rb") as handle:
        return "data:image/jpeg;base64," + base64.b64encode(handle.read()).decode("ascii")

def extract_json_object(text: str) -> dict[str, Any] | None:
    """Find the first balanced JSON object in free text (models love prose)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [fence.group(1)] if fence else []
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start:i + 1])
                    break
        start = text.find("{", start + 1)
        if len(candidates) > 6:
            break
    for cand in candidates:
        try:
            value = json.loads(cand)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return None

class LLMClient:
    name = "abstract"
    is_real = False

    def chat(self, messages: list[dict[str, Any]], stream: StreamFn | None = None,
             json_mode: bool = False, max_tokens: int | None = None, images: list[str] | None = None,
             temperature: float | None = None) -> LLMResponse:
        raise NotImplementedError

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        return None

    def health(self) -> dict[str, Any]:
        return {"ok": True, "backend": self.name}

class OpenAICompatibleClient(LLMClient):
    is_real = True

    def __init__(self, config: dict[str, Any]) -> None:
        self.base_url = str(config.get("base_url", "http://127.0.0.1:11434/v1")).rstrip("/")
        self.api_key = str(config.get("api_key", "none"))
        self.model = str(config.get("model", "qwen3.5:9b"))
        self.vision_model = config.get("vision_model") or self.model
        self.temperature = float(config.get("temperature", 0.2))
        self.max_tokens = int(config.get("max_tokens", 1200))
        self.timeout = float(config.get("timeout_s", 180))
        self.thinking = bool(config.get("thinking", False))

        self.thinking_control = str(config.get("thinking_control", "auto"))
        if self.thinking_control == "auto":
            self.thinking_control = "reasoning_effort" if "11434" in self.base_url else "chat_template_kwargs"
        self.embedding_model = str(config.get("embedding_model", "nomic-embed-text"))
        self.name = f"openai_compatible:{self.model}"
        self._client = httpx.Client(timeout=httpx.Timeout(self.timeout, connect=10.0))

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "none":
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def health(self) -> dict[str, Any]:
        try:
            r = self._client.get(f"{self.base_url}/models", headers=self._headers(), timeout=10)
            models = [m.get("id") for m in r.json().get("data", [])]
            return {"ok": r.status_code == 200, "backend": self.name, "models": models,
                    "configured_model_served": self.model in models or not models}
        except Exception as error:
            return {"ok": False, "backend": self.name, "error": str(error)}

    def chat(self, messages, stream=None, json_mode=False, max_tokens=None, images=None, temperature=None):
        messages = [dict(m) for m in messages]
        model = self.model
        if images:
            model = self.vision_model
            last = messages[-1]
            content = [{"type": "text", "text": str(last.get("content", ""))}]
            for path in images:
                content.append({"type": "image_url", "image_url": {"url": image_to_data_url(path)}})
            last["content"] = content
        payload: dict[str, Any] = {
            "model": model, "messages": messages, "stream": stream is not None,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.thinking_control == "reasoning_effort":
            payload["reasoning_effort"] = "medium" if self.thinking else "none"
        else:
            payload["chat_template_kwargs"] = {"enable_thinking": self.thinking}
        if stream is not None:
            payload["stream_options"] = {"include_usage": True}
        t0 = time.perf_counter()
        if stream is None:
            r = self._client.post(f"{self.base_url}/chat/completions", headers=self._headers(), json=payload)
            r.raise_for_status()
            data = r.json()
            choice = data["choices"][0]["message"]
            usage = data.get("usage", {})
            content_text = choice.get("content") or ""
            reasoning_text = choice.get("reasoning_content") or choice.get("reasoning") or ""
            think = re.search(r"<think>(.*?)</think>", content_text, re.S)
            if think:
                reasoning_text = reasoning_text or think.group(1)
                content_text = content_text.replace(think.group(0), "").strip()
            return LLMResponse(
                content=content_text,
                reasoning=reasoning_text,
                tool_calls=choice.get("tool_calls") or [],
                tokens_in=int(usage.get("prompt_tokens", 0)), tokens_out=int(usage.get("completion_tokens", 0)),
                latency_ms=(time.perf_counter() - t0) * 1000, model=model, raw=data,
            )
        content, reasoning = [], []
        usage: dict[str, Any] = {}
        with self._client.stream("POST", f"{self.base_url}/chat/completions", headers=self._headers(),
                                 json=payload) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    chunk = json.loads(body)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    r_delta = delta.get("reasoning_content") or delta.get("reasoning") or delta.get("thinking")
                    if r_delta:
                        reasoning.append(r_delta)
                        stream("reasoning", r_delta)
                    c_delta = delta.get("content")
                    if c_delta:
                        content.append(c_delta)
                        stream("content", c_delta)
        text = "".join(content)

        think = re.search(r"<think>(.*?)</think>", text, re.S)
        if think:
            reasoning.append(think.group(1))
            text = text.replace(think.group(0), "").strip()
        approx_in = sum(len(str(m.get("content", ""))) for m in messages) // 4
        return LLMResponse(content=text, reasoning="".join(reasoning),
                           tokens_in=int(usage.get("prompt_tokens", approx_in)),
                           tokens_out=int(usage.get("completion_tokens", (len(text) + len("".join(reasoning))) // 4)),
                           latency_ms=(time.perf_counter() - t0) * 1000, model=model)

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        try:
            r = self._client.post(f"{self.base_url}/embeddings", headers=self._headers(),
                                  json={"model": self.embedding_model, "input": texts}, timeout=60)
            r.raise_for_status()
            return [d["embedding"] for d in r.json()["data"]]
        except Exception as error:
            log.warning("Embedding endpoint unavailable: %s", error)
            return None

class FixtureLLM(LLMClient):
    """Deterministic stand-in: replays a scripted list of JSON replies if
    provided, otherwise defers to the heuristic planner (see master.py)."""

    name = "fixture_llm"

    def __init__(self, script: list[dict[str, Any]] | None = None) -> None:
        self.script = list(script or [])
        self.calls = 0

    def chat(self, messages, stream=None, json_mode=False, max_tokens=None, images=None, temperature=None):
        self.calls += 1
        if self.script:
            reply = self.script.pop(0)
            text = json.dumps(reply)
            if stream:
                stream("reasoning", str(reply.get("thought", "")))
                stream("content", text)
            return LLMResponse(content=text, reasoning=str(reply.get("thought", "")), tokens_in=400,
                               tokens_out=80, latency_ms=5.0, model=self.name)
        return LLMResponse(content="", reasoning="", model=self.name)

def create_llm(config: dict[str, Any], fixture: dict[str, Any] | None = None) -> LLMClient:
    backend = str(config.get("backend", "fixture"))
    if backend == "fixture":
        return FixtureLLM((fixture or {}).get("llm_script"))
    if backend == "openai_compatible":
        return OpenAICompatibleClient(config)
    raise ValueError(f"Unknown llm backend: {backend}")
