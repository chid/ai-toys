"""
Flexible LLM backend.

Backends:
  openai    – OpenAI-compatible HTTP API (Ollama, llama.cpp server,
              LM Studio, real OpenAI, etc.)
  llamacpp  – direct via llama-cpp-python, loads a local .gguf file

Configure via environment variables or pass kwargs directly:
  LLM_BACKEND   openai (default) | llamacpp
  LLM_MODEL     model name (openai) or /path/to/model.gguf (llamacpp)
  LLM_BASE_URL  base URL for openai backend (default: Ollama localhost)
  LLM_API_KEY   API key (default: "ollama")
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from typing import Generator, Iterable


def _first_available_model(base_url: str, api_key: str) -> str | None:
    """Query /models and return the id of the first listed model."""
    try:
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=4) as r:
            data = json.loads(r.read())
        entries = data.get("data") or data.get("models") or []
        if entries:
            return entries[0].get("id") or entries[0].get("name")
    except Exception:
        pass
    return None


class LLMBackend:
    def __init__(
        self,
        backend: str = "openai",
        *,
        # llamacpp options
        model_path: str | None = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        # openai-compat options
        model: str | None = None,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
    ) -> None:
        self.backend = backend

        if backend == "llamacpp":
            from llama_cpp import Llama  # pip install llama-cpp-python
            if not model_path:
                raise ValueError("model_path is required for llamacpp backend")
            self._llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )

        elif backend == "openai":
            from openai import OpenAI  # pip install openai
            # Most OpenAI-compat servers (llama.cpp, Ollama, LM Studio) expose
            # endpoints under /v1. The SDK appends paths directly to base_url,
            # so http://host:8080 → .../chat/completions (wrong);
            #    http://host:8080/v1 → .../v1/chat/completions (correct).
            if not base_url.rstrip("/").endswith("/v1"):
                corrected = base_url.rstrip("/") + "/v1"
                print(
                    f"[llm_backend] base_url '{base_url}' doesn't end with /v1 — "
                    f"correcting to '{corrected}'.",
                    file=sys.stderr,
                )
                base_url = corrected
            self._client = OpenAI(base_url=base_url, api_key=api_key)
            # Auto-detect model from server when not specified
            if model is None:
                model = _first_available_model(base_url, api_key)
                if model:
                    print(f"[llm_backend] Auto-detected model: {model}", file=sys.stderr)
                else:
                    model = "default"
                    print(
                        "[llm_backend] Could not detect model; falling back to 'default'. "
                        "Set LLM_MODEL to suppress this.",
                        file=sys.stderr,
                    )
            self.model = model

        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose 'openai' or 'llamacpp'.")

    # ------------------------------------------------------------------ #
    #  Core interface                                                       #
    # ------------------------------------------------------------------ #

    def chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
        show_reasoning: bool = True,
    ) -> str | Generator[str, None, None]:
        """Send a list of chat messages and return a string or token generator."""
        try:
            if self.backend == "llamacpp":
                result = self._llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=stream,
                )
                if stream:
                    return self._llamacpp_stream(result)
                return result["choices"][0]["message"]["content"]

            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
            )
            if stream:
                return self._openai_stream(resp, show_reasoning=show_reasoning)
            return resp.choices[0].message.content

        except Exception as exc:
            backend_info = f"backend={self.backend}, model={getattr(self, 'model', '?')}"
            raise RuntimeError(f"LLM request failed ({backend_info}): {exc}") from exc

    def _openai_stream(self, resp, *, show_reasoning: bool = True) -> Generator[str, None, None]:
        """Yield content tokens; optionally print reasoning_content inline in dim style."""
        DIM, RESET = "\033[2m", "\033[0m"
        in_reasoning = False

        for chunk in resp:
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            content = getattr(delta, "content", None)

            if reasoning and show_reasoning:
                if not in_reasoning:
                    print(f"{DIM}[thinking] ", end="", flush=True)
                    in_reasoning = True
                print(reasoning, end="", flush=True)

            if content:
                if in_reasoning:
                    print(f"{RESET}\n", end="", flush=True)
                    in_reasoning = False
                yield content

        if in_reasoning:
            print(RESET, end="", flush=True)

    def _llamacpp_stream(self, result) -> Generator[str, None, None]:
        for chunk in result:
            yield chunk["choices"][0]["delta"].get("content", "")

    def complete(self, prompt: str, **kwargs) -> str | Generator[str, None, None]:
        """Single-turn completion shorthand."""
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    # ------------------------------------------------------------------ #
    #  Utilities                                                            #
    # ------------------------------------------------------------------ #

    def stream_print(self, gen: Iterable[str]) -> str:
        """Print a streaming generator token-by-token; return the full text."""
        buf: list[str] = []
        for token in gen:
            print(token, end="", flush=True)
            buf.append(token)
        print()
        return "".join(buf)

    @staticmethod
    def extract_json(text: str) -> dict | list | None:
        """Extract the first JSON object or array from a response string."""
        match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        # fallback: bare JSON object/array
        match = re.search(r"(\{[\s\S]+\}|\[[\s\S]+\])", text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return None


def default_backend() -> LLMBackend:
    """Build an LLMBackend from environment variables."""
    btype = os.getenv("LLM_BACKEND", "openai")
    if btype == "llamacpp":
        model_path = os.getenv("LLM_MODEL")
        if not model_path:
            raise EnvironmentError("Set LLM_MODEL=/path/to/model.gguf for llamacpp backend")
        return LLMBackend("llamacpp", model_path=model_path)
    # model=None triggers auto-detection from /v1/models
    return LLMBackend(
        "openai",
        model=os.getenv("LLM_MODEL") or None,
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("LLM_API_KEY", "ollama"),
    )
