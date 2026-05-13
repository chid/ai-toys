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
from typing import Any, Callable, Generator, Iterable


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
            # Most OpenAI-compat servers expose endpoints under /v1. The SDK
            # appends paths directly to base_url, so http://host:8080 sends
            # requests to .../chat/completions instead of .../v1/chat/completions.
            if not base_url.rstrip("/").endswith("/v1"):
                corrected = base_url.rstrip("/") + "/v1"
                print(
                    f"[llm_backend] base_url '{base_url}' doesn't end with /v1 — "
                    f"correcting to '{corrected}'.",
                    file=sys.stderr,
                )
                base_url = corrected
            self._client = OpenAI(base_url=base_url, api_key=api_key)
            if model is None:
                model = _first_available_model(base_url, api_key)
                if model:
                    print(f"[llm_backend] Auto-detected model: {model}", file=sys.stderr)
                else:
                    model = "default"
                    print(
                        "[llm_backend] Could not detect model; set LLM_MODEL to suppress this.",
                        file=sys.stderr,
                    )
            self.model = model

        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose 'openai' or 'llamacpp'.")

    # ------------------------------------------------------------------ #
    #  Core chat                                                            #
    # ------------------------------------------------------------------ #

    def chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
        show_reasoning: bool = True,
        tools: list[dict] | None = None,
    ) -> str | Generator[str, None, None]:
        """Send messages; return a string or a streaming token generator."""
        try:
            if self.backend == "llamacpp":
                kw: dict = dict(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=stream,
                )
                if tools:
                    kw["tools"] = tools
                result = self._llm.create_chat_completion(**kw)
                if stream:
                    return self._llamacpp_stream(result)
                return result["choices"][0]["message"]["content"]

            kw = dict(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
            )
            if tools:
                kw["tools"] = tools
            resp = self._client.chat.completions.create(**kw)
            if stream:
                return self._openai_stream(resp, show_reasoning=show_reasoning)
            return resp.choices[0].message.content

        except Exception as exc:
            raise RuntimeError(
                f"LLM request failed (backend={self.backend}, "
                f"model={getattr(self, 'model', '?')}): {exc}"
            ) from exc

    def complete(self, prompt: str, **kwargs) -> str | Generator[str, None, None]:
        """Single-turn completion shorthand."""
        return self.chat(messages=[{"role": "user", "content": prompt}], **kwargs)

    # ------------------------------------------------------------------ #
    #  Tool-use loop                                                        #
    # ------------------------------------------------------------------ #

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_fns: dict[str, Callable],
        *,
        max_rounds: int = 5,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        on_tool_call: Callable[[str, dict, Any], None] | None = None,
    ) -> str:
        """
        Run a tool-use loop: send messages, execute any tool_calls the model
        requests, feed results back, repeat until the model stops or max_rounds
        is reached. Returns the final text response.

        tools        – list of OpenAI-format tool schemas
        tool_fns     – {function_name: callable} mapping
        on_tool_call – optional callback(name, args, result) for display/logging
        """
        msgs = list(messages)

        for _ in range(max_rounds):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as exc:
                raise RuntimeError(f"Tool-use request failed: {exc}") from exc

            msg = resp.choices[0].message
            msgs.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                return msg.content or ""

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                    result = tool_fns[fn_name](**fn_args)
                except Exception as exc:
                    result = f"Error calling {fn_name}: {exc}"

                if on_tool_call:
                    on_tool_call(fn_name, fn_args, result)

                msgs.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })

        return "[max tool rounds reached]"

    # ------------------------------------------------------------------ #
    #  Embeddings                                                           #
    # ------------------------------------------------------------------ #

    def embed(self, texts: str | list[str]) -> list[float] | list[list[float]]:
        """
        Return embedding vector(s). Requires an embedding-capable model.
        str  input → list[float]
        list input → list[list[float]]
        """
        single = isinstance(texts, str)
        inputs: list[str] = [texts] if single else texts  # type: ignore[list-item]

        if self.backend == "llamacpp":
            results = [self._llm.embed(t) for t in inputs]
        else:
            try:
                resp = self._client.embeddings.create(model=self.model, input=inputs)
                results = [e.embedding for e in sorted(resp.data, key=lambda e: e.index)]
            except Exception as exc:
                raise RuntimeError(f"Embedding request failed: {exc}") from exc

        return results[0] if single else results

    # ------------------------------------------------------------------ #
    #  Streaming internals                                                  #
    # ------------------------------------------------------------------ #

    def _openai_stream(self, resp, *, show_reasoning: bool = True) -> Generator[str, None, None]:
        """Yield content tokens; print reasoning_content inline in dim style."""
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
        decoder = json.JSONDecoder()
        for i, c in enumerate(text):
            if c in "{[":
                try:
                    return decoder.raw_decode(text, i)[0]
                except (json.JSONDecodeError, ValueError, IndexError):
                    continue
        return None


# ──────────────────────────────────────────────────────── Conversation ──

class Conversation:
    """
    Stateful chat session. Owns history and wraps an LLMBackend.

    Usage:
        conv = Conversation(llm, system="You are a helpful assistant.")

        # non-streaming
        reply: str = conv.chat("Hello!")

        # streaming — history is committed once the generator is exhausted
        gen = conv.stream("Hello!")
        full: str = llm.stream_print(gen)
    """

    def __init__(
        self,
        llm: LLMBackend,
        system: str | None = None,
        max_history: int = 20,
    ) -> None:
        self.llm = llm
        self.system = system
        self.max_history = max_history
        self.history: list[dict] = []

    def _build_messages(self, user_content: str) -> list[dict]:
        msgs: list[dict] = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        msgs.extend(self.history)
        msgs.append({"role": "user", "content": user_content})
        return msgs

    def _commit(self, user: str, assistant: str) -> None:
        self.history.append({"role": "user", "content": user})
        self.history.append({"role": "assistant", "content": assistant})
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-(self.max_history * 2):]

    def chat(self, message: str, **kwargs) -> str:
        """Send a message, update history, return the response string."""
        msgs = self._build_messages(message)
        response = self.llm.chat(msgs, **kwargs)
        assert isinstance(response, str), "Use stream() for streaming responses"
        self._commit(message, response)
        return response

    def stream(self, message: str, **kwargs) -> Generator[str, None, None]:
        """
        Stream a response. History is committed once the generator is fully
        consumed (e.g. after llm.stream_print()). Pass stream=True automatically.
        """
        kwargs.setdefault("stream", True)
        msgs = self._build_messages(message)
        gen = self.llm.chat(msgs, **kwargs)

        def _capturing() -> Generator[str, None, None]:
            buf: list[str] = []
            for token in gen:
                buf.append(token)
                yield token
            self._commit(message, "".join(buf))

        return _capturing()

    def reset(self) -> None:
        """Clear conversation history (system prompt is preserved)."""
        self.history.clear()


# ──────────────────────────────────────────────────── default_backend ───

def default_backend() -> LLMBackend:
    """Build an LLMBackend from environment variables."""
    btype = os.getenv("LLM_BACKEND", "openai")
    if btype == "llamacpp":
        model_path = os.getenv("LLM_MODEL")
        if not model_path:
            raise EnvironmentError("Set LLM_MODEL=/path/to/model.gguf for llamacpp backend")
        return LLMBackend("llamacpp", model_path=model_path)
    return LLMBackend(
        "openai",
        model=os.getenv("LLM_MODEL") or None,
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("LLM_API_KEY", "ollama"),
    )
