## Done

- [x] Tool/function calling — `chat_with_tools()` with automatic round-trip loop
- [x] Conversation class — thin wrapper that owns its history and calls `self.llm.chat()` automatically
- [x] Embeddings — `embed(texts)` via `/v1/embeddings`

## High value

- [ ] Async variant — `achat()` using `AsyncOpenAI` for concurrent personas/dungeon events
- [ ] Token budget tracking — read usage from non-streaming responses, accumulate session total, warn near context limit
- [ ] Retry with backoff — exponential backoff on 429/503

## Moderate value

- [ ] Grammar-constrained JSON — GBNF grammar for llamacpp backend
- [ ] Structured output via `response_format` — `{"type": "json_object"}` or JSON schema
- [ ] Vision support — image URLs or base64 in messages
