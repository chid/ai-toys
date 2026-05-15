# AI Toys

A small collection of single-file Python programs that show what you can
actually *do* with a large language model. Each toy is 100–300 lines, picks
one LLM pattern, and is meant to be read end-to-end. The shared
`llm_backend.py` hides the HTTP/SDK boilerplate so the toys themselves stay
focused on their idea.

If you've used ChatGPT or Claude but never written code against an LLM, this
repo is a friendly place to start. Skim a file, run it, change a prompt, see
what happens.

## The toys

| File | What it is | What it teaches |
|---|---|---|
| `code_archaeologist.py` | Dr. A.R. Tifact writes a "field report" on any source file | The simplest pattern: one prompt, one reply, one persona |
| `debate_club.py` | Two contrasting personas argue a topic for N rounds | Multi-persona conversations, manual history management |
| `ai_dungeon.py` | A text adventure where the LLM is the dungeon master | Streaming output, structured JSON for game state, a stateful loop |
| `pen_pal.py` | A persona writes you a letter; remembers across runs via disk | Long-horizon memory, file persistence, history summarization |
| `prompt_injection_dojo.py` | You try to make a guard persona leak its secret across 4 levels | Adversarial prompting; why system-prompt guardrails are not security |

More ideas (not yet built) live in [IDEAS.md](IDEAS.md).

## What you'll need

Python 3.10+ and **one** of these three backends. Pick whichever is easiest:

- **A free hosted API key** — easiest, no install, runs in the cloud.
- **Ollama running locally** — private, no key, costs nothing, needs a ~4 GB
  model download.
- **A llama.cpp `.gguf` file** — most control, most setup.

All three speak the same protocol from this code's point of view, so swapping
between them is just changing environment variables.

### Option A — free hosted API (recommended for first try)

These providers each have a free tier and expose an OpenAI-compatible HTTP
endpoint, which is all this repo needs. Pick one:

**[OpenRouter](https://openrouter.ai)** — Aggregator that proxies many models.
Look for models with a `:free` suffix.

```bash
export LLM_BASE_URL=https://openrouter.ai/api/v1
export LLM_API_KEY=sk-or-v1-...                       # from openrouter.ai dashboard
export LLM_MODEL=meta-llama/llama-3.1-8b-instruct:free
```

**[Cerebras](https://cloud.cerebras.ai)** — Very fast inference on their own
hardware; generous free tier.

```bash
export LLM_BASE_URL=https://api.cerebras.ai/v1
export LLM_API_KEY=csk-...                            # from cloud.cerebras.ai
export LLM_MODEL=llama3.1-8b
```

(Model names and free-tier details drift — check the provider's model list if
the example above 404s.)

### Option B — Ollama (local, private, free)

1. Install from <https://ollama.com>.
2. Pull a small chat model:
   ```bash
   ollama pull llama3.2
   ```
3. That's it. Ollama serves at `http://localhost:11434/v1`, which is also
   this repo's default — no env vars needed.

### Option C — llama.cpp direct

```bash
pip install llama-cpp-python
python3 ai_dungeon.py --backend llamacpp --model-path ~/models/qwen2.gguf
```

For GPU offload:
`CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python`.

## Install

```bash
git clone <this-repo>
cd ai-toys
pip install -r requirements.txt
```

## Run

Once your backend env vars are set (or Ollama is running), any toy works:

```bash
python3 code_archaeologist.py llm_backend.py
python3 debate_club.py "Cats are better than dogs" --rounds 4
python3 ai_dungeon.py
python3 pen_pal.py
python3 prompt_injection_dojo.py
```

Every toy accepts `--backend`, `--model`, `--base-url`, etc. as flags too —
useful if you want to compare models without changing your shell:

```bash
python3 debate_club.py "Python beats Rust" --model mistral
LLM_BASE_URL=https://api.cerebras.ai/v1 \
LLM_API_KEY=csk-... \
LLM_MODEL=llama3.1-8b \
python3 ai_dungeon.py
```

## A reading order, if you want to learn

The point of this repo is to be readable, not just runnable. Suggested path:

1. **`llm_backend.py`** — the only "infrastructure" file. Start with
   `LLMBackend.chat()` (around line 103). That single method is the entire
   surface area of "talking to an LLM": you send a list of `{role, content}`
   dicts, you get back a string (or a token-by-token stream).
2. **`code_archaeologist.py`** — the smallest real toy. One system prompt,
   one user message, one reply. This is the whole game.
3. **`debate_club.py`** — adds multiple personas and a turn loop. Shows
   that "conversation memory" is just a list you keep appending to.
4. **`ai_dungeon.py`** — adds *structured output*: the model emits a JSON
   block alongside its prose, and Python parses it into a game state. Shows
   how to coax structure out of a model with prompt engineering.
5. **`pen_pal.py`** — adds persistence across separate runs of the program,
   and demonstrates summarizing older history to keep the context window
   bounded.
6. **`prompt_injection_dojo.py`** — the adversarial one. Run it and try to
   break each level. The lesson is that there is no "system-prompt is
   trusted, user-message is not" — they're all just text the model reads.

## Configuration reference

All toys read the same environment variables. Each is also overridable per
run via a CLI flag (see `--help` on any toy).

| Env var        | Default                          | Purpose |
|----------------|----------------------------------|---------|
| `LLM_BACKEND`  | `openai`                         | `openai` (any OpenAI-compatible HTTP API) or `llamacpp` (local `.gguf` file) |
| `LLM_BASE_URL` | `http://localhost:11434/v1`      | HTTP endpoint of the OpenAI-compatible server |
| `LLM_API_KEY`  | `ollama`                         | API key (Ollama ignores it; hosted providers require it) |
| `LLM_MODEL`    | auto-detected from `/v1/models`  | Model name on the server, or path to `.gguf` for llamacpp |

## Troubleshooting

- **"Could not detect model"** — the provider's `/models` listing failed or
  is empty. Set `LLM_MODEL` explicitly.
- **`base_url 'http://...' doesn't end with /v1` warning** — harmless; the
  backend auto-appends `/v1`. Fix it in your env var if you want the warning
  gone.
- **Empty / garbled output on a tiny local model** — some 1–3B-parameter
  models can't keep up with structured-output toys like `ai_dungeon.py`. Try
  an 8B model.
- **The hosted free tier rate-limits you** — switch providers, or pull a
  local model with Ollama. The toys don't care.

## What this is not

Not a framework, not a library, not production code. Each file is meant to
be copied, hacked on, and thrown away. If you find yourself wanting to share
code between toys, that's a hint that `llm_backend.py` should grow a small
helper — see [TODO.md](TODO.md) for ideas already on the list.
