All four files pass syntax checks. Here's what was created:

---

### Files

| File                    | Role                                            |
| ----------------------- | ----------------------------------------------- |
| `llm_backend.py`        | Shared backend — configure once, use everywhere |
| `ai_dungeon.py`         | Interactive text adventure                      |
| `code_archaeologist.py` | Code analysis with a fun framing                |
| `debate_club.py`        | Two AI personas argue a topic                   |
| `requirements.txt`      | Dependencies                                    |

---

### The three examples

**`ai_dungeon.py`** — A stateful text adventure where the LLM is the dungeon master. After every narration it emits a JSON block (`health_delta`, `add_items`, `exits`, `victory`, etc.) that the game engine reads and applies to a live `GameState`. There's a persistent world with 5 rooms and a lich boss named Morthax — items found early matter later.

**`code_archaeologist.py`** — Feed it any source file (or pipe from stdin). The LLM plays Dr. A.R. Tifact, an eccentric archaeologist who writes a structured "Field Report" covering classification, purpose, notable curiosities, hazards, and estimated provenance. Point it at old code and it will find the suspicious bits.

**`debate_club.py`** — Two sharply contrasting personas (a verbose Victorian polymath and a terse startup bro) argue opposite sides of any topic you give them. They alternate for N rounds, each building on the previous argument.

---

### Quickstart (Ollama)

```bash
# default: Ollama on localhost with llama3.2
python3 ai_dungeon.py

# different model
python3 debate_club.py "Python vs Rust" --model mistral

# point at your own llamacpp server or LM Studio
LLM_BASE_URL=http://localhost:1234/v1 python3 code_archaeologist.py llm_backend.py

# use a local .gguf directly
python3 ai_dungeon.py --backend llamacpp --model-path ~/models/llama3-q4.gguf

# env-var config
LLM_BACKEND=llamacpp LLM_MODEL=~/models/qwen2.gguf python3 debate_club.py "AI will replace programmers"
```

LLM_API_KEY=your-api-key LLM_BASE_URL=http://localhost:8080 python3 debate_club.py "AI will replace programmers"
LLM_API_KEY=your-secret-key LLM_BASE_URL=http://localhost:8080 python3 debate_club.py "AI will replace programmers"