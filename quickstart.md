## Quickstart (Ollama)

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
