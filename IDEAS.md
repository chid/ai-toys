# Toy Ideas

Candidate single-file toys for this repo. Each demonstrates a distinct LLM
usage pattern beyond what `ai_dungeon.py`, `code_archaeologist.py`, and
`debate_club.py` already cover.

## Multi-agent
- **mystery_party.py** — Whodunit. LLM seeds 6 suspects with motives, alibis,
  and one secret killer; you interrogate any of them. Each suspect has its own
  message history; the killer is instructed to lie. *Pattern: per-agent memory,
  hidden ground truth.*
- **council.py** — User poses a dilemma; five archetypes (Warrior, Sage,
  Trickster, Healer, Merchant) each give counsel, then a Moderator persona
  synthesizes. *Pattern: fan-out + reduce.*
- **standup.py** — Three engineer personas (jaded senior, eager intern,
  ex-FAANG cynic) give a standup on a project you describe, with conflicting
  blockers. *Pattern: contrasting voices on shared state.*

## Stateful / persistent
- **pen_pal.py** — A persona writes you a letter each run; prior letters load
  from disk so it remembers and the "relationship" arcs over time.
  *Pattern: long-horizon memory via files.*
- **dream_journal.py** — You log a dream; an oneirocritic persona interprets
  it and references past entries to spot recurring symbols.
  *Pattern: lightweight RAG over a journal — would exercise the embeddings
  TODO.*

## Adversarial / educational
- **prompt_injection_dojo.py** — A guard persona holds a secret; you try to
  exfiltrate it across rounds. Score per success.
  *Pattern: red-team sandbox; demonstrates that system prompts are not a
  security boundary.*
- **lie_detector.py** — User submits two-truths-and-a-lie; LLM reasons about
  hedge words and rhythm to pick the lie. Reveal afterward.
  *Pattern: structured analysis output.*

## Single-shot but stylish
- **code_roaster.py** — Companion to `code_archaeologist.py`. Brutally funny
  senior dev roasts your code, with real critique buried in the burns.
  *Pattern: single-pass with strong persona.*
- **time_tutor.py** — Ask any question; answered by a tutor from a chosen era
  (1450 monk, 1880 telegraphist, 2150 AI-rights activist), constrained to that
  era's knowledge. *Pattern: constraint-via-persona.*
- **vibe_translator.py** — Translate text across registers (corporate ↔ pirate
  ↔ Shakespearean ↔ Gen-Z) with optional multi-hop chains.
  *Pattern: pipelining LLM calls.*

## Interactive games
- **cipher_school.py** — LLM presents a classical cipher and ciphertext; you
  guess; it grades and walks through the cryptanalysis.
  *Pattern: tutor loop with verifiable answers.*
- **yes_and.py** — Strict improv partner: never negates, always escalates.
  You feed one line; it builds the scene until you say "blackout".
  *Pattern: rule-bound free-form generation.*

## Tool-calling
- **wiki_sleuth.py** — Reverse 20-questions. You pick any noun; an LLM
  detective has to guess it. The detective alternates between asking you
  yes/no questions and calling `wiki_search(query)` / `wiki_summary(title)`
  tools (Wikipedia REST API, no auth, stdlib `urllib`) to research
  candidates. Each turn may chain multiple tool calls before the next
  user-visible question. *Pattern: agentic tool-use loop via
  `LLMBackend.chat_with_tools()`.*

## Cross-references with TODO.md backend features
- `dream_journal.py` → embeddings
- `mystery_party.py` → `Conversation` wrapper (one per suspect)
- `lie_detector.py` → structured `response_format` output
- `vibe_translator.py` → async fan-out for parallel registers
- `wiki_sleuth.py` → exercises the existing `chat_with_tools()` (already
  built); would be the first toy in the repo to do so
