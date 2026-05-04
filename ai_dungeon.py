"""
AI Dungeon — a procedural terminal text adventure driven by a local LLM.

The LLM acts as dungeon master: it narrates what happens, then appends a JSON
block so we can track health, inventory, exits, and win/lose conditions.

Run:
    python ai_dungeon.py
    python ai_dungeon.py --model mistral
    python ai_dungeon.py --backend llamacpp --model-path ~/models/llama3.gguf

Controls: type any action, 'quit' to exit.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

from llm_backend import LLMBackend, default_backend

# ──────────────────────────────────────────────────────────────── state ──

@dataclass
class GameState:
    health: int = 20
    max_health: int = 20
    gold: int = 0
    inventory: list[str] = field(default_factory=list)
    location: str = "the entrance of a crumbling dungeon"
    turn: int = 0
    game_over: bool = False
    victory: bool = False

    def status(self) -> str:
        filled = "♥" * self.health
        empty = "♡" * (self.max_health - self.health)
        inv = ", ".join(self.inventory) if self.inventory else "nothing"
        return f"  HP {self.health}/{self.max_health} [{filled}{empty}]  Gold: {self.gold}  Carrying: {inv}"

    def apply(self, patch: dict) -> None:
        if (d := patch.get("health_delta")):
            self.health = max(0, min(self.max_health, self.health + d))
        for item in patch.get("add_items") or []:
            self.inventory.append(item)
        for item in patch.get("remove_items") or []:
            if item in self.inventory:
                self.inventory.remove(item)
        if (g := patch.get("gold_delta")):
            self.gold = max(0, self.gold + g)
        if (loc := patch.get("location")):
            self.location = loc
        if patch.get("game_over"):
            self.game_over = True
        if patch.get("victory"):
            self.game_over = True
            self.victory = True

# ──────────────────────────────────────────────────────────────── prompt ─

SYSTEM = """You are the dungeon master of a dark fantasy text adventure.
Narrate what happens when the player acts — atmospheric, tense, 2-4 short paragraphs.

After every narration you MUST append exactly one JSON block (fenced ```json ... ```)
with this schema:
{
  "location":     "brief description of current room",
  "health_delta": <integer — negative=damage, positive=heal, 0=none>,
  "add_items":    ["items the player picks up"],
  "remove_items": ["items used or lost"],
  "gold_delta":   <integer>,
  "exits":        ["available directions/doors the player can go"],
  "game_over":    <true if player dies>,
  "victory":      <true if the player defeats the final boss and escapes>
}

World rules:
- The dungeon has exactly 5 rooms: entrance, library, armory, crypt, boss chamber.
- The boss is a lich named Morthax who guards a stolen sunstone.
- Some items from earlier rooms help against Morthax (the sunstone key, a silver blade).
- Reward creative play and punish recklessness.
- Never break character or acknowledge being an AI."""


def build_msgs(state: GameState, history: list[dict], action: str) -> list[dict]:
    ctx = (
        f"[State — HP:{state.health}/{state.max_health} Gold:{state.gold} "
        f"Inventory:{state.inventory} Location:'{state.location}' Turn:{state.turn}]"
    )
    return (
        [{"role": "system", "content": SYSTEM}]
        + history
        + [{"role": "user", "content": f"{ctx}\n\nAction: {action}"}]
    )

# ──────────────────────────────────────────────────────────────── loop ───

def run(llm: LLMBackend, max_tokens: int, show_reasoning: bool) -> None:
    state = GameState()
    history: list[dict] = []

    print("\n" + "═" * 50)
    print("          AI  DUNGEON  ·  local LLM")
    print("═" * 50)
    print("Type actions to survive. 'quit' to escape.\n")

    action = "Describe where I am and what I can see."
    while True:
        msgs = build_msgs(state, history, action)
        print("\n[Dungeon Master speaks]\n")
        gen = llm.chat(msgs, max_tokens=max_tokens, temperature=0.82, stream=True, show_reasoning=show_reasoning)
        full = llm.stream_print(gen)

        patch = llm.extract_json(full)
        if patch:
            state.apply(patch)
            exits = patch.get("exits") or []
            print(f"\n  Exits: {', '.join(exits) if exits else 'none visible'}")

        print(state.status())
        state.turn += 1

        # keep last 8 assistant/user pairs to preserve context without overflow
        history.append({"role": "assistant", "content": full})
        if len(history) > 16:
            history = history[-16:]

        if state.health <= 0 or state.game_over:
            if state.victory:
                print("\n★  You escape with the sunstone. The dungeon crumbles behind you.")
                print("   Victory after", state.turn, "turns. Gold collected:", state.gold, "★\n")
            else:
                print("\n✝  You have died. The dungeon devours another soul. ✝\n")
            break

        try:
            action = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nYou flee into the night. Farewell.")
            sys.exit(0)

        if not action or action.lower() in ("quit", "exit", "q"):
            print("You retreat. The dungeon waits patiently.")
            sys.exit(0)

        history.append({"role": "user", "content": action})


def main() -> None:
    p = argparse.ArgumentParser(description="AI Dungeon — local LLM text adventure")
    p.add_argument("--max-tokens", type=int, default=2048, metavar="N", help="Max tokens per turn (default 2048)")
    p.add_argument("--hide-reasoning", action="store_true", help="Hide chain-of-thought reasoning output")
    p.add_argument("--backend", choices=["openai", "llamacpp"], default=None)
    p.add_argument("--model", default=None, help="Model name (openai) or path (llamacpp)")
    p.add_argument("--model-path", default=None, help="Path to .gguf file")
    p.add_argument("--base-url", default=None, help="OpenAI-compat server URL")
    args = p.parse_args()

    if args.backend:
        kwargs: dict = {}
        if args.model:
            kwargs["model"] = args.model
        if args.model_path:
            kwargs["model_path"] = args.model_path
        if args.base_url:
            kwargs["base_url"] = args.base_url
        llm = LLMBackend(args.backend, **kwargs)
    else:
        llm = default_backend()

    run(llm, max_tokens=args.max_tokens, show_reasoning=not args.hide_reasoning)


if __name__ == "__main__":
    main()
