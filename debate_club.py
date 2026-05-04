"""
AI Debate Club — two contrasting AI personas argue opposite sides of a topic.

Each persona has a distinct voice and rhetorical style. They alternate for N rounds,
each building on the previous argument. You decide who wins.

Built-in personas (FOR vs AGAINST):
  Maximilian  — a flamboyant Victorian polymath, florid prose, Latin phrases
  Ren         — a terse Silicon Valley pragmatist, bullet points, startup jargon

Usage:
    python debate_club.py "Is coffee better than tea?"
    python debate_club.py "Python vs Rust" --rounds 4
    python debate_club.py "Remote work is better" --model llama3.1
    python debate_club.py "AI will replace programmers" --backend llamacpp --model-path ~/m.gguf
"""

from __future__ import annotations

import argparse
import sys

from llm_backend import LLMBackend, default_backend

PERSONAS: dict[str, dict] = {
    "Maximilian": {
        "desc": "a flamboyant Victorian gentleman polymath",
        "side": "FOR",
        "style": (
            "theatrical and verbose, peppers speech with Latin phrases, "
            "uses elaborate metaphors, supremely self-assured, "
            "opens with a dramatic flourish"
        ),
    },
    "Ren": {
        "desc": "a terse Silicon Valley pragmatist",
        "side": "AGAINST",
        "style": (
            "data-driven and clipped, loves bullet points and percentages, "
            "sprinkles in startup jargon, dismisses sentiment as 'not scalable', "
            "occasionally drops a deadpan quip"
        ),
    },
}


def make_system(name: str, topic: str) -> str:
    p = PERSONAS[name]
    return (
        f"You are {name}, {p['desc']}. "
        f"You are arguing {p['side']} the proposition: \"{topic}\". "
        f"Rhetorical style: {p['style']}. "
        "Rules: keep each turn to 4–6 sentences. Be sharp and entertaining. "
        "You may acknowledge a minor sub-point only to immediately dismantle it. "
        "Never concede the core proposition. Never break character or mention AI."
    )


def debate(topic: str, rounds: int, llm: LLMBackend, max_tokens: int, show_reasoning: bool) -> None:
    names = list(PERSONAS.keys())
    histories: dict[str, list[dict]] = {n: [] for n in names}

    print("\n" + "═" * 56)
    print("  AI DEBATE CLUB")
    print(f"  Topic:  {topic}")
    print(f"  {names[0]} (FOR)   vs   {names[1]} (AGAINST)")
    print("═" * 56 + "\n")

    last_argument = f'The proposition is: "{topic}". Deliver your opening statement.'

    for rnd in range(1, rounds + 1):
        print(f"── Round {rnd} {'─' * 46}\n")

        for i, name in enumerate(names):
            opponent = names[1 - i]
            side = PERSONAS[name]["side"]

            if rnd == 1 and i == 0:
                prompt = last_argument
            else:
                prompt = (
                    f"{opponent} just argued:\n\n\"{last_argument}\"\n\n"
                    "Respond decisively."
                )

            histories[name].append({"role": "user", "content": prompt})
            # keep context window lean: last 6 turns per persona
            msgs = [{"role": "system", "content": make_system(name, topic)}] + histories[name][-6:]

            print(f"[{name} — {side}]")
            gen = llm.chat(msgs, max_tokens=max_tokens, temperature=0.88, stream=True, show_reasoning=show_reasoning)
            response = llm.stream_print(gen)

            histories[name].append({"role": "assistant", "content": response})
            last_argument = response
            print()

    print("═" * 56)
    print("  The debate concludes. You decide the winner.\n")


def main() -> None:
    p = argparse.ArgumentParser(description="AI Debate Club — two personas, one topic")
    p.add_argument("topic", help="Proposition to debate")
    p.add_argument("--rounds", type=int, default=3, metavar="N", help="Number of rounds (default 3)")
    p.add_argument("--max-tokens", type=int, default=2048, metavar="N", help="Max tokens per turn (default 2048)")
    p.add_argument("--hide-reasoning", action="store_true", help="Hide chain-of-thought reasoning output")
    p.add_argument("--backend", choices=["openai", "llamacpp"], default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--model-path", default=None)
    p.add_argument("--base-url", default=None)
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

    debate(args.topic, args.rounds, llm, max_tokens=args.max_tokens, show_reasoning=not args.hide_reasoning)


if __name__ == "__main__":
    main()
