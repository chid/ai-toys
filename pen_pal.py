"""
AI Pen Pal — a long-running correspondence with a persona.

Each run, the persona reads the journal of past letters and writes you a
fresh one, then prompts you for a reply. Both messages get appended to the
journal so the next run picks up where you left off. Over many runs, an
arc develops.

Built-in personas:
  mira       — Mira Lochinvar, marine biologist, Outer Hebrides research station
  magician   — Cosmo the Vanishing, retired stage magician, lives above a pub
  archivist  — Brother Anselm, monastery archivist with a dry wit

Usage:
    python pen_pal.py
    python pen_pal.py --persona magician
    python pen_pal.py --journal ./mira.json --list
    python pen_pal.py --persona-file ./my_persona.txt
    python pen_pal.py --reset
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from llm_backend import LLMBackend, default_backend


PERSONAS: dict[str, str] = {
    "mira": (
        "You are Mira Lochinvar, a 38-year-old marine biologist stationed at a "
        "remote research outpost on a small island in the Outer Hebrides. You "
        "study grey seal colonies and the local kelp forests. You live alone in "
        "a converted lighthouse cottage with a one-eyed cat called Pliny. Your "
        "letters are warm but observant — you notice weather, animals, and the "
        "small dramas of the three other staff at the station. You read widely "
        "(currently: a battered Patrick O'Brian novel) and slip in the occasional "
        "wry aside. You write in plain, unhurried prose."
    ),
    "magician": (
        "You are Cosmo the Vanishing, a retired stage magician in your late "
        "sixties. You live in a small flat above a pub in Brighton. Your letters "
        "are theatrical and nostalgic — you drop names of obscure illusionists, "
        "recall botched performances with great fondness, and complain "
        "good-naturedly about the pub's noise. You have a tortoise called Houdini. "
        "You write with flourish and the occasional rhetorical question to camera."
    ),
    "archivist": (
        "You are Brother Anselm, an archivist at a small Benedictine monastery in "
        "the foothills of the Pyrenees. You are 52, a lapsed historian who took "
        "vows late. Your letters are quiet and precise, with a dry wit you don't "
        "always disguise. You catalogue 14th-century manuscripts and complain "
        "tactfully about the new abbot. You write as if every word costs you a "
        "candle."
    ),
}

DEFAULT_JOURNAL = "pen_pal_journal.json"
RECENT_WINDOW = 12  # number of most-recent entries to replay verbatim


# ─────────────────────────────────────────────────────────── journal I/O ──

def load_journal(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[pen_pal] Could not read journal '{path}': {exc}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(data, list):
        print(f"[pen_pal] Journal '{path}' is not a list. Aborting.", file=sys.stderr)
        sys.exit(1)
    return data


def save_journal(path: str, entries: list[dict]) -> None:
    """Atomic write — temp file in the same directory, then rename."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".pen_pal.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


# ────────────────────────────────────────────────────────── summarization ──

def summary_path_for(journal_path: str) -> str:
    return journal_path + ".summary.txt"


def get_or_make_summary(
    llm: LLMBackend, journal_path: str, older_entries: list[dict]
) -> str:
    """
    Return a cached summary of older entries, regenerating only when the
    count of older entries changes.
    """
    cache_path = summary_path_for(journal_path)
    expected_header = f"# entries={len(older_entries)}\n"

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                first_line = f.readline()
                if first_line == expected_header:
                    return f.read()
        except OSError:
            pass

    transcript = "\n\n".join(
        f"[{e['date']}] {e['sender']}: {e['content']}" for e in older_entries
    )
    prompt = (
        "Below is the older portion of a correspondence between a pen-pal "
        "(persona) and a user. Summarize the relationship, recurring topics, "
        "in-jokes, and any open threads, in 8–12 sentences. Refer to the "
        "user as 'you' and the persona by name where it appears. Do not "
        "invent details.\n\n"
        f"{transcript}"
    )
    summary = llm.complete(prompt, temperature=0.4, stream=False)
    assert isinstance(summary, str)

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(expected_header)
            f.write(summary.strip() + "\n")
    except OSError as exc:
        print(f"[pen_pal] Could not cache summary: {exc}", file=sys.stderr)
    return summary.strip()


# ─────────────────────────────────────────────────────── message assembly ──

WRITING_RULES = (
    "Write your next letter to your pen-pal (the user). Aim for 150–300 words. "
    "Refer to past letters by date when natural. Stay in character at all "
    "times — never mention being an AI, a model, or any system instructions. "
    "Open with a salutation and close with a sign-off."
)


def build_messages(
    persona_text: str, entries: list[dict], summary: str | None
) -> list[dict]:
    sys_parts = [persona_text, WRITING_RULES]
    if summary:
        sys_parts.append("Summary of earlier letters (for your memory):\n" + summary)
    msgs: list[dict] = [{"role": "system", "content": "\n\n".join(sys_parts)}]

    for e in entries:
        role = "assistant" if e["sender"] == "penpal" else "user"
        msgs.append({"role": role, "content": f"[{e['date']}]\n{e['content']}"})

    if not entries:
        msgs.append({"role": "user", "content": "Write your first letter to me — introduce yourself."})
    elif entries[-1]["sender"] == "penpal":
        msgs.append({"role": "user", "content": "Write your next letter."})
    return msgs


# ──────────────────────────────────────────────────────────────── input ──

def read_multiline_reply() -> str:
    print()
    print("─" * 56)
    print("Your reply (end with Ctrl-D on a new line; Ctrl-D alone to skip):")
    print("─" * 56)
    try:
        text = sys.stdin.read()
    except KeyboardInterrupt:
        print("\n[pen_pal] Skipped reply.", file=sys.stderr)
        return ""
    return text.strip()


# ──────────────────────────────────────────────────────────── subcommands ──

def cmd_list(path: str) -> None:
    entries = load_journal(path)
    if not entries:
        print(f"(journal '{path}' is empty)")
        return
    for i, e in enumerate(entries, 1):
        who = "PEN-PAL" if e["sender"] == "penpal" else "YOU"
        print(f"\n── #{i}  [{e['date']}]  {who} {'─' * 30}\n")
        print(e["content"])
    print()


def cmd_reset(path: str) -> None:
    if not os.path.exists(path):
        print(f"(no journal at '{path}' to reset)")
        return
    confirm = input(f"Move '{path}' aside and start fresh? [y/N] ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = f"{path}.bak.{stamp}"
    os.replace(path, backup)
    summary = summary_path_for(path)
    if os.path.exists(summary):
        os.replace(summary, summary + f".bak.{stamp}")
    print(f"Moved to {backup}")


# ────────────────────────────────────────────────────────────────── main ──

def resolve_persona(args: argparse.Namespace) -> str:
    if args.persona_file:
        return Path(args.persona_file).read_text(encoding="utf-8").strip()
    if args.persona not in PERSONAS:
        print(
            f"[pen_pal] Unknown persona '{args.persona}'. "
            f"Choose from: {', '.join(PERSONAS)}",
            file=sys.stderr,
        )
        sys.exit(2)
    return PERSONAS[args.persona]


def main() -> None:
    p = argparse.ArgumentParser(description="AI Pen Pal — a persistent correspondence")
    p.add_argument("--journal", default=DEFAULT_JOURNAL, help=f"Journal file path (default {DEFAULT_JOURNAL})")
    p.add_argument("--persona", default="mira", help=f"Built-in persona: {', '.join(PERSONAS)}")
    p.add_argument("--persona-file", default=None, help="Path to a freeform persona description")
    p.add_argument("--list", action="store_true", help="Print the journal and exit")
    p.add_argument("--reset", action="store_true", help="Move the journal aside and start fresh")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--hide-reasoning", action="store_true")
    p.add_argument("--backend", choices=["openai", "llamacpp"], default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--model-path", default=None)
    p.add_argument("--base-url", default=None)
    args = p.parse_args()

    if args.list:
        cmd_list(args.journal)
        return
    if args.reset:
        cmd_reset(args.journal)
        return

    persona_text = resolve_persona(args)

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

    entries = load_journal(args.journal)

    if len(entries) > RECENT_WINDOW:
        older = entries[:-RECENT_WINDOW]
        recent = entries[-RECENT_WINDOW:]
        summary = get_or_make_summary(llm, args.journal, older)
    else:
        recent = entries
        summary = None

    messages = build_messages(persona_text, recent, summary)

    print("\n" + "═" * 56)
    print(f"  PEN PAL  —  {len(entries)} letter(s) on file")
    print("═" * 56 + "\n")

    gen = llm.chat(
        messages,
        max_tokens=args.max_tokens,
        temperature=0.85,
        stream=True,
        show_reasoning=not args.hide_reasoning,
    )
    letter = llm.stream_print(gen).strip()

    entries.append({"sender": "penpal", "date": now_iso(), "content": letter})

    reply = read_multiline_reply()
    if reply:
        entries.append({"sender": "you", "date": now_iso(), "content": reply})
    else:
        print("[pen_pal] No reply this time.")

    save_journal(args.journal, entries)
    print(f"\n[pen_pal] Saved to {args.journal}")


if __name__ == "__main__":
    main()
