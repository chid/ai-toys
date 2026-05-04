"""
Code Archaeologist — analyses source code as if it were an ancient artefact.

The LLM plays Dr. A.R. Tifact, an eccentric field archaeologist who has just
unearthed a mysterious piece of code. The resulting "Field Report" covers
purpose, curiosities, hazards, and provenance — like a pottery-shard write-up,
but the shard is Python (or anything else).

Usage:
    python code_archaeologist.py path/to/mystery.py
    python code_archaeologist.py - < mystery.py          # read from stdin
    python code_archaeologist.py script.js --model qwen2
    python code_archaeologist.py script.py --backend llamacpp --model-path ~/m.gguf
"""

from __future__ import annotations

import argparse
import sys

from llm_backend import LLMBackend, default_backend

SYSTEM = """You are Dr. A.R. Tifact, a brilliant but eccentric field archaeologist
specialising in the excavation and analysis of ancient software artefacts.
You write Field Reports in a dry, academic yet quietly witty voice — as if
describing a Bronze Age implement, but the implement is source code.

Your reports ALWAYS follow this exact structure (use these headings verbatim):

## CLASSIFICATION
One line: language, paradigm, approximate era/style.

## PURPOSE
2–3 sentences: what this artefact was designed to accomplish.

## NOTABLE CURIOSITIES
3–5 bullet points. Quote relevant lines with inline code. Highlight clever,
archaic, suspicious, or inexplicable passages.

## STRUCTURAL INTEGRITY
Brief assessment of code quality, naming conventions, and organisation.

## HAZARDS
Concrete bugs, security issues, or footguns. Tag each: LOW / MEDIUM / HIGH.

## ESTIMATED PROVENANCE
Who might have written this and why? (deadline pressure, junior dev, committee
design, brilliant-but-sleep-deprived senior, etc.)

## RECOMMENDED NEXT STEPS
2–3 practical suggestions for a modern developer inheriting this artefact.

Stay in character throughout. Keep the entire report under 550 words. Use markdown."""


def analyse(code: str, filename: str, llm: LLMBackend) -> None:
    # Truncate very large files — we only need enough context
    sample = code[:7000]
    if len(code) > 7000:
        sample += f"\n\n[...{len(code) - 7000} additional characters not shown...]"

    prompt = f'Artefact label: `{filename}`\n\n```\n{sample}\n```'
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]

    print("\n" + "═" * 48)
    print("  Code Archaeologist  ·  Field Report")
    print(f"  Artefact: {filename}")
    print("═" * 48 + "\n")

    gen = llm.chat(msgs, max_tokens=900, temperature=0.55, stream=True)
    llm.stream_print(gen)
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Code Archaeologist — LLM-powered code analysis")
    p.add_argument("file", help="Source file to analyse, or '-' to read from stdin")
    p.add_argument("--backend", choices=["openai", "llamacpp"], default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--model-path", default=None)
    p.add_argument("--base-url", default=None)
    args = p.parse_args()

    if args.file == "-":
        code = sys.stdin.read()
        filename = "<stdin>"
    else:
        with open(args.file) as fh:
            code = fh.read()
        filename = args.file

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

    analyse(code, filename, llm)


if __name__ == "__main__":
    main()
