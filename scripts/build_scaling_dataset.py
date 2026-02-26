#!/usr/bin/env python3
"""Build a larger deterministic JSONL dataset for scaling runs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


PROMPT_SUFFIXES = [
    "Include exact next actions.",
    "Keep the response under 120 words.",
    "Mention SLA expectations.",
    "Use a neutral enterprise tone.",
    "Add one verification question.",
]

RESPONSE_PREFIXES = [
    "Resolution plan:",
    "Support guidance:",
    "Policy-aligned response:",
    "Action summary:",
    "Recommended handling:",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand a seed JSONL dataset for scaling experiments.")
    parser.add_argument("--input", required=True, help="Input JSONL with prompt/response rows")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--size", type=int, default=5000, help="Number of rows to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic expansion")
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_num}") from exc
            prompt = row.get("prompt") or row.get("question")
            response = row.get("response") or row.get("answer")
            if not prompt or not response:
                raise ValueError(f"Row missing prompt/response keys at {path}:{line_num}")
            rows.append({"prompt": str(prompt), "response": str(response)})
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def make_row(base: Dict[str, str], idx: int, rng: random.Random) -> Dict[str, str]:
    case_id = f"CASE-{idx:05d}"
    hour = idx % 24
    minute = (idx * 7) % 60
    prompt_suffix = rng.choice(PROMPT_SUFFIXES)
    response_prefix = rng.choice(RESPONSE_PREFIXES)

    prompt = (
        f"[{case_id}] {base['prompt']} "
        f"Context: region=EU-West, priority=P{1 + (idx % 3)}, timestamp={hour:02d}:{minute:02d} UTC. "
        f"{prompt_suffix}"
    )
    response = f"{response_prefix} {base['response']} (reference={case_id})"
    return {"prompt": prompt, "response": response}


def main() -> None:
    args = parse_args()
    if args.size <= 0:
        raise ValueError("--size must be > 0")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seed_rows = load_rows(input_path)
    rng = random.Random(args.seed)

    with output_path.open("w", encoding="utf-8") as f:
        for idx in range(args.size):
            base = seed_rows[idx % len(seed_rows)]
            row = make_row(base, idx=idx + 1, rng=rng)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {args.size} rows to {output_path}")


if __name__ == "__main__":
    main()

