#!/usr/bin/env python3
"""Parse training JSONL logs and generate scaling summary artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create scaling table/plot from training logs")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input rank0 JSONL logs")
    parser.add_argument("--output_dir", default="artifacts", help="Output directory")
    parser.add_argument("--warmup_steps", type=int, default=20, help="Ignore steps <= warmup_steps")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_num}") from exc
    return rows


def summarize_run(rows: List[Dict[str, Any]], warmup_steps: int) -> Dict[str, Any]:
    step_rows = [r for r in rows if r.get("event") == "step" and int(r.get("step", 0)) > warmup_steps]
    if not step_rows:
        step_rows = [r for r in rows if r.get("event") == "step"]
    if not step_rows:
        raise ValueError("No step rows found")

    world_size = int(step_rows[0].get("world_size", 1))
    avg_step_time = sum(float(r["step_time_s"]) for r in step_rows) / len(step_rows)
    avg_tps = sum(float(r["tokens_per_s"]) for r in step_rows) / len(step_rows)

    return {
        "world_size": world_size,
        "avg_step_time_s": avg_step_time,
        "avg_tokens_per_s": avg_tps,
        "num_points": len(step_rows),
    }


def markdown_table(rows: List[Dict[str, Any]]) -> str:
    header = "| world_size | avg_step_time_s | avg_tokens_per_s | speedup | efficiency | points |\n"
    sep = "|---:|---:|---:|---:|---:|---:|\n"
    body = ""
    for r in rows:
        body += (
            f"| {r['world_size']} | {r['avg_step_time_s']:.4f} | {r['avg_tokens_per_s']:.2f} "
            f"| {r['speedup']:.2f} | {r['efficiency']:.3f} | {r['num_points']} |\n"
        )
    return header + sep + body


def maybe_plot(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    x = [int(r["world_size"]) for r in rows]
    y = [float(r["avg_tokens_per_s"]) for r in rows]

    plt.figure(figsize=(7, 4))
    plt.plot(x, y, marker="o", linewidth=2)
    plt.xticks(x)
    plt.xlabel("World Size (GPUs)")
    plt.ylabel("Average Tokens / sec")
    plt.title("LUMI LoRA Throughput Scaling")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "scaling_plot.png", dpi=160)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    for input_path in args.inputs:
        rows = load_jsonl(Path(input_path))
        summary = summarize_run(rows, warmup_steps=args.warmup_steps)
        summary["source_log"] = input_path
        summaries.append(summary)

    summaries.sort(key=lambda x: int(x["world_size"]))
    baseline_tps = summaries[0]["avg_tokens_per_s"]

    for r in summaries:
        r["speedup"] = r["avg_tokens_per_s"] / max(baseline_tps, 1e-9)
        r["efficiency"] = r["speedup"] / max(float(r["world_size"]), 1e-9)

    csv_path = output_dir / "scaling_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "world_size",
                "avg_step_time_s",
                "avg_tokens_per_s",
                "speedup",
                "efficiency",
                "num_points",
                "source_log",
            ],
        )
        writer.writeheader()
        writer.writerows(summaries)

    md_path = output_dir / "scaling_summary.md"
    md_path.write_text(markdown_table(summaries), encoding="utf-8")

    maybe_plot(summaries, output_dir)

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    plot_path = output_dir / "scaling_plot.png"
    if plot_path.exists():
        print(f"Wrote {plot_path}")
    else:
        print("Skipped plot generation (matplotlib unavailable).")


if __name__ == "__main__":
    main()
