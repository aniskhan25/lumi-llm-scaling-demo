#!/usr/bin/env python3
"""Run deterministic-ish before/after inference for canned prompts."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Before/after inference with optional LoRA adapter.")
    parser.add_argument("--base_model", required=True, help="HF model id or local model path")
    parser.add_argument("--adapter_path", default=None, help="Optional PEFT adapter path")
    parser.add_argument("--prompts_file", required=True, help="JSONL prompts file")
    parser.add_argument("--output_file", default=None, help="Optional output JSONL file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=220)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_prompts(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_num}") from exc
            rows.append(row)
    if not rows:
        raise ValueError(f"No prompts found in {path}")
    return rows


def row_to_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    if "messages" in row and isinstance(row["messages"], list):
        return row["messages"]
    prompt = row.get("prompt") or row.get("question")
    if not prompt:
        raise ValueError(f"Prompt row missing expected keys: {row}")
    return [{"role": "user", "content": prompt}]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device is required for this demo script.")

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )

    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model = model.to("cuda")
    model.eval()

    prompts = load_prompts(args.prompts_file)
    output_rows: List[Dict[str, Any]] = []

    for idx, row in enumerate(prompts, start=1):
        row_id = row.get("id", f"row_{idx}")
        messages = row_to_messages(row)
        prompt_text = row.get("prompt") or row.get("question") or "[messages-only prompt]"

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )

        completion_tokens = generated[0, input_ids.shape[1] :]
        response_text = tokenizer.decode(completion_tokens, skip_special_tokens=True).strip()

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {row_id}")
        print(f"PROMPT: {prompt_text}")
        print(f"RESPONSE: {response_text}")
        print("-" * 80)

        output_rows.append(
            {
                "timestamp": ts,
                "id": row_id,
                "prompt": prompt_text,
                "response": response_text,
                "base_model": args.base_model,
                "adapter_path": args.adapter_path,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
            }
        )

    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in output_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved outputs to {out_path}")


if __name__ == "__main__":
    main()
