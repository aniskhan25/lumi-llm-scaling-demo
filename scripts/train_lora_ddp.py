#!/usr/bin/env python3
"""Distributed LoRA training for short live demos on ROCm (LUMI)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc


@dataclass
class DistInfo:
    is_distributed: bool
    rank: int
    local_rank: int
    world_size: int


def parse_pre_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    return pre.parse_known_args()[0]


def load_config_defaults(config_path: str | None) -> Dict[str, Any]:
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must map keys to values: {config_path}")
    return loaded


def build_arg_parser(defaults: Dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoRA DDP trainer for short LUMI demos")
    parser.add_argument("--config", type=str, default=defaults.get("config"))
    parser.add_argument("--base_model", type=str, default=defaults.get("base_model", "Qwen/Qwen2.5-7B-Instruct"))
    parser.add_argument("--dataset_path", type=str, default=defaults.get("dataset_path", "data/demo_support_subset.sample.jsonl"))
    parser.add_argument("--output_dir", type=str, default=defaults.get("output_dir", "artifacts/adapters/adapter_live"))
    parser.add_argument("--log_file", type=str, default=defaults.get("log_file", "logs/train_rank0.jsonl"))

    parser.add_argument("--seed", type=int, default=int(defaults.get("seed", 42)))
    parser.add_argument("--max_steps", type=int, default=int(defaults.get("max_steps", 150)))
    parser.add_argument("--max_length", type=int, default=int(defaults.get("max_length", 1024)))
    parser.add_argument("--learning_rate", type=float, default=float(defaults.get("learning_rate", 2e-4)))
    parser.add_argument("--weight_decay", type=float, default=float(defaults.get("weight_decay", 0.01)))
    parser.add_argument("--warmup_steps", type=int, default=int(defaults.get("warmup_steps", 20)))
    parser.add_argument("--max_grad_norm", type=float, default=float(defaults.get("max_grad_norm", 1.0)))

    parser.add_argument("--micro_batch_size", type=int, default=int(defaults.get("micro_batch_size", 2)))
    parser.add_argument("--grad_accum", type=int, default=int(defaults.get("grad_accum", 8)))
    parser.add_argument("--num_workers", type=int, default=int(defaults.get("num_workers", 2)))

    parser.add_argument("--logging_steps", type=int, default=int(defaults.get("logging_steps", 5)))
    parser.add_argument("--save_steps", type=int, default=int(defaults.get("save_steps", 50)))
    parser.add_argument("--warmup_log_steps", type=int, default=int(defaults.get("warmup_log_steps", 20)))
    parser.add_argument("--steady_state_window", type=int, default=int(defaults.get("steady_state_window", 30)))

    parser.add_argument("--lora_r", type=int, default=int(defaults.get("lora_r", 16)))
    parser.add_argument("--lora_alpha", type=int, default=int(defaults.get("lora_alpha", 32)))
    parser.add_argument("--lora_dropout", type=float, default=float(defaults.get("lora_dropout", 0.05)))
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=defaults.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    parser.add_argument("--trust_remote_code", action="store_true", default=bool(defaults.get("trust_remote_code", False)))
    return parser


def parse_args() -> argparse.Namespace:
    pre_args = parse_pre_args()
    defaults = load_config_defaults(pre_args.config)
    parser = build_arg_parser(defaults)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def init_distributed() -> DistInfo:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        init_kwargs = {"backend": "nccl", "timeout": dt.timedelta(minutes=30)}
        try:
            dist.init_process_group(**init_kwargs, device_id=torch.device("cuda", local_rank))
        except TypeError:
            dist.init_process_group(**init_kwargs)
        return DistInfo(True, rank, local_rank, world_size)
    torch.cuda.set_device(0)
    return DistInfo(False, 0, 0, 1)


def distributed_barrier(info: DistInfo) -> None:
    if not info.is_distributed or not dist.is_initialized():
        return
    try:
        dist.barrier(device_ids=[info.local_rank])
    except TypeError:
        dist.barrier()


def cleanup_distributed(info: DistInfo) -> None:
    if info.is_distributed and dist.is_initialized():
        distributed_barrier(info)
        dist.destroy_process_group()


def rank0_print(info: DistInfo, msg: str) -> None:
    if info.rank == 0:
        print(msg, flush=True)


def jsonl_log(log_path: Path, payload: Dict[str, Any]) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def normalize_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    if "messages" in row and isinstance(row["messages"], list):
        return row["messages"]
    prompt = row.get("prompt") or row.get("question")
    answer = row.get("response") or row.get("answer")
    if not prompt or not answer:
        raise ValueError(f"Unsupported sample format: {row}")
    return [
        {"role": "user", "content": str(prompt)},
        {"role": "assistant", "content": str(answer)},
    ]


class JsonlCausalLMDataset(Dataset):
    def __init__(self, dataset_path: str, tokenizer: AutoTokenizer, max_length: int) -> None:
        self.samples: List[Dict[str, torch.Tensor]] = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {dataset_path}:{line_num}") from exc

                messages = normalize_messages(row)
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                enc = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].squeeze(0)
                attention_mask = enc["attention_mask"].squeeze(0)
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100
                self.samples.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                    }
                )

        if not self.samples:
            raise ValueError(f"No samples loaded from {dataset_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = ["input_ids", "attention_mask", "labels"]
    return {k: torch.stack([x[k] for x in batch], dim=0) for k in keys}


def cycle(loader: Iterable[Dict[str, torch.Tensor]]) -> Iterable[Dict[str, torch.Tensor]]:
    while True:
        for item in loader:
            yield item


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def all_reduce_sum(value: torch.Tensor, info: DistInfo) -> torch.Tensor:
    if info.is_distributed:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device is required for distributed training.")

    info = init_distributed()
    device = torch.device("cuda", info.local_rank)

    log_path = Path(args.log_file)
    output_dir = Path(args.output_dir)
    if info.rank == 0:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        if log_path.exists():
            log_path.unlink()

    if info.is_distributed:
        distributed_barrier(info)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = JsonlCausalLMDataset(args.dataset_path, tokenizer=tokenizer, max_length=args.max_length)
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed) if info.is_distributed else None

    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, lora_config)
    model = model.to(device)

    if info.is_distributed:
        model = DDP(model, device_ids=[info.local_rank], output_device=info.local_rank, find_unused_parameters=False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    rank0_print(info, f"rank={info.rank} local_rank={info.local_rank} world_size={info.world_size}")
    rank0_print(info, f"dataset_size={len(dataset)} micro_batch_size={args.micro_batch_size} grad_accum={args.grad_accum}")
    rank0_print(info, f"base_model={args.base_model} output_dir={args.output_dir}")

    if info.rank == 0:
        jsonl_log(
            log_path,
            {
                "event": "run_start",
                "time": utc_now_iso(),
                "rank": info.rank,
                "world_size": info.world_size,
                "base_model": args.base_model,
                "dataset_size": len(dataset),
                "micro_batch_size": args.micro_batch_size,
                "grad_accum": args.grad_accum,
                "max_steps": args.max_steps,
            },
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loader_it = cycle(loader)

    global_step = 0
    accum_idx = 0
    tokens_accum_local = 0
    step_start_time = time.perf_counter()

    step_history: List[Tuple[float, float]] = []

    while global_step < args.max_steps:
        if sampler is not None and accum_idx == 0 and global_step % max(1, len(loader)) == 0:
            sampler.set_epoch(global_step)

        batch = next(loader_it)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        tokens_accum_local += int(batch["attention_mask"].sum().item())

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss / args.grad_accum

        loss.backward()
        accum_idx += 1

        if accum_idx < args.grad_accum:
            continue

        torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        accum_idx = 0
        global_step += 1

        step_end = time.perf_counter()
        step_time_s = step_end - step_start_time
        step_start_time = step_end

        tokens_tensor = torch.tensor(float(tokens_accum_local), device=device)
        tokens_global = all_reduce_sum(tokens_tensor, info).item()
        tokens_accum_local = 0
        tokens_per_s = tokens_global / max(step_time_s, 1e-6)

        lr = scheduler.get_last_lr()[0]
        loss_value = float(loss.item() * args.grad_accum)
        step_history.append((step_time_s, tokens_per_s))

        if info.rank == 0 and (global_step % args.logging_steps == 0 or global_step == 1):
            payload = {
                "event": "step",
                "time": utc_now_iso(),
                "step": global_step,
                "world_size": info.world_size,
                "loss": round(loss_value, 6),
                "lr": lr,
                "step_time_s": round(step_time_s, 6),
                "tokens_per_s": round(tokens_per_s, 2),
            }
            print(payload, flush=True)
            jsonl_log(log_path, payload)

        if info.rank == 0 and global_step % args.save_steps == 0:
            ckpt_dir = output_dir / f"step_{global_step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            unwrap_model(model).save_pretrained(ckpt_dir)

    if info.rank == 0:
        unwrap_model(model).save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        steady = step_history[args.warmup_log_steps :]
        if args.steady_state_window > 0:
            steady = steady[-args.steady_state_window :]
        avg_step = sum(x[0] for x in steady) / max(len(steady), 1)
        avg_tps = sum(x[1] for x in steady) / max(len(steady), 1)

        summary = {
            "event": "run_summary",
            "time": utc_now_iso(),
            "world_size": info.world_size,
            "steps": global_step,
            "avg_step_time_s": round(avg_step, 6),
            "avg_tokens_per_s": round(avg_tps, 2),
            "warmup_log_steps": args.warmup_log_steps,
            "steady_state_window": args.steady_state_window,
            "output_dir": str(output_dir),
        }
        print(summary, flush=True)
        jsonl_log(log_path, summary)

    cleanup_distributed(info)


if __name__ == "__main__":
    main()
