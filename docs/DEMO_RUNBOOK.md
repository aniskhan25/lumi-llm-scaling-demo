# Demo Runbook (8-10 minutes)

## 0) One-time prep (before event day)

1. Provision environment from [ENVIRONMENT.md](ENVIRONMENT.md).
2. Pre-train and store `adapter_demo` under `artifacts/adapters/adapter_demo`.
3. Generate and save scaling artifacts in `artifacts/`.
4. Capture fallback screenshots and a short recording.

## 1) Session start checks (T-5 min)

```bash
cd /path/to/lumi-llm-scaling-demo
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
export SIF_IMAGE=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif
export VENV_ACTIVATE=/scratch/project_462000131/anisrahm/venvs/myvenv/bin/activate

singularity run "$SIF_IMAGE" python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
```

Expected: Torch version prints and GPU count > 0 in allocated job.

## 2) Before inference (1-2 min)

```bash
singularity run "$SIF_IMAGE" bash -lc '
source "$VENV_ACTIVATE"
python scripts/infer_before_after.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --prompts_file prompts/demo_prompts.jsonl \
  --output_file artifacts/before_outputs.jsonl
'
```

Presenter cue: point out generic answers and weaker policy specificity.

## 3) Live distributed run (2-3 min)

Terminal A (monitoring):

```bash
bash run-scripts/watch_rocm_smi.sh logs/rocm_smi_live.log
```

Terminal B (launch 4-GPU live training):

```bash
# Optional: override writable output location (defaults to SLURM submit dir)
# export RUN_ROOT=/scratch/<project>/<user>/lumi-demo-runs
sbatch run-scripts/run_4gpu.sh
squeue -u $USER
```

Optional 8-GPU (2-node) precompute run:

```bash
sbatch --time=00:45:00 --export=ALL,MAX_STEPS=120 run-scripts/run_2node_8gpu.sh
```

Tail logs after job starts:

```bash
tail -f "${RUN_ROOT:-$PWD}/logs/train_4gpu_rank0.jsonl"
```

Expected within ~30-60s:

- DDP init line with `world_size=4`
- Step logs every 5 steps
- `tokens_per_s` stabilizing after warmup
- `rocm-smi` shows activity on 4 GPUs

## 4) Scaling proof (1-2 min)

```bash
python scripts/parse_logs.py \
  --inputs logs/train_1gpu_rank0.jsonl logs/train_4gpu_rank0.jsonl logs/train_8gpu_rank0.jsonl \
  --output_dir artifacts \
  --warmup_steps 20
```

Show:

- `artifacts/scaling_summary.md`
- `artifacts/scaling_plot.png`

Presenter cue: explain speedup and efficiency, then link to AIF optimization role.

## 5) After inference (1-2 min)

```bash
singularity run "$SIF_IMAGE" bash -lc '
source "$VENV_ACTIVATE"
python scripts/infer_before_after.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --adapter_path artifacts/adapters/adapter_demo \
  --prompts_file prompts/demo_prompts.jsonl \
  --output_file artifacts/after_outputs.jsonl
'
```

Presenter cue: compare same prompts; show improved domain language and policy grounding.

## 6) Troubleshooting quick checks

- ROCm/GPU visibility:
  - `rocm-smi`
  - `python -c "import torch; print(torch.cuda.device_count())"`
- Slow model load:
  - ensure model cached under `HF_HOME` on scratch
  - reduce model to `Qwen/Qwen2.5-3B-Instruct`
- DDP init hang:
  - verify `MASTER_ADDR`, `MASTER_PORT`, `NCCL_SOCKET_IFNAME`
  - verify same container/module stack across nodes
- OOM:
  - lower `--max_length`
  - lower `--micro_batch_size`
  - increase `--grad_accum`

## 7) Fallback ladder

1. Full live: 4-GPU training + live `rocm-smi` + live before/after
2. Semi-live: live inference + precomputed scaling logs/plots
3. Offline: recorded clip + static outputs/screenshots

## 8) Suggested speaking points for AIF value

- Parallel strategy tuning (`1 -> 4 -> 8` GPUs) improves latency-to-result.
- `bf16` on MI250X enables practical throughput with stable memory.
- Data and logging instrumentation make optimization measurable, not anecdotal.
- Profiling/tuning support from AIF shortens iteration time for production deployment.
