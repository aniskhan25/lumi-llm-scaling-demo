# Demo Runbook (Simple + Reproducible)

## Demo modes

- **Mode A (recommended):** precomputed 1/4/8 scaling + live 4-GPU run + optional before/after inference.
- **Mode B (safe fallback):** precomputed scaling + precomputed before/after outputs only.
- **Mode C (semi-live async):** submit with `sbatch`, poll status, then show generated outputs when done.

## 0) Fixed paths/env

```bash
cd /scratch/project_462000131/$USER/lumi-llm-scaling-demo
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SIF_IMAGE=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif
export VENV_ACTIVATE=/project/project_462000131/$USER/venvs/myvenv/bin/activate
```

GPU sanity check (inside allocation):

```bash
srun --ntasks=1 --gpus=1 singularity exec "$SIF_IMAGE" bash -lc '
source "$VENV_ACTIVATE"
python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
'
```

## 1) One-time precompute

Build scaling dataset:

```bash
python scripts/build_scaling_dataset.py \
  --input data/demo_support_subset.sample.jsonl \
  --output data/demo_support_subset.scaling_5000.jsonl \
  --size 5000 \
  --seed 42
```

Run scaling jobs:

```bash
sbatch --time=00:45:00 --export=ALL,TRAIN_CONFIG=configs/train_lora_scaling.yaml,MAX_STEPS=140 run-scripts/run_1gpu.sh
sbatch --time=00:55:00 --export=ALL,TRAIN_CONFIG=configs/train_lora_scaling.yaml,MAX_STEPS=140 run-scripts/run_4gpu.sh
sbatch --time=00:55:00 --export=ALL,TRAIN_CONFIG=configs/train_lora_scaling.yaml,MAX_STEPS=140 run-scripts/run_2node_8gpu.sh
```

Generate final scaling artifacts:

```bash
python scripts/parse_logs.py \
  --inputs logs/train_1gpu_rank0.jsonl logs/train_4gpu_rank0.jsonl logs/train_8gpu_rank0.jsonl \
  --output_dir artifacts \
  --warmup_steps 30
```

Expected artifacts:

- `artifacts/scaling_summary.md`
- `artifacts/scaling_summary.csv`
- `artifacts/scaling_plot.png`

## 2) Live demo flow (Mode A)

### Terminal 1: submit live 4-GPU run

```bash
sbatch --export=ALL,TRAIN_CONFIG=configs/train_lora_scaling.yaml,MAX_STEPS=140 run-scripts/run_4gpu.sh
squeue -u $USER
```

### Terminal 2: monitor rocm-smi on the same running job

```bash
JOBID=<jobid_from_sbatch>
NODE=$(squeue -j "$JOBID" -h -o %N)

srun --overlap --jobid="$JOBID" -w "$NODE" --ntasks=1 \
  bash run-scripts/watch_rocm_smi.sh logs/rocm_smi_live.log 1
```

### Terminal 1: tail training log

```bash
tail -f logs/train_4gpu_rank0.jsonl
```

What to point out:

- `world_size=4`
- recurring step logs with `step_time_s` and `tokens_per_s`
- GPU activity visible from `rocm-smi`

### Show precomputed scaling result

Open and present:

- `artifacts/scaling_summary.md`
- `artifacts/scaling_plot.png`

## 3) Optional quality demo (before/after inference)

Before:

```bash
srun --ntasks=1 --gpus=1 singularity exec "$SIF_IMAGE" bash -lc '
source "$VENV_ACTIVATE"
python scripts/infer_before_after.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --prompts_file prompts/demo_prompts.jsonl \
  --output_file artifacts/before_outputs.jsonl
'
```

After (adapter must exist):

```bash
test -f artifacts/adapters/adapter_demo/adapter_config.json

srun --ntasks=1 --gpus=1 singularity exec "$SIF_IMAGE" bash -lc '
source "$VENV_ACTIVATE"
python scripts/infer_before_after.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --adapter_path artifacts/adapters/adapter_demo \
  --prompts_file prompts/demo_prompts.jsonl \
  --output_file artifacts/after_outputs.jsonl
'
```

## 4) Fallback ladder

1. Full live (Mode A): live 4-GPU + live `rocm-smi` + precomputed scaling.
2. Semi-live (Mode C): submit job live, show queue/start, then show precomputed outputs.
3. Offline (Mode B): precomputed logs/plots + precomputed before/after outputs.

## 5) Fast troubleshooting

- No GPUs visible: run commands with `srun --ntasks=1 --gpus=1 ...`.
- `amdgpu not found`: you are likely on login node; monitor through `srun --jobid ...` on compute node.
- `adapter_config.json` missing: wrong adapter path or adapter not prepared.
- Timeouts: reduce `MAX_STEPS` or increase `--time` in `sbatch`.
