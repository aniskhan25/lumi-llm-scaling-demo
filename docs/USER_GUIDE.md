# User Guide (Single Workflow)

## 1) Goal

Run and present:

- 1/4/8 GPU LoRA scaling on LUMI-G
- scaling summary and plot from real logs
- optional before/after inference outputs

## 2) Prerequisites

- Repo path: `/scratch/project_462000131/$USER/lumi-llm-scaling-demo`
- Venv path: `/project/project_462000131/$USER/venvs/myvenv/bin/activate`
- Run commands from repo root.

## 3) Environment setup

```bash
cd /scratch/project_462000131/$USER/lumi-llm-scaling-demo
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SIF_IMAGE=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif
export VENV_ACTIVATE=/project/project_462000131/$USER/venvs/myvenv/bin/activate
```

If venv does not exist yet:

```bash
singularity run "$SIF_IMAGE" bash -lc '
python -m venv --system-site-packages /project/project_462000131/$USER/venvs/myvenv
source /project/project_462000131/$USER/venvs/myvenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
'
```

GPU check (inside a GPU allocation):

```bash
srun --ntasks=1 --gpus=1 singularity exec "$SIF_IMAGE" bash -lc '
source "$VENV_ACTIVATE"
python -c "import torch; print(\"torch\", torch.__version__); print(\"gpus\", torch.cuda.device_count())"
python -c "import torch; print(\"nccl available\", torch.distributed.is_nccl_available())"
'
```

## 4) Generate scaling results (1/4/8 GPUs)

Build the scaling dataset:

```bash
python scripts/build_scaling_dataset.py \
  --input data/demo_support_subset.sample.jsonl \
  --output data/demo_support_subset.scaling_5000.jsonl \
  --size 5000 \
  --seed 42
```

Submit training jobs:

```bash
sbatch --time=00:45:00 --export=ALL,TRAIN_CONFIG=configs/train_lora_scaling.yaml,MAX_STEPS=140 run-scripts/run_1gpu.sh
sbatch --time=00:55:00 --export=ALL,TRAIN_CONFIG=configs/train_lora_scaling.yaml,MAX_STEPS=140 run-scripts/run_4gpu.sh
sbatch --time=00:55:00 --export=ALL,TRAIN_CONFIG=configs/train_lora_scaling.yaml,MAX_STEPS=140 run-scripts/run_2node_8gpu.sh
```

Build summary artifacts:

```bash
python scripts/parse_logs.py \
  --inputs logs/train_1gpu_rank0.jsonl logs/train_4gpu_rank0.jsonl logs/train_8gpu_rank0.jsonl \
  --output_dir artifacts \
  --warmup_steps 30
```

Expected outputs:

- `artifacts/scaling_summary.md`
- `artifacts/scaling_summary.csv`
- `artifacts/scaling_plot.png`

## 5) Live demo flow (recommended)

Launch a live 4-GPU run:

```bash
sbatch --export=ALL,TRAIN_CONFIG=configs/train_lora_scaling.yaml,MAX_STEPS=140 run-scripts/run_4gpu.sh
squeue -u $USER
```

Monitor GPU activity:

```bash
JOBID=<jobid_from_sbatch>
NODE=$(squeue -j "$JOBID" -h -o %N)

srun --overlap --jobid="$JOBID" -w "$NODE" --ntasks=1 \
  bash run-scripts/watch_rocm_smi.sh logs/rocm_smi_live.log 1
```

Watch training logs:

```bash
tail -f logs/train_4gpu_rank0.jsonl
```

Present:

- `artifacts/scaling_summary.md`
- `artifacts/scaling_plot.png`

## 6) Optional before/after inference

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

After:

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

## 7) Troubleshooting

- No GPUs visible: run via `srun --ntasks=1 --gpus=1 ...`.
- `amdgpu not found`: you are on a login node, not a compute node.
- `adapter_config.json` missing: adapter path is wrong or adapter is not prepared.
- Timeout: lower `MAX_STEPS` or raise job `--time`.

## 8) Completion checklist

- `logs/train_1gpu_rank0.jsonl` exists.
- `logs/train_4gpu_rank0.jsonl` exists.
- `logs/train_8gpu_rank0.jsonl` exists.
- `artifacts/scaling_summary.md` exists.
- `artifacts/scaling_plot.png` exists.
- Optional: `artifacts/before_outputs.jsonl` and `artifacts/after_outputs.jsonl` exist.
