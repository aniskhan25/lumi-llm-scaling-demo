# Demo Runbook

## 1) Setup (run from repo root)

```bash
cd /scratch/project_462000131/$USER/lumi-llm-scaling-demo
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SIF_IMAGE=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif
export VENV_ACTIVATE=/project/project_462000131/$USER/venvs/myvenv/bin/activate
```

GPU sanity check (inside a GPU allocation):

```bash
srun --ntasks=1 --gpus=1 singularity exec "$SIF_IMAGE" bash -lc '
source "$VENV_ACTIVATE"
python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
'
```

## 2) Build scaling artifacts (1/4/8 GPUs)

```bash
python scripts/build_scaling_dataset.py \
  --input data/demo_support_subset.sample.jsonl \
  --output data/demo_support_subset.scaling_5000.jsonl \
  --size 5000 \
  --seed 42

sbatch --time=00:45:00 --export=ALL,TRAIN_CONFIG=configs/train_lora_scaling.yaml,MAX_STEPS=140 run-scripts/run_1gpu.sh
sbatch --time=00:55:00 --export=ALL,TRAIN_CONFIG=configs/train_lora_scaling.yaml,MAX_STEPS=140 run-scripts/run_4gpu.sh
sbatch --time=00:55:00 --export=ALL,TRAIN_CONFIG=configs/train_lora_scaling.yaml,MAX_STEPS=140 run-scripts/run_2node_8gpu.sh

python scripts/parse_logs.py \
  --inputs logs/train_1gpu_rank0.jsonl logs/train_4gpu_rank0.jsonl logs/train_8gpu_rank0.jsonl \
  --output_dir artifacts \
  --warmup_steps 30
```

Expected files:

- `artifacts/scaling_summary.md`
- `artifacts/scaling_summary.csv`
- `artifacts/scaling_plot.png`

## 3) Live demo (recommended)

Submit a live 4-GPU run:

```bash
sbatch --export=ALL,TRAIN_CONFIG=configs/train_lora_scaling.yaml,MAX_STEPS=140 run-scripts/run_4gpu.sh
squeue -u $USER
```

Monitor GPU activity from the same running job:

```bash
JOBID=<jobid_from_sbatch>
NODE=$(squeue -j "$JOBID" -h -o %N)

srun --overlap --jobid="$JOBID" -w "$NODE" --ntasks=1 \
  bash run-scripts/watch_rocm_smi.sh logs/rocm_smi_live.log 1
```

Follow training logs:

```bash
tail -f logs/train_4gpu_rank0.jsonl
```

Show results:

- `artifacts/scaling_summary.md`
- `artifacts/scaling_plot.png`

## 4) Optional before/after inference

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

## 5) Quick troubleshooting

- No GPUs visible: run commands with `srun --ntasks=1 --gpus=1 ...`.
- `amdgpu not found`: you are on a login node; run monitoring via `srun --jobid ...` on compute node.
- `adapter_config.json` missing: wrong adapter path or adapter not prepared.
- Timeout: reduce `MAX_STEPS` or increase `--time`.
