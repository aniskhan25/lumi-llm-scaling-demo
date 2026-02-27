# LUMI LLM Scaling Demo (ROCm)

This repository is a handoff-ready implementation scaffold for a real-world enterprise assistant demo on LUMI (AMD MI250X), including:

- Before/after inference with a pre-trained LoRA adapter
- Live short LoRA fine-tune with distributed training (1/4/8 GPUs)
- ROCm monitoring helpers and scaling artifact generation
- Exact environment, run scripts, and presenter runbook

Start here:

1. [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)
2. [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md)
3. [docs/DEMO_RUNBOOK.md](docs/DEMO_RUNBOOK.md)
4. [docs/HANDOFF_PLAN.md](docs/HANDOFF_PLAN.md)

LUMI venv path used by default in launcher scripts:

`/project/project_462000131/anisrahm/venvs/myvenv/bin/activate`

## Interactive Demo (Requires GPU Allocation)

For live terminal demos, request an interactive GPU allocation first.

1 GPU interactive session (inference checks):

```bash
salloc --account=project_462000131 --partition=small-g --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=7 --time=00:30:00
```

4 GPU interactive session (live training demo):

```bash
salloc --account=project_462000131 --partition=small-g --nodes=1 --gpus-per-node=4 --ntasks-per-node=1 --cpus-per-task=28 --time=00:45:00
```

After allocation starts:

```bash
cd /scratch/project_462000131/anisrahm/lumi-llm-scaling-demo
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
export SIF_IMAGE=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif
export VENV_ACTIVATE=/project/project_462000131/anisrahm/venvs/myvenv/bin/activate

srun --ntasks=1 --gpus=1 singularity exec "$SIF_IMAGE" bash -lc '
source "$VENV_ACTIVATE"
python -c "import torch; print(\"GPUs:\", torch.cuda.device_count())"
python scripts/infer_before_after.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --prompts_file prompts/demo_prompts.jsonl \
  --output_file artifacts/before_outputs.jsonl
'
```
