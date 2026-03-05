# LUMI LLM Scaling Demo (ROCm)

This repo demonstrates distributed LoRA training on LUMI-G (1/4/8 GPUs), scaling summaries, and optional before/after inference.

## Read in this order

1. [docs/DEMO_RUNBOOK.md](docs/DEMO_RUNBOOK.md) (main step-by-step guide)
2. [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) (setup details)
3. [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) (optional architecture details)

## Canonical paths on LUMI

- Repo: `/scratch/project_462000131/$USER/lumi-llm-scaling-demo`
- Venv: `/project/project_462000131/$USER/venvs/myvenv/bin/activate`

## Recommended demo mode

Use **Mode A** from the runbook:

- precomputed 1/4/8 scaling
- live 4-GPU launch
- live `rocm-smi` on the compute node
- optional before/after inference
