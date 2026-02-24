# Environment (LUMI ROCm, exact baseline)

## Target baseline

- System: LUMI-G (AMD MI250X)
- Runtime: PyTorch ROCm inside LUMI AI Factory container
- Distributed backend: `nccl` (RCCL on ROCm)
- Precision: `bf16`

## Container and module baseline

Use the AI Factory modules and one tested multi-framework container.

```bash
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# Recommended default from current LUMI AI docs (Jan 2026 refresh)
export SIF_IMAGE="${SIF_IMAGE:-/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif}"

# Quick sanity checks
singularity run "$SIF_IMAGE" python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('gpus', torch.cuda.device_count())"
singularity run "$SIF_IMAGE" python -c "import torch; print('nccl available', torch.distributed.is_nccl_available())"
```

If the exact path changes on your system, find current images:

```bash
ls -1 /appl/local/laifs/containers
```

## Project layout assumptions on LUMI

```bash
export PROJECT_ROOT=/path/to/lumi-llm-scaling-demo
export DATA_DIR=$PROJECT_ROOT/data
export OUTPUT_DIR=$PROJECT_ROOT/artifacts
export HF_HOME=/scratch/$PROJECT/hf-cache
mkdir -p "$HF_HOME" "$OUTPUT_DIR" "$PROJECT_ROOT/logs"
```

## Python package pinning (inside container)

Create a per-project venv mounted in writable storage and install only add-ons not guaranteed by container:

```bash
singularity run "$SIF_IMAGE" bash -lc '
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python - <<PY
import torch, transformers, peft, yaml
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("pyyaml", yaml.__version__)
PY
'
```

## ROCm/distributed environment defaults

Set these before `torchrun` for stable multi-GPU behavior:

```bash
export OMP_NUM_THREADS=7
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
```

Multi-node add-ons:

```bash
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export FI_PROVIDER=cxi
export FI_CXI_DEFAULT_CQ_SIZE=131072
```

## Verification checklist

- `torch.cuda.device_count()` matches allocated GPUs.
- `torch.distributed.is_nccl_available()` is `True`.
- `rocm-smi` shows all allocated GPUs.
- 10-step smoke test (1 GPU) finishes without OOM.
- 50-step smoke test (4 GPUs) shows throughput increase.

## Source references

- LUMI PyTorch setup docs: <https://docs.lumi-supercomputer.eu/software/packages/pytorch/>
- LUMI AI software stack transition note: <https://docs.lumi-supercomputer.eu/ai/>
