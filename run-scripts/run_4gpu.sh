#!/bin/bash
#SBATCH -J lora-4gpu
#SBATCH -p small-g
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=28
#SBATCH -t 00:25:00
#SBATCH -o logs/slurm-%x-%j.out

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SIF_IMAGE="${SIF_IMAGE:-/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-7}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}

mkdir -p logs artifacts/adapters

echo "start_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "host=$(hostname)"
echo "world_size=4"
echo "slurm_job_id=${SLURM_JOB_ID:-none}"
echo "hip_visible_devices=$HIP_VISIBLE_DEVICES"

singularity run "$SIF_IMAGE" bash -lc '
set -euo pipefail
cd "'$PROJECT_ROOT'"
source .venv/bin/activate

torchrun --standalone --nproc_per_node=4 scripts/train_lora_ddp.py \
  --config configs/train_lora_demo.yaml \
  --max_steps 150 \
  --log_file logs/train_4gpu_rank0.jsonl \
  --output_dir artifacts/adapters/adapter_live_4gpu
'
