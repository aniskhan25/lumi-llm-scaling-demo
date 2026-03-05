#!/bin/bash
#SBATCH --job-name=lora-8gpu-2n
#SBATCH --account=project_462000131
#SBATCH --partition=small-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=28
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

VENV_ACTIVATE="${VENV_ACTIVATE:-/project/project_462000131/$USER/venvs/myvenv/bin/activate}"
LOG_DIR="${LOG_DIR:-logs}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts}"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "ERROR: venv activate script not found: $VENV_ACTIVATE" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$ARTIFACTS_DIR/adapters"

export LOG_DIR ARTIFACTS_DIR
export VENV_ACTIVATE

module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SIF_IMAGE="${SIF_IMAGE:-/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-7}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-hsn0,hsn1,hsn2,hsn3}
export FI_PROVIDER=${FI_PROVIDER:-cxi}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=${MASTER_PORT:-29500}
export NNODES=${SLURM_NNODES}
export GPUS_PER_NODE=4
export MAX_STEPS=${MAX_STEPS:-120}
export TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train_lora_demo.yaml}

echo "start_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')"
echo "job_id=${SLURM_JOB_ID:-none}"
echo "world_size=$((NNODES * GPUS_PER_NODE))"
echo "max_steps=$MAX_STEPS"

srun --kill-on-bad-exit=1 --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash -lc '
set -euo pipefail
echo "node_rank=${SLURM_NODEID} host=$(hostname)"

singularity run "$SIF_IMAGE" bash -lc "
set -euo pipefail
source \"$VENV_ACTIVATE\"

torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$GPUS_PER_NODE \
  --node_rank=$SLURM_NODEID \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  scripts/train_lora_ddp.py \
    --config \"$TRAIN_CONFIG\" \
    --max_steps \"$MAX_STEPS\" \
    --log_file \"$LOG_DIR/train_8gpu_rank0.jsonl\" \
    --output_dir \"$ARTIFACTS_DIR/adapters/adapter_live_8gpu\"
"
'
