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

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if ! mkdir -p "$RUN_ROOT" 2>/dev/null; then
  RUN_ROOT="/tmp/${USER:-${LOGNAME:-user}}/lumi-llm-scaling-demo/${SLURM_JOB_ID:-manual}"
  mkdir -p "$RUN_ROOT"
fi
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$RUN_ROOT/artifacts}"
mkdir -p "$LOG_DIR" "$ARTIFACTS_DIR/adapters"

export CODE_ROOT RUN_ROOT LOG_DIR ARTIFACTS_DIR
export VENV_ACTIVATE="${VENV_ACTIVATE:-$CODE_ROOT/.venv/bin/activate}"

module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SIF_IMAGE="${SIF_IMAGE:-/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-7}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-hsn0,hsn1,hsn2,hsn3}
export FI_PROVIDER=${FI_PROVIDER:-cxi}
export FI_CXI_DEFAULT_CQ_SIZE=${FI_CXI_DEFAULT_CQ_SIZE:-131072}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=${MASTER_PORT:-29500}
export NNODES=${SLURM_NNODES}
export GPUS_PER_NODE=4

echo "start_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')"
echo "master_addr=$MASTER_ADDR"
echo "master_port=$MASTER_PORT"
echo "world_size=$((NNODES * GPUS_PER_NODE))"
echo "code_root=$CODE_ROOT"
echo "run_root=$RUN_ROOT"
echo "log_dir=$LOG_DIR"
echo "artifacts_dir=$ARTIFACTS_DIR"

srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash -lc '
set -euo pipefail
echo "node_rank=${SLURM_NODEID} host=$(hostname)"

singularity run "$SIF_IMAGE" bash -lc "
set -euo pipefail
cd \"$CODE_ROOT\"
source \"$VENV_ACTIVATE\"

torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$GPUS_PER_NODE \
  --node_rank=$SLURM_NODEID \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  \"$CODE_ROOT/scripts/train_lora_ddp.py\" \
    --config \"$CODE_ROOT/configs/train_lora_demo.yaml\" \
    --max_steps 200 \
    --log_file \"$LOG_DIR/train_8gpu_rank0.jsonl\" \
    --output_dir \"$ARTIFACTS_DIR/adapters/adapter_live_8gpu\"
"
'
