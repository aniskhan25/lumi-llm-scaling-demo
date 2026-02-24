#!/bin/bash
#SBATCH --job-name=lora-1gpu
#SBATCH --account=project_462000131
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --time=00:20:00
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
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}

echo "start_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "host=$(hostname)"
echo "world_size=1"
echo "slurm_job_id=${SLURM_JOB_ID:-none}"
echo "hip_visible_devices=$HIP_VISIBLE_DEVICES"
echo "code_root=$CODE_ROOT"
echo "run_root=$RUN_ROOT"
echo "log_dir=$LOG_DIR"
echo "artifacts_dir=$ARTIFACTS_DIR"

singularity run "$SIF_IMAGE" bash -lc '
set -euo pipefail
cd "$CODE_ROOT"
source "$VENV_ACTIVATE"

torchrun --standalone --nproc_per_node=1 "$CODE_ROOT/scripts/train_lora_ddp.py" \
  --config "$CODE_ROOT/configs/train_lora_demo.yaml" \
  --max_steps 80 \
  --log_file "$LOG_DIR/train_1gpu_rank0.jsonl" \
  --output_dir "$ARTIFACTS_DIR/adapters/adapter_live_1gpu"
'
