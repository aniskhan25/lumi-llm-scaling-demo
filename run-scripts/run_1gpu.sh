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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${CODE_ROOT:-}" ]]; then
  CODE_ROOT="$CODE_ROOT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/train_lora_ddp.py" ]]; then
  CODE_ROOT="$SLURM_SUBMIT_DIR"
elif [[ -f "$SCRIPT_DIR/../scripts/train_lora_ddp.py" ]]; then
  CODE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
  echo "ERROR: Could not determine CODE_ROOT." >&2
  echo "Set CODE_ROOT to your repo path before sbatch." >&2
  exit 1
fi

DEFAULT_SCRATCH_VENV="/scratch/project_462000131/${USER:-anisrahm}/venvs/myvenv/bin/activate"
if [[ -z "${VENV_ACTIVATE:-}" ]]; then
  if [[ -f "$DEFAULT_SCRATCH_VENV" ]]; then
    VENV_ACTIVATE="$DEFAULT_SCRATCH_VENV"
  else
    VENV_ACTIVATE="$CODE_ROOT/.venv/bin/activate"
  fi
fi
if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "ERROR: VENV activate script not found: $VENV_ACTIVATE" >&2
  echo "Set VENV_ACTIVATE=/scratch/project_462000131/anisrahm/venvs/myvenv/bin/activate" >&2
  exit 1
fi

RUN_ROOT="${RUN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if ! mkdir -p "$RUN_ROOT" 2>/dev/null; then
  RUN_ROOT="/tmp/${USER:-${LOGNAME:-user}}/lumi-llm-scaling-demo/${SLURM_JOB_ID:-manual}"
  mkdir -p "$RUN_ROOT"
fi
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$RUN_ROOT/artifacts}"
mkdir -p "$LOG_DIR" "$ARTIFACTS_DIR/adapters"

export CODE_ROOT RUN_ROOT LOG_DIR ARTIFACTS_DIR
export VENV_ACTIVATE

module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SIF_IMAGE="${SIF_IMAGE:-/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-7}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
export NCCL_DEBUG=${NCCL_DEBUG:-ERROR}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-OFF}
export HF_HUB_DISABLE_PROGRESS_BARS=${HF_HUB_DISABLE_PROGRESS_BARS:-1}
export MAX_STEPS=${MAX_STEPS:-80}
export TRAIN_CONFIG=${TRAIN_CONFIG:-$CODE_ROOT/configs/train_lora_demo.yaml}

echo "start_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "host=$(hostname)"
echo "world_size=1"
echo "slurm_job_id=${SLURM_JOB_ID:-none}"
echo "hip_visible_devices=$HIP_VISIBLE_DEVICES"
echo "max_steps=$MAX_STEPS"
echo "train_config=$TRAIN_CONFIG"
echo "code_root=$CODE_ROOT"
echo "run_root=$RUN_ROOT"
echo "log_dir=$LOG_DIR"
echo "artifacts_dir=$ARTIFACTS_DIR"

singularity run "$SIF_IMAGE" bash -lc '
set -euo pipefail
cd "$CODE_ROOT"
source "$VENV_ACTIVATE"

python "$CODE_ROOT/scripts/train_lora_ddp.py" \
  --config "$TRAIN_CONFIG" \
  --max_steps "$MAX_STEPS" \
  --log_file "$LOG_DIR/train_1gpu_rank0.jsonl" \
  --output_dir "$ARTIFACTS_DIR/adapters/adapter_live_1gpu"
'
