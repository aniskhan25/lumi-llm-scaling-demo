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
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
export MAX_STEPS=${MAX_STEPS:-80}
export TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train_lora_demo.yaml}

echo "start_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "host=$(hostname)"
echo "world_size=1"
echo "job_id=${SLURM_JOB_ID:-none}"
echo "max_steps=$MAX_STEPS"

singularity run "$SIF_IMAGE" bash -lc '
set -euo pipefail
source "$VENV_ACTIVATE"

python scripts/train_lora_ddp.py \
  --config "$TRAIN_CONFIG" \
  --max_steps "$MAX_STEPS" \
  --log_file "$LOG_DIR/train_1gpu_rank0.jsonl" \
  --output_dir "$ARTIFACTS_DIR/adapters/adapter_live_1gpu"
'
