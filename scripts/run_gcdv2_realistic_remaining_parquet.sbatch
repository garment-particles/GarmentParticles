#!/bin/bash
#SBATCH --job-name=gcdreal-parquet
#SBATCH --account=marlowe-m000051-pm06
#SBATCH --partition=batch
#SBATCH --qos=medium
#SBATCH --time=1-23:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --requeue
#SBATCH --export=ALL
#SBATCH --output=outputs/gcdv2_realistic_remaining_parquet/slurm/edit-%A_%a.out
#SBATCH --error=outputs/gcdv2_realistic_remaining_parquet/slurm/edit-%A_%a.err

set -euo pipefail

REPO_DIR=/scratch/m000051/george/GarmentParticles
PYTHON_BIN=/scratch/m000051/george/miniconda3/envs/aipparel/bin/python
MAX_ITEMS=${1:-0}
WORKER_COUNT=${2:-20}
RETRY_FAILED=${3:-0}

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

args=(
  --run-dir "$REPO_DIR/outputs/gcdv2_realistic_remaining_parquet"
  --worker-index "${SLURM_ARRAY_TASK_ID:-0}"
  --worker-count "$WORKER_COUNT"
  --max-items "$MAX_ITEMS"
  --quality medium
  --size 1024x1024
  --request-attempts 4
  --item-attempts 3
  --stop-before-job-end-seconds 900
  --max-runtime-seconds 165600
)
if [[ "$RETRY_FAILED" == 1 ]]; then
  args+=(--retry-failed)
fi

set +e
"$PYTHON_BIN" -m tools.gcdv2_realistic_parquet run "${args[@]}"
worker_status=$?
set -e

if [[ "$worker_status" == 75 ]]; then
  task_ref="${SLURM_ARRAY_JOB_ID:?}_${SLURM_ARRAY_TASK_ID:?}"
  echo "Requeueing $task_ref after a clean Parquet checkpoint"
  scontrol requeue "$task_ref"
  exit 0
fi
exit "$worker_status"
