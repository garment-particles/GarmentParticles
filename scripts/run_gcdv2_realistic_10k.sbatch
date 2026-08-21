#!/bin/bash
#SBATCH --job-name=gcdreal-edit
#SBATCH --account=marlowe-m000133
#SBATCH --partition=preempt
#SBATCH --qos=normal
#SBATCH --time=11:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --requeue
#SBATCH --export=ALL
#SBATCH --output=outputs/gcdv2_realistic_10k/slurm/edit-%A_%a.out
#SBATCH --error=outputs/gcdv2_realistic_10k/slurm/edit-%A_%a.err

set -euo pipefail

REPO_DIR=/scratch/m000051/george/GarmentParticles
PYTHON_BIN=/scratch/m000051/george/miniconda3/envs/aipparel/bin/python
CHUNK_SIZE=${1:-250}
MAX_ITEMS=${2:-0}

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m tools.gcdv2_realistic_batch run \
  --run-dir "$REPO_DIR/outputs/gcdv2_realistic_10k" \
  --chunk-index "${SLURM_ARRAY_TASK_ID:-0}" \
  --chunk-size "$CHUNK_SIZE" \
  --max-items "$MAX_ITEMS" \
  --quality medium \
  --size 1024x1024 \
  --request-attempts 4 \
  --item-attempts 2
