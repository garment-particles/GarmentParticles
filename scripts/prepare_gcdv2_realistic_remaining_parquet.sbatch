#!/bin/bash
#SBATCH --job-name=gcdreal-pq-prepare
#SBATCH --account=marlowe-m000051-pm06
#SBATCH --partition=batch
#SBATCH --qos=medium
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --requeue
#SBATCH --export=ALL
#SBATCH --output=outputs/gcdv2_realistic_remaining_parquet/slurm/prepare-%j.out
#SBATCH --error=outputs/gcdv2_realistic_remaining_parquet/slurm/prepare-%j.err

set -euo pipefail

REPO_DIR=/scratch/m000051/george/GarmentParticles
PYTHON_BIN=/scratch/m000051/george/miniconda3/envs/aipparel/bin/python

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m tools.gcdv2_realistic_parquet prepare \
  --archive /scratch/m000051-pm06/george/gcdv2_archives/garmentcodedatav2.tar \
  --captions /scratch/m000051/george/gcdv2/short_captions_v2.json \
  --exclude-manifest "$REPO_DIR/outputs/gcdv2_realistic_10k/manifest.jsonl" \
  --run-dir "$REPO_DIR/outputs/gcdv2_realistic_remaining_parquet" \
  --seed 20260819 \
  --style-index-offset 10000 \
  --rows-per-shard 64
