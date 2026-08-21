#!/bin/bash
#SBATCH --job-name=gcdreal-prepare
#SBATCH --account=marlowe-m000133
#SBATCH --partition=preempt
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --requeue
#SBATCH --export=ALL
#SBATCH --output=outputs/gcdv2_realistic_10k/slurm/prepare-%j.out
#SBATCH --error=outputs/gcdv2_realistic_10k/slurm/prepare-%j.err

set -euo pipefail

REPO_DIR=/scratch/m000051/george/GarmentParticles
PYTHON_BIN=/scratch/m000051/george/miniconda3/envs/aipparel/bin/python

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m tools.gcdv2_realistic_batch prepare \
  --archive /scratch/m000051-pm06/george/gcdv2_archives/garmentcodedatav2.tar \
  --captions /scratch/m000051/george/gcdv2/short_captions_v2.json \
  --run-dir "$REPO_DIR/outputs/gcdv2_realistic_10k" \
  --count 10000 \
  --seed 20260819
