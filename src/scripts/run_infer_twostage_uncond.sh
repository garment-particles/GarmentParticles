#!/usr/bin/bash

#SBATCH -N 1
#SBATCH -G 1
#SBATCH --cpus-per-task 8
#SBATCH --job-name=infer_twostage_uncond
#SBATCH --mem=128G
#SBATCH -A marlowe-m000051-pm05
#SBATCH -p batch
#SBATCH --time=01:59:00
#SBATCH --output=/scratch/m000051/george/cluster_logs/%x-%j/%j.out
#SBATCH --error=/scratch/m000051/george/cluster_logs/%x-%j/%j.err

mkdir -p /scratch/m000051/george/cluster_logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}

source ~/.bashrc
source /scratch/m000051/george/miniconda3/bin/activate
conda activate interact_garment
cd "${SLURM_SUBMIT_DIR:-$PWD}"   # run/submit this from the repo's src/ directory

export PYTHONPATH="$PWD:${PYTHONPATH}"
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp/$USER}/triton"
mkdir -p "$TRITON_CACHE_DIR"

echo "Host: $(hostname)"
echo "GPU(s):"; nvidia-smi -L

torchrun --standalone --nproc_per_node=1 inference/infer_twostage.py \
  eval.sample_per_batch=1 \
  eval.n_samples=0 \
  eval.evaluate=False \
  train.exp_name=uncond_samples \
  sample.num_sampling_steps=100 \
  gpf_ckpt=null \
  dataset.front_only=True \
  dataset.use_all_captions=True \
  dataset.img_drop_prob=1 \
  dataset.text_drop_prob=1 \
  model.use_qknorm=True \
  edge_model.use_qknorm=True \
  edge_model_ckpt=checkpoints/edge \
  model=sparse_lightningdit_v3_xl1_w_text_fsdp2 \
  pgf_weight_init=checkpoints/pgf_text \
  --config-name sparselightningdit_xl_garment_particle_inference
