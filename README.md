# GarmentParticles

Diffusion-based garment generation. A **two-stage pipeline** generates a garment
as a 3D particle cloud (stage 1 — the *particle generative function*, PGF) and
then reconstructs its sewing pattern (stage 2 — the *edge model*). Three
conditioning modes are supported:

- **Text / unconditional**
- **Image** — conditioned on a single rendered garment image (GarmentCodeData-v2)
- **Sketch** — conditioned on a line-art drawing

---

## Installation

```bash
conda create -n interact_garment python=3.10
conda activate interact_garment

pip install torch torchvision                  # tested with torch >=2.4, CUDA 12.x
pip install -r src/requirements.txt
pip install flash-attn --no-build-isolation    # optional; may need to build from source

# add src/ to the Python import path (run from the repo root)
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

The interactive GUI / inference server additionally needs the Earth Mover's
Distance CUDA extension, built against your torch/CUDA:

```bash
cd external/PyTorchEMD
python setup.py build_ext --inplace
cd ../..
```

---

## Repository layout

```
src/
├── inference/infer_twostage.py   two-stage inference entry point
├── train_fsdp2.py                training entry point (all models, FSDP2)
├── models/                       PGF + edge model architectures
├── datasets/                     particle & edge datasets
├── configs/                      Hydra configs
├── transport/  patterns/  utils/
└── tools/run_server.py           Flask inference server for the GUI
external/
├── paint/                        PyQt interactive editor (HTTP client)
├── PyTorchEMD/                   EMD CUDA extension
└── GarmentCode*/  NvidiaWarp-GarmentCode/   data-generation pipeline
```

---

## Pre-trained models

Download the checkpoints into `src/checkpoints/`. Each is a Distributed
Checkpoint (DCP) **directory**.

| Model        | Role                       | Path                        |
|--------------|----------------------------|-----------------------------|
| PGF — text   | stage 1, text/unconditional | `src/checkpoints/pgf_text`   |
| PGF — image  | stage 1, GCDv2 image        | `src/checkpoints/pgf_image`  |
| PGF — sketch | stage 1, line-art           | `src/checkpoints/pgf_sketch` |
| Edge model   | stage 2 (shared)            | `src/checkpoints/edge`       |

Download all four from the model repo:

```bash
hf download georgeNakayama/GarmentParticles --local-dir src/checkpoints
```

---

## Dataset

The garment-particle dataset lives in a separate HuggingFace **dataset** repo —
`georgeNakayama/GarmentParticles` with `--repo-type dataset` (same name as the
model repo, different type):

- `data/particles-*.tar` — 26 shards of per-garment particle data
  (`rand_<id>/garment_particles_rand_<id>.h5` + `stats.txt`)
- `splits/garment_particle_v2_{train,test}_11182025.txt` — train / test splits

Download and unpack the shards:

```bash
hf download georgeNakayama/GarmentParticles --repo-type dataset --local-dir garment_data
cd garment_data && for t in data/*.tar; do tar -xf "$t"; done && cd ..
# -> garment_data/rand_<id>/garment_particles_rand_<id>.h5
```

Put the split files where the configs expect them, then point the dataset
configs at the extracted data:

```bash
cp garment_data/splits/*.txt src/assets/
```

Set `data_dir` in `src/configs/dataset/garment_particles_v2.2.yaml` (and
`garment_particle_dir` in `garment_edges_v2.2.yaml`) to the absolute path of
`garment_data/`.

**GarmentCodeData-v2** (rendered garment images + sewing-pattern specs) is *not*
included here — image-/sketch-conditioned inference and edge-model training need
it. Download it from the official source and set `gcd_dir` / `text_path` (and,
for edge training, `garment_edge_dir`) in the dataset configs accordingly.

---

## Inference

All modes run through `inference/infer_twostage.py`. Run from `src/`:

```bash
cd src
```

### Text / unconditional

```bash
torchrun --standalone --nproc_per_node=1 inference/infer_twostage.py \
  eval.sample_per_batch=1 eval.n_samples=0 eval.evaluate=False \
  train.exp_name=uncond_samples sample.num_sampling_steps=100 \
  gpf_ckpt=null \
  dataset.front_only=True dataset.use_all_captions=True \
  dataset.img_drop_prob=1 dataset.text_drop_prob=1 \
  model.use_qknorm=True \
  edge_model.use_qknorm=True \
  edge_model_ckpt=checkpoints/edge \
  model=sparse_lightningdit_v3_xl1_w_text_fsdp2 \
  pgf_weight_init=checkpoints/pgf_text \
  --config-name sparselightningdit_xl_garment_particle_inference
```

### Image-conditioned (GCDv2)

```bash
torchrun --standalone --nproc_per_node=1 inference/infer_twostage.py \
  eval.sample_per_batch=1 eval.n_samples=0 eval.evaluate=False \
  train.exp_name=img_cond_samples sample.num_sampling_steps=100 \
  gpf_ckpt=null \
  dataset.front_only=True dataset.use_all_captions=True \
  dataset.img_drop_prob=0 dataset.text_drop_prob=1 \
  model.use_qknorm=True model.use_rope=False model.in_channels=6 model.freeze_everything=False \
  edge_model.use_qknorm=True \
  edge_model_ckpt=checkpoints/edge \
  model=sparse_lightningdit_v3_xl1_w_img_text_v2 \
  pgf_weight_init=checkpoints/pgf_image \
  --config-name sparselightningdit_xl_garment_particle_inference
```

### Sketch-conditioned (line-art)

Same as the image command, with the line-art dataset and the sketch PGF:

```bash
torchrun --standalone --nproc_per_node=1 inference/infer_twostage.py \
  eval.sample_per_batch=1 eval.n_samples=0 eval.evaluate=False \
  train.exp_name=lineart_cond_samples sample.num_sampling_steps=100 \
  gpf_ckpt=null \
  dataset.front_only=True dataset.use_all_captions=True \
  dataset.img_drop_prob=0 dataset.text_drop_prob=1 \
  model.use_qknorm=True model.use_rope=False model.in_channels=6 model.freeze_everything=False \
  edge_model.use_qknorm=True \
  edge_model_ckpt=checkpoints/edge \
  dataset=garment_edges_v2.2_w_img_text_lineart \
  model=sparse_lightningdit_v3_xl1_w_img_text_v2 \
  pgf_weight_init=checkpoints/pgf_sketch \
  --config-name sparselightningdit_xl_garment_particle_inference
```

Results are written to `outputs/<exp_name>/<date>/<time>/samples_*/` — generated
point clouds (`.ply`), renders (`.png`), and reconstructed sewing-pattern panels
(`.npz`).

---

## Training

`train_fsdp2.py` is the single, config-driven entry point for **all** models
(FSDP2; launch with `torchrun`, set `--nproc_per_node` to your GPU count).
Training reads the GarmentCodeData-v2 dataset — set `data_dir`, `gcd_dir`,
`text_path` in the dataset configs (`src/configs/dataset/`) to your data
location first.

### PGF — text

```bash
torchrun --nproc_per_node=8 train_fsdp2.py \
  model.use_qknorm=True \
  --config-name sparselightningdit_xl_garment_particle_v2.2_w_text_fsdp2
```

### PGF — image

Fine-tuned from the text PGF (`train.weight_init`):

```bash
torchrun --nproc_per_node=8 train_fsdp2.py \
  model=sparse_lightningdit_v3_xl1_w_img_text_v2 \
  model.use_qknorm=True model.freeze_everything=False \
  dataset.pad_everything=False dataset.text_drop_prob=0.8 \
  data.sampler.target_tokens_per_batch=16382 \
  train.weight_init=checkpoints/pgf_text \
  --config-name sparselightningdit_xl_garment_particle_v2.2_w_img_text_fsdp2
```

### PGF — sketch

Same as image, with the line-art dataset:

```bash
torchrun --nproc_per_node=8 train_fsdp2.py \
  model=sparse_lightningdit_v3_xl1_w_img_text_v2 \
  model.use_qknorm=True model.freeze_everything=False \
  dataset=garment_particles_v2.2_w_img_text_lineart \
  dataset.pad_everything=False dataset.text_drop_prob=0.8 \
  data.sampler.target_tokens_per_batch=16382 \
  train.weight_init=checkpoints/pgf_text \
  --config-name sparselightningdit_xl_garment_particle_v2.2_w_img_text_fsdp2
```

### Edge model

Stage 2. Two architectures are provided — `varlen` (variable-length attention,
recommended) and the non-varlen baseline.

```bash
# varlen (recommended)
torchrun --nproc_per_node=8 train_fsdp2.py \
  train.global_batch_size=128 \
  dataset.predict_rigid_transformations=True \
  dataset.predict_attachment_type=True \
  dataset.predict_stitch_tags=True \
  dataset.predict_valid_mask=True \
  dataset.order_panels_by=3d \
  model.in_channels=15 model.use_panel_embedding=False \
  model.use_qknorm=True model.use_rope=True \
  --config-name sparselightningdit_l_edges_v2.2_varlen_fsdp2

# non-varlen baseline: same overrides, swap the config name to
#   --config-name sparselightningdit_l_edges_v2.2_fsdp2
```

---

## Interactive GUI

The `external/paint/` PyQt editor performs inverse design (draw a guide → get a
garment) by talking to a Flask inference server over HTTP.

**Server** (on a GPU machine):

```bash
cd src
python tools/run_server.py --host 0.0.0.0 --port 12345
```

**Client** (the GUI, on any machine with a display):

```bash
conda install -c conda-forge pyqt vtk pillow numpy
pip install requests

export INTERACT_GARMENT_HOST=<server-host>   # default 127.0.0.1
export INTERACT_GARMENT_PORT=12345
cd external/paint
python main.py
```

`File → Inference` posts the drawn guide to the server and renders the result.
