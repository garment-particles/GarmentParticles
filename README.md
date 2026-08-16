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

### Prepare stage-one particle data from simulated GCDv2 garments

The downloadable particle shards above are already ready for training. Use
this preparation step only when regenerating them from GarmentCodeData-v2 (or
from garments produced by the GarmentCode simulation pipeline).

The input is a text file containing one absolute garment-directory path per
line. Each directory must have a specification, its box mesh, and the draped
simulation mesh, with the directory basename used as the file prefix:

```text
/absolute/path/to/garmentcodedatav2/.../rand_ABC123/
├── rand_ABC123_specification.json
├── rand_ABC123_boxmesh.ply
└── rand_ABC123_sim.ply
```

From the repository root, activate the repository environment. The external
preparation pipeline uses geometry packages that are not part of
`src/requirements.txt`, so install the local GarmentCode package and its
Triangle dependency once:

```bash
conda activate interact_garment
python -m pip install -e external/GarmentCode
python -m pip install triangle

# Optional dependency check before a long run
python -c "import CGAL, h5py, igl, shapely, triangle, trimesh"
```

If the garment simulations already exist, build the pattern list from the
repository root. Use paths that are valid on the machine that will perform the
preprocessing:

```bash
export GCD_ROOT=/absolute/path/to/garmentcodedatav2
export PATTERN_LIST=/absolute/path/to/all_pattern_list.txt

find "$GCD_ROOT" -type f -name '*_specification.json' -printf '%h\n' \
  | sort -u > "$PATTERN_LIST"
```

It is worth running a small visual smoke test before processing the full
dataset:

```bash
export PARTICLE_ROOT=/absolute/path/to/garment_particles
sed -n '1,3p' "$PATTERN_LIST" > /tmp/garmentparticles_pattern_smoke.txt

python external/GarmentCode/process_garment_particles_normal_bnd.py \
  --pattern_list /tmp/garmentparticles_pattern_smoke.txt \
  --output_dir "${PARTICLE_ROOT}_smoke" \
  --packing_strategy fine_to_coarse \
  --packing_padding 3 \
  --packing_max_iterations 500 \
  --vis
```

Inspect each generated `panel_vis.png`. It shows the packed front and back
boundary particles in red and interior particles in blue. Then run the full
preparation, normally under a batch scheduler for a large dataset:

```bash
python external/GarmentCode/process_garment_particles_normal_bnd.py \
  --pattern_list "$PATTERN_LIST" \
  --output_dir "$PARTICLE_ROOT" \
  --packing_strategy fine_to_coarse \
  --packing_padding 3 \
  --packing_max_iterations 500 \
  --resume
```

`fine_to_coarse` is the recommended strategy. It packs front and back panels
independently, preserves semantic garment hierarchy, and rejects a garment if
overlaps remain after the requested iteration limit. The default clearance is
3 cm. Other available strategies are `hierarchical`, `individual`, and
`joint_optimization`.

Each successful garment produces:

```text
garment_particles/rand_ABC123/
├── garment_particles_rand_ABC123.h5   # train-ready completion marker
├── garment_particles_rand_ABC123.npz  # legacy representation
├── panel_offsets_rand_ABC123.json
├── packing_metadata_rand_ABC123.json
├── stats.txt
└── panel_vis.png                       # only when --vis is supplied
```

The HDF5 schema is
`front|back/<panel_name>/boundary_verts|interior_verts`. Every particle row is
`[packed_u, packed_v, simulated_x, simulated_y, simulated_z]`; packing settings
and panel offsets are also stored as HDF5 attributes. The HDF5 file is written
atomically and published last, so `--resume` skips only completed samples.
Failures are recorded in the timestamped processing log and
`failed_folders_<timestamp>.txt`; the command exits nonzero when any garment
fails.

Finally, set `data_dir` in
[`src/configs/dataset/garment_particles_v2.2.yaml`](src/configs/dataset/garment_particles_v2.2.yaml)
to `PARTICLE_ROOT`. The loader discovers
`PARTICLE_ROOT/*/garment_particles_*.h5`. For image-conditioned training, also
set `gcd_dir`, `gcd_list_file`, and `text_path`; garment basenames in the GCD
list must match the particle directories. For sketch-conditioned training, set
`image_dir` to the line-art renders and configure `text_path`.

For GarmentCode sampling and simulation instructions, see
[`external/GarmentCode/docs/Running_data_generation.md`](external/GarmentCode/docs/Running_data_generation.md).

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
