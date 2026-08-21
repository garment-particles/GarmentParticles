# GCDv2 realistic garment-image generation

This note records how we converted GarmentCodeData-v2-derived synthetic renders
into photorealistic images of people wearing the garments, and how we then
created diverse material and texture variants. It documents the experiment in
this repository; it is not a claim that we reproduced every component of the
original [FIT-VTO](https://github.com/HarryWang355/FIT-VTO/tree/main) data
pipeline.

## Summary

```text
GCDv2 Blender front render
        |
        v
Luna garment caption (shape, construction, color)
        |
        v
GPT Image 2 image edit (synthetic mannequin -> realistic person)
        |
        v
Visual/QA check for garment preservation
        |
        v
Optional GPT Image 2 edit with a distinct material/texture prompt
```

The main idea was to use **image editing**, not unconstrained text-to-image
generation. The input render supplied the pose, framing, silhouette, garment
pieces, seams, panels, closures, lengths, and initial colors. The prompt told
the model to replace the synthetic body and rendering style while keeping those
garment attributes fixed.

## Models and API routes

| Purpose | Model | NVIDIA OpenAI-compatible route |
| --- | --- | --- |
| Garment captioning and QA | `azure/openai/gpt-5.6-luna` | `https://inference-api.nvidia.com/openai/v1/responses` |
| Photorealistic conversion | `openai/openai/gpt-image-2` | `https://inference-api.nvidia.com/v1/images/edits` |
| Texture/material editing | `openai/openai/gpt-image-2` | `https://inference-api.nvidia.com/v1/images/edits` |

GPT Image 2 accepts image inputs and supports the image-edit endpoint according
to the [official OpenAI model documentation](https://developers.openai.com/api/docs/models/gpt-image-2).
The local implementation is
[`src/tools/fit_vto_api.py`](src/tools/fit_vto_api.py).

The client reads the inference credential from the second-to-last non-empty
entry in `/scratch/m000133/george/foveated_diffusion_3d/key.txt`. The final
entry is a management credential and is not used for inference. No credential
is copied into this repository, prompt file, usage file, or generated image.

## Inputs used in the sample run

The full GCDv2 source directory and GarmentParticles checkpoints were not
available during this experiment. We therefore reused three existing,
deterministic GCDv2-derived Blender front renders from
`/scratch/m000051/george/InteractGarment/renders`:

| GCDv2 ID | Garment summary |
| --- | --- |
| `rand_0AAMPMWVT4` | Strapless fitted pale-aqua bodice and calf-length A-line skirt |
| `rand_0A6BT9ADFV` | Sleeveless color-blocked top and straight purple slit skirt |
| `rand_5LBE3RK8FW` | Raised-hood cropped jacket with flared sleeves and straight skirt |

Copies of these inputs are under
[`outputs/gcdv2_realism/sources/`](outputs/gcdv2_realism/sources/).

## Stage 1: describe the source garment

Luna received the source render as a base64 image input through the Responses
API. It described the person/mannequin and the upper and lower garments,
including silhouette, neckline or hood, sleeves, fabric appearance, color,
seams, panels, waistband, skirt shape, and slit.

Example:

```bash
python src/tools/fit_vto_api.py fit-caption \
  --image outputs/gcdv2_realism/sources/rand_0AAMPMWVT4.png \
  --format released-code \
  --output outputs/gcdv2_realism/captions/rand_0AAMPMWVT4.txt
```

The saved captions are in
[`outputs/gcdv2_realism/captions/`](outputs/gcdv2_realism/captions/). They make
the structural constraints explicit instead of asking the image model to infer
and remember every garment detail from a generic instruction.

## Stage 2: convert the mannequin render into a photograph

The `fit-photorealistic` command sent the source render and an expanded prompt
to `/v1/images/edits`. The prompt performed two jobs:

1. Replace the mannequin or synthetic body with one plausible adult, including
   a realistic face, skin, hands, legs, and understated shoes.
2. Preserve the garment silhouette, proportions, neckline or hood, sleeves,
   cuffs, waistband, skirt length and slit, seams, panels, closures, colors,
   relative fit, pose, framing, and garment position.

It also fixed the presentation to a centered, head-to-toe fashion-catalog
photograph on a neutral studio background and prohibited text, brands,
accessories, props, and extra people.

Example:

```bash
python src/tools/fit_vto_api.py fit-photorealistic \
  --image outputs/gcdv2_realism/sources/rand_0AAMPMWVT4.png \
  --garment-description "A fitted strapless pale-aqua bodice with vertical panel seams and a matching calf-length A-line skirt." \
  --quality medium \
  --usage-output outputs/gcdv2_realism/realistic/rand_0AAMPMWVT4_usage.json \
  --prompt-output outputs/gcdv2_realism/prompts/rand_0AAMPMWVT4.txt \
  --output outputs/gcdv2_realism/realistic/rand_0AAMPMWVT4_realistic.png
```

The original three-image run predated the explicit `--quality` option and used
the gateway default. The later controlled trial used `--quality medium`.
Future dataset runs should set the quality explicitly so cost and output-token
counts are predictable.

## Stage 3: check garment fidelity

We inspected source/output pairs for:

- overall silhouette and fit;
- neckline, hood state, sleeves, and cuffs;
- waistband, skirt shape, length, and slit;
- panels, seams, closures, and original colors;
- unwanted people, accessories, text, or structural redesigns.

The three samples passed visual inspection. Luna-based comparisons gave the
first two evaluated samples a 9/10 garment-fidelity score. Small deviations
included a marginally shorter hem and slightly changed skirt drape. The hooded
sample required a targeted second pass because preserving the raised hood was
especially important.

The comparison overview is:

![GCDv2 source and photorealistic comparison](outputs/gcdv2_realism/comparisons/overview.png)

Exact prompts, individual comparisons, and QA records are under
[`outputs/gcdv2_realism/`](outputs/gcdv2_realism/).

## Stage 4: generate diverse textures

The photorealistic image became the input to a second GPT Image 2 edit. Each
request used a deliberately different material specification rather than a
generic request for "another texture." The specifications changed several
appearance dimensions together:

- base material, such as velvet, linen, leather, suede, satin, bouclé, wool,
  ripstop, nylon, or ribbed knit;
- palette;
- weave, nap, grain, quilting, moiré, embroidery, or printed motif;
- matte, satin, glossy, metallic, or softly reflective response.

The retexture prompt explicitly said to replace the previous material and
palette while keeping garment geometry, garment pieces, person identity, face,
body, pose, camera, background, and lighting unchanged. This explicit
appearance replacement is what produced substantially more diversity than
sampling a generic realism prompt repeatedly.

Two texture descriptions per garment were manually designed. A third was
proposed by Luna using the FIT-style instruction to ignore the existing
texture. Structurally incompatible Luna suggestions, such as adding new
pockets or logos, were removed before rendering.

Example:

```bash
python src/tools/fit_vto_api.py fit-retexture \
  --image outputs/gcdv2_realism/realistic/rand_0AAMPMWVT4_realistic.png \
  --texture-description "Deep emerald silk velvet with directional nap and restrained antique-gold botanical embroidery; preserve every existing seam and panel." \
  --quality medium \
  --usage-output outputs/gcdv2_texture_variants/rand_0AAMPMWVT4_usage.json \
  --prompt-output outputs/gcdv2_texture_variants/prompts/rand_0AAMPMWVT4_emerald_velvet.txt \
  --output outputs/gcdv2_texture_variants/images/rand_0AAMPMWVT4_emerald_velvet.png
```

The nine generated variants are summarized in
[`outputs/gcdv2_texture_variants/README.md`](outputs/gcdv2_texture_variants/README.md).

![GCDv2 diverse texture variants](outputs/gcdv2_texture_variants/comparisons/overview.png)

## Medium-quality trial and token usage

We repeated the emerald-velvet edit at an explicit 1024 x 1024 medium quality.
NVIDIA returned this exact usage object:

| Token category | Count |
| --- | ---: |
| Input image | 1,024 |
| Input text | 316 |
| Total input | 1,340 |
| Output image | 1,756 |
| Total | 3,096 |

At the supplied gateway rates of `$0.000005` per input token and `$0.000010`
per output token, the request cost was:

```text
(1,340 x $0.000005) + (1,756 x $0.000010) = $0.02426
```

The output, exact prompt, and usage JSON are stored in
[`outputs/gcdv2_medium_trial/`](outputs/gcdv2_medium_trial/). The trial image is
[`rand_0AAMPMWVT4_emerald_velvet_medium.png`](outputs/gcdv2_medium_trial/rand_0AAMPMWVT4_emerald_velvet_medium.png).

The CLI now prints the returned `usage` object for every image request. Passing
`--usage-output path/to/usage.json` also saves it without storing the base64 API
response. This gives exact post-request counts for `input_tokens`, split image
and text input tokens, and `output_tokens`. Before a request, output tokens can
be estimated from size and quality using OpenAI's
[image-generation cost guidance](https://developers.openai.com/api/docs/guides/image-generation),
but actual returned usage should be used for billing records.

## Scaling to five images per garment

`short_captions_v2.json` contained 128,974 garment IDs during the estimate.
Five separately prompted images per garment would therefore require 644,870
image edits. If their usage matched the medium trial exactly, the projected
inference cost would be approximately `$15,644.55`, or `$17,209` with a 10%
retry and rejection allowance.

OpenAI's Batch API officially supports `/v1/images/generations` and
`/v1/images/edits`, with asynchronous completion and separate batch limits; see
the [official Batch guide](https://developers.openai.com/api/docs/guides/batch).
The NVIDIA gateway's `/v1/batches` listing responded successfully during our
probe, but its `/v1/files` listing returned a server-side LiteLLM error. We did
not submit a large image batch, so image-batch compatibility and any NVIDIA
batch discount remain unverified.

## Resumable 10,000-garment campaign

The 10,000-output pilot uses
[`src/tools/gcdv2_realistic_batch.py`](src/tools/gcdv2_realistic_batch.py).
It keeps the original 339 GiB GCDv2 tar intact and does the following:

1. scans tar headers without unpacking the full dataset;
2. uses seeded reservoir sampling to select 10,000 distinct `default_body`
   front renders that also have entries in `short_captions_v2.json`, excluding
   records tagged `no top` or `no bottom` so the full-human edit does not create
   nudity, safety refusals, or untracked base garments;
3. extracts only those selected PNGs and freezes their archive member, byte
   offset, structural tags, prompt, and style assignment in `manifest.jsonl`;
4. uses a seeded affine permutation over 24 human specifications, 24 materials,
   24 palettes, and 20 motifs, producing 276,480 possible style packages with
   no repeated complete package in this 10,000-row run and near-uniform
   marginal counts;
5. combines photorealistic person conversion and deliberate texture replacement
   in one medium-quality image edit, avoiding the cost of a second texture pass;
6. validates every 1024 x 1024 result, commits it atomically, and stores exact
   usage in a per-image metadata file.

Each worker owns a fixed manifest chunk, takes a non-blocking per-garment lock,
and skips an already valid output. Interrupted or preempted chunks can therefore
be rerun without intentionally paying for completed images again. Corrupt files
are moved to `quarantine/` instead of being silently overwritten.

The preparation and worker scripts use CPU-only Slurm allocations charged to
`marlowe-m000133` on the `preempt` partition:

```bash
sbatch scripts/prepare_gcdv2_realistic_10k.sbatch
sbatch --array=0-39%8 scripts/run_gcdv2_realistic_10k.sbatch 250 0
```

The first argument to the worker script is chunk size; the second is an optional
per-chunk item limit used for smoke tests. Campaign progress and exact recorded
cost can be checked with:

```bash
PYTHONPATH=src /scratch/m000051/george/miniconda3/envs/aipparel/bin/python \
  -m tools.gcdv2_realistic_batch status \
  --run-dir outputs/gcdv2_realistic_10k
```

## How this differs from FIT

This experiment replaces FIT's proprietary external VLM/image-editing calls
with Luna and GPT Image 2, and it performs appearance changes directly in RGB
image space. It does **not** reproduce FIT's complete preprocessing stack:

- no Sapiens normal or segmentation estimation was rerun;
- no FIT-trained FLUX retexturing LoRA was used;
- the repository's PGF and edge-model stages were not rerun from raw GCDv2
  inputs for this sample;
- direct RGB edits can slightly change geometry, drape, identity, or small
  construction details despite strong preservation prompts.

FIT's normal/segmentation-conditioned FLUX stage provides a stronger formal
separation between garment geometry and appearance. Our GPT Image 2 workflow
is a practical substitute that produced convincing photographs and varied
materials, but every output should still pass automatic and visual fidelity
checks before being used as training data.

## Remaining complete-outfit GCDv2 campaign in Parquet

The continuation campaign uses
[`src/tools/gcdv2_realistic_parquet.py`](src/tools/gcdv2_realistic_parquet.py)
to avoid creating a source PNG, generated PNG, and metadata JSON for every
garment. GCDv2 has 128,974 captioned `default_body` front renders in this
archive: 56,119 complete outfits, 38,701 lower-body-only garments tagged
`no top`, and 34,154 upper-body-only garments tagged `no bottom`. This campaign
uses only the complete-outfit population. After excluding the 10,000 IDs in
the pilot manifest, 46,119 garments remain. The separate-garment records are
left for a later campaign.

The storage layout is:

```text
outputs/gcdv2_realistic_remaining_parquet/
  config.json
  manifest.parquet
  parquet/part-000000.parquet ...
  logs/worker_*.events.jsonl
  staging/worker_*.sqlite3
  slurm/edit-*.out, edit-*.err
```

Each output Parquet row stores the garment ID, archive offset, caption tags,
human and textile assignment, prompt and hashes, model settings, exact token
usage, status, and the generated PNG in the binary `image_bytes` column.
Parts contain 64 manifest rows, yielding 721 output files instead of roughly
92,000 separate image and metadata files. Source images are read directly by
offset from the original tar and are never extracted into the run directory.

Every paid result is first committed transactionally to one small SQLite file
per worker. Once a part is ready, the worker writes and verifies a temporary
Parquet file and atomically renames it into place. A restart loads both
committed Parquet and staged rows, skips completed images, and processes only
missing records. Workers stop safely after 46 hours and requeue their own
array element before the 47.5-hour batch limit if more assigned shards remain.

The jobs are CPU-only and use the verified medium allocation account on the
non-preemptible batch partition:

```bash
sbatch scripts/prepare_gcdv2_realistic_remaining_parquet.sbatch

# Two-row live smoke test using production shard ownership.
sbatch --array=0-0 \
  scripts/run_gcdv2_realistic_remaining_parquet.sbatch 2 20 0

# Twenty persistent CPU workers; 0 means no item cap.
sbatch --array=0-19%20 \
  scripts/run_gcdv2_realistic_remaining_parquet.sbatch 0 20 0
```

Check progress without decoding all image blobs:

```bash
PYTHONPATH=src /scratch/m000051/george/miniconda3/envs/aipparel/bin/python \
  -m tools.gcdv2_realistic_parquet status \
  --run-dir outputs/gcdv2_realistic_remaining_parquet
```

Read image bytes from one part:

```python
from io import BytesIO
from PIL import Image
import pyarrow.parquet as pq

table = pq.read_table(
    "outputs/gcdv2_realistic_remaining_parquet/parquet/part-000000.parquet",
    columns=["garment_id", "status", "image_bytes"],
)
row = next(item for item in table.to_pylist() if item["status"] == "completed")
image = Image.open(BytesIO(row["image_bytes"]))
```

Using the pilot's measured mean of $0.02486842 per successful medium-quality
image, 46,119 successful outputs project to $1,146.91. A 10% allowance is
$1,261.60. The pilot's mean 1.19 MB PNG size projects to about 51 GiB of image
payload before small Parquet metadata overhead.

## Reproducibility checklist

1. Use the inference credential, never the final management credential.
2. Keep source IDs, captions, expanded prompts, images, and usage JSON together.
3. Set `--size` and `--quality` explicitly; use `1024x1024` and `medium` for the
   measured configuration above.
4. Use a distinct, concrete texture description for every desired appearance.
5. Preserve structure and person identity explicitly in every edit prompt.
6. Reject outputs with silhouette, garment-piece, seam, closure, length, slit,
   hood, identity, or coverage errors.
7. Record returned token usage and calculate cost from the actual gateway rate.
8. Run the client tests after changes:

   ```bash
   python -m unittest discover -s tests -v
   ```
