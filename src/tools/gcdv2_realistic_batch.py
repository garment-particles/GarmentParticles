#!/usr/bin/env python3
"""Prepare and run a resumable 10k GCDv2-to-photorealistic image campaign.

The input archive is kept compressed as a tar file. ``prepare`` scans its tar
headers, uniformly samples real front renders, records their byte offsets, and
extracts only the selected PNGs. ``run`` performs exactly one GPT Image edit
per selected garment, combining mannequin-to-human conversion and deliberate
surface retexturing in a single paid request. ``status`` validates outputs and
sums the exact token usage returned by the gateway.

No credential is copied into the run directory. The gateway client reads the
configured key file only when it sends a request.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import random
import sys
import tarfile
import time
from typing import Any, Iterator, Optional, Sequence

from PIL import Image, UnidentifiedImageError

from tools.fit_vto_api import (
    DEFAULT_CREDENTIAL_INDEX_FROM_END,
    DEFAULT_IMAGE_BASE_URL,
    DEFAULT_IMAGE_MODEL,
    DEFAULT_KEY_FILE,
    GatewayClient,
    GatewayError,
    image_items,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = Path(
    "/scratch/m000051-pm06/george/gcdv2_archives/garmentcodedatav2.tar"
)
DEFAULT_CAPTIONS = Path("/scratch/m000051/george/gcdv2/short_captions_v2.json")
DEFAULT_RUN_DIR = REPO_ROOT / "outputs/gcdv2_realistic_10k"
DEFAULT_COUNT = 10_000
DEFAULT_SEED = 20_260_819
DEFAULT_QUALITY = "medium"
DEFAULT_SIZE = "1024x1024"
INPUT_TOKEN_RATE = 0.000005
OUTPUT_TOKEN_RATE = 0.000010
SCHEMA_VERSION = "gcdv2_realistic_edit_v1"
PROMPT_VERSION = "gcdv2_realistic_prompt_v2"
STYLE_VERSION = "affine_cartesian_style_v2"
STYLE_MULTIPLIER = 104_729
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


HUMAN_SPECS = (
    "a woman in her late 20s with deep-brown skin, a tall athletic build, and a short natural coily crop",
    "a man in his early 30s with fair freckled skin, a lean build, and short wavy auburn hair",
    "a woman in her 40s with medium-brown skin, a softly curvy build, and long dark wavy hair",
    "an androgynous nonbinary adult in their late 20s with warm olive skin, an average build, and cropped black hair",
    "a woman in her early 60s with light skin, a medium build, and a neat silver bob",
    "a man in his 50s with deep-brown skin, a broad build, close-cropped salt-and-pepper hair, and a trimmed beard",
    "a woman in her early 30s with golden-brown skin, a petite build, and thick dark curls",
    "a man in his late 20s with light-medium skin, a slender build, and straight black hair in a short side part",
    "a woman in her 50s with olive-brown skin, a statuesque build, and shoulder-length silver-streaked waves",
    "an androgynous nonbinary adult in their 30s with dark skin, a compact athletic build, and a close shaved hairstyle",
    "a woman in her mid-20s with fair skin, a full-figured build, and long copper hair",
    "a man in his early 40s with tan skin, an average build, and short dark curls",
    "a woman in her late 30s with dark-brown skin, a slender build, and long micro-braids",
    "a man in his 60s with medium skin, a lean build, and short silver hair",
    "a woman in her early 20s with light-medium skin, an athletic build, and a blunt black bob",
    "an androgynous nonbinary adult in their 40s with fair skin, a tall build, and short sandy-blond curls",
    "a woman in her mid-50s with deep skin, a curvy build, and a natural gray afro",
    "a man in his mid-20s with olive skin, a slim build, and shoulder-length wavy dark hair",
    "a woman in her early 40s with warm beige skin, an average build, and straight dark hair in a low ponytail",
    "a man in his late 30s with rich brown skin, an athletic build, and short twists",
    "a woman in her late 20s with tan skin, a petite build, and loose dark curls",
    "an androgynous nonbinary adult in their 50s with medium-brown skin, a sturdy build, and short gray curls",
    "a woman in her 60s with pale freckled skin, a slender build, and swept-back white hair",
    "a man in his early 30s with golden-brown skin, a tall lean build, and a shaved head",
)

MATERIALS = (
    "washed linen with a visible natural slub weave",
    "plush silk velvet with directional nap",
    "fine cotton poplin with crisp micro-weave",
    "matte wool crepe with a softly pebbled hand",
    "boucle tweed with looped multitone yarns",
    "fluid silk satin with restrained luster",
    "supple matte lambskin with fine natural grain",
    "brushed suede with soft directional texture",
    "midweight denim with visible diagonal twill",
    "ribbed merino knit with narrow vertical ribs",
    "raised jacquard brocade with woven relief",
    "lightly quilted technical nylon",
    "crisp silk taffeta with subtle shot-color sheen",
    "soft chambray with fine crosshatch texture",
    "narrow-wale cotton corduroy",
    "raw silk dupioni with irregular horizontal slubs",
    "structured scuba jersey with a smooth matte face",
    "brushed herringbone wool",
    "embroidered cotton voile over a fully opaque lining",
    "fine lace overlay over a fully opaque tonal base",
    "recycled ripstop with a visible micro-grid",
    "soft metallic lame with a controlled low-glare shimmer",
    "coated cotton canvas with a waxed matte finish",
    "silk georgette over a fully opaque color-matched lining",
)

PALETTES = (
    "deep emerald, forest green, and restrained antique gold",
    "cobalt blue, ultramarine, and cool silver",
    "burnt orange, rust, and dark chocolate",
    "plum, aubergine, and muted rose",
    "ivory, sand, and warm camel",
    "charcoal, graphite, and soft black",
    "crimson, oxblood, and blush pink",
    "teal, petrol blue, and pale aqua",
    "mustard, ochre, and espresso brown",
    "lavender, violet, and midnight purple",
    "terracotta, coral, and cream",
    "navy, powder blue, and pearl gray",
    "sage, moss, and warm stone",
    "magenta, burgundy, and dusty mauve",
    "copper, bronze, and deep umber",
    "black, winter white, and a small scarlet accent",
    "indigo, faded blue, and tobacco brown",
    "pistachio, mint, and dark spruce",
    "sunflower yellow, marigold, and deep navy",
    "peacock blue, turquoise, and dark jade",
    "clay red, paprika, and muted peach",
    "cool taupe, mushroom, and off-white",
    "raspberry, wine, and smoky gray",
    "ice blue, steel blue, and ink black",
)

MOTIFS = (
    "a clean tonal treatment with no print",
    "a small-scale windowpane check aligned across existing panels",
    "fine pinstripes following the garment grain",
    "a restrained botanical embroidery placed inside existing panels",
    "a woven Art Deco fan motif at tailoring scale",
    "a subtle ombre that moves from dark at the hem to light near the neckline",
    "a miniature houndstooth pattern",
    "an irregular hand-blocked floral print",
    "a quiet geometric jacquard of interlocking diamonds",
    "tonal topographic contour lines",
    "a sparse constellation of tiny embroidered dots",
    "a watercolor-like abstract brushstroke print",
    "a traditional herringbone arrangement",
    "a small gingham check",
    "a gradient color-block treatment contained by the existing seams",
    "a delicate vine motif woven into the surface",
    "a modern micro-chevron pattern",
    "a scattered terrazzo-style speckle print",
    "a subtle moire wave effect",
    "a sparse metallic thread highlight along existing seam lines",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def atomic_write_text(path: Path, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))


def stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_member_info(
    member_name: str, body_kind: str = "default_body"
) -> Optional[tuple[str, str]]:
    """Return ``(batch, garment_id)`` for a GCDv2 front-render member."""

    parts = PurePosixPath(member_name).parts
    if len(parts) != 5:
        return None
    root, batch, body, garment_id, filename = parts
    if root != "garmentcodedatav2" or body != body_kind:
        return None
    if not batch.startswith("garments_5000_") or not garment_id.startswith("rand_"):
        return None
    if filename != f"{garment_id}_render_front.png":
        return None
    return batch, garment_id


def style_for_index(index: int, seed: int = DEFAULT_SEED) -> dict[str, str]:
    """Return a unique, deterministic point in the Cartesian style space.

    An affine permutation avoids the coupling caused by independently striding
    same-length lists. ``STYLE_MULTIPLIER`` is coprime to the full style-space
    size, so no complete package repeats before all combinations are visited.
    """

    if index < 0:
        raise ValueError("style index must be non-negative")
    style_space = (
        len(HUMAN_SPECS) * len(MATERIALS) * len(PALETTES) * len(MOTIFS)
    )
    offset = int(stable_digest(f"style-offset:{seed}"), 16) % style_space
    code = (STYLE_MULTIPLIER * index + offset) % style_space
    style_code = code
    human_index = code % len(HUMAN_SPECS)
    code //= len(HUMAN_SPECS)
    material_index = code % len(MATERIALS)
    code //= len(MATERIALS)
    palette_index = code % len(PALETTES)
    code //= len(PALETTES)
    motif_index = code % len(MOTIFS)

    human = HUMAN_SPECS[human_index]
    material = MATERIALS[material_index]
    palette = PALETTES[palette_index]
    motif = MOTIFS[motif_index]
    texture = (
        f"{material}, using a palette of {palette}, with {motif}. "
        "Keep pattern scale plausible and continue it cleanly across existing seams."
    )
    return {
        "style_version": STYLE_VERSION,
        "style_code": str(style_code),
        "human": human,
        "material": material,
        "palette": palette,
        "motif": motif,
        "texture": texture,
    }


def build_edit_prompt(
    structural_tags: Sequence[str], human_spec: str, texture_spec: str
) -> str:
    description = ", ".join(str(tag).strip() for tag in structural_tags if str(tag).strip())
    return f"""Use case: faithful GCDv2 render conversion and garment surface redesign.
Asset type: one photorealistic, full-body fashion catalog photograph.

Edit the supplied synthetic front render. Replace the synthetic mannequin and body rendering with exactly one {human_spec}. Keep all hair behind the shoulders and away from garment details. Make the face, skin, eyes, hands, legs, and feet anatomically plausible and photographically realistic.

Garment construction labels: {description}.
Target textile design: {texture_spec}

Actively replace the source garment's old color and synthetic surface with the target textile design. Render physically plausible weave, grain, nap, print or embroidery scale, reflectance, folds, wrinkles, and shadows. Apply the coordinated design to every visible garment piece.

Preserve exactly the source garment geometry and construction: number of pieces, silhouette, proportions, volume, fit, neckline, collar or hood state, sleeves and cuffs, waistband, skirt or trouser shape and length, slits, hems, panels, seam locations, and closures. Do not add, remove, shorten, lengthen, or redesign any garment part. The construction labels clarify the image but never override visible source geometry.

Keep the source front-facing pose and garment placement. If the source crop omits the head, legs, hands, or feet, outpaint and recompose it as one complete head-to-toe adult in the square canvas while preserving the visible garment's scale, proportions, and construction. Do not crop the final head or feet. Use a seamless neutral light-gray studio background and soft catalog lighting. No extra people, props, jewelry, bag, hat, visible brand, text, logo, or watermark."""


def load_captions(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    result: dict[str, list[str]] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, list):
            tags = [str(item).strip() for item in value if str(item).strip()]
            if tags:
                result[key] = tags
    return result


def scan_archive_sample(
    archive_path: Path,
    captions: dict[str, list[str]],
    count: int,
    seed: int,
    body_kind: str,
    require_complete_outfit: bool = True,
) -> tuple[list[dict[str, Any]], int, int]:
    """Reservoir-sample eligible PNG members without retaining all tar headers."""

    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    eligible = 0
    scanned = 0
    seen_ids: set[str] = set()

    with tarfile.open(archive_path, mode="r:") as archive:
        while True:
            member = archive.next()
            if member is None:
                break
            scanned += 1
            info = source_member_info(member.name, body_kind=body_kind)
            if info is not None and member.isfile():
                batch, garment_id = info
                if garment_id in captions and garment_id not in seen_ids:
                    seen_ids.add(garment_id)
                    lowered_tags = {
                        tag.strip().lower() for tag in captions[garment_id]
                    }
                    if require_complete_outfit and lowered_tags.intersection(
                        {"no top", "no bottom"}
                    ):
                        archive.members.clear()
                        continue
                    eligible += 1
                    record = {
                        "garment_id": garment_id,
                        "batch": batch,
                        "body_kind": body_kind,
                        "source_member": member.name,
                        "source_offset": int(member.offset_data),
                        "source_size": int(member.size),
                        "structural_tags": captions[garment_id],
                    }
                    if len(selected) < count:
                        selected.append(record)
                    else:
                        replacement = rng.randrange(eligible)
                        if replacement < count:
                            selected[replacement] = record
            if scanned % 100_000 == 0:
                print(
                    f"tar_members_scanned={scanned} eligible_front_renders={eligible}",
                    flush=True,
                )
            # TarFile otherwise retains every TarInfo object it has seen.
            archive.members.clear()

    if eligible < count:
        raise ValueError(
            f"Archive has only {eligible} eligible {body_kind} front renders; requested {count}"
        )
    selected.sort(
        key=lambda item: stable_digest(f"{seed}:{item['garment_id']}")
    )
    return selected, eligible, scanned


def make_manifest_records(
    sampled: Sequence[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, source in enumerate(sampled):
        garment_id = str(source["garment_id"])
        style = style_for_index(index, seed=seed)
        prompt = build_edit_prompt(
            source["structural_tags"], style["human"], style["texture"]
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "prompt_version": PROMPT_VERSION,
                "style_version": style["style_version"],
                "style_code": style["style_code"],
                "index": index,
                **source,
                "source_path": f"sources/{garment_id}.png",
                "output_path": f"images/{garment_id}.png",
                "metadata_path": f"metadata/{garment_id}.json",
                "human_spec": style["human"],
                "texture_spec": style["texture"],
                "material": style["material"],
                "palette": style["palette"],
                "motif": style["motif"],
                "prompt": prompt,
                "prompt_sha256": stable_digest(prompt),
                "selection_seed": seed,
            }
        )
    return records


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    value = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    atomic_write_text(path, value)


def load_manifest(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run prepare first")
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for expected_index, record in enumerate(records):
        if record.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported manifest schema at row {expected_index}")
        if record.get("prompt_version") != PROMPT_VERSION:
            raise ValueError(
                f"Stale prompt version at row {expected_index}; run refresh-prompts"
            )
        if record.get("style_version") != STYLE_VERSION:
            raise ValueError(
                f"Stale style version at row {expected_index}; run refresh-styles"
            )
        if record.get("index") != expected_index:
            raise ValueError(f"Non-contiguous manifest index at row {expected_index}")
    return records


def png_file_looks_complete(path: Path, expected_bytes: Optional[int] = None) -> bool:
    try:
        if not path.is_file():
            return False
        if expected_bytes is not None and path.stat().st_size != expected_bytes:
            return False
        with path.open("rb") as handle:
            return handle.read(len(PNG_SIGNATURE)) == PNG_SIGNATURE
    except OSError:
        return False


def validate_image(path: Path, expected_size: Optional[tuple[int, int]] = None) -> tuple[bool, str]:
    try:
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG":
                return False, f"format={image.format}"
            if expected_size is not None and image.size != expected_size:
                return False, f"size={image.size}"
            if image.width < 64 or image.height < 64:
                return False, f"implausible size={image.size}"
        return True, "ok"
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        return False, str(exc)


def extract_selected_sources(
    archive_path: Path, run_dir: Path, records: Sequence[dict[str, Any]]
) -> tuple[int, int]:
    extracted = 0
    skipped = 0
    ordered = sorted(records, key=lambda record: int(record["source_offset"]))
    with archive_path.open("rb") as archive:
        for number, record in enumerate(ordered, 1):
            destination = run_dir / str(record["source_path"])
            expected_bytes = int(record["source_size"])
            if png_file_looks_complete(destination, expected_bytes=expected_bytes):
                skipped += 1
                continue
            archive.seek(int(record["source_offset"]))
            value = archive.read(expected_bytes)
            if len(value) != expected_bytes or not value.startswith(PNG_SIGNATURE):
                raise RuntimeError(
                    f"Could not read a complete PNG for {record['garment_id']} from tar offset"
                )
            atomic_write_bytes(destination, value)
            extracted += 1
            if extracted == 1 or extracted % 250 == 0:
                print(
                    f"extracted={extracted} skipped={skipped} scanned_selected={number}/{len(ordered)}",
                    flush=True,
                )
    return extracted, skipped


def prepare_run(args: argparse.Namespace) -> int:
    args.run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading captions from {args.captions}", flush=True)
    captions = load_captions(args.captions)
    print(
        f"Scanning tar headers in {args.archive} for a deterministic reservoir sample",
        flush=True,
    )
    sampled, eligible, scanned = scan_archive_sample(
        args.archive,
        captions,
        args.count,
        args.seed,
        args.body_kind,
        require_complete_outfit=args.require_complete_outfit,
    )
    records = make_manifest_records(sampled, args.seed)
    write_jsonl(args.run_dir / "manifest.jsonl", records)
    archive_stat = args.archive.stat()
    config = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "archive": str(args.archive.resolve()),
        "archive_size": archive_stat.st_size,
        "archive_mtime_ns": archive_stat.st_mtime_ns,
        "captions": str(args.captions.resolve()),
        "caption_count": len(captions),
        "body_kind": args.body_kind,
        "require_complete_outfit": args.require_complete_outfit,
        "selection_seed": args.seed,
        "selected_count": len(records),
        "eligible_count": eligible,
        "tar_members_scanned": scanned,
        "image_model": DEFAULT_IMAGE_MODEL,
        "quality": DEFAULT_QUALITY,
        "prompt_version": PROMPT_VERSION,
        "style_version": STYLE_VERSION,
        "size": DEFAULT_SIZE,
        "input_token_rate": INPUT_TOKEN_RATE,
        "output_token_rate": OUTPUT_TOKEN_RATE,
    }
    atomic_write_text(
        args.run_dir / "config.json", json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    if args.extract_sources:
        extracted, skipped = extract_selected_sources(args.archive, args.run_dir, records)
    else:
        extracted, skipped = 0, 0
    print(
        json.dumps(
            {
                "selected": len(records),
                "eligible": eligible,
                "tar_members_scanned": scanned,
                "sources_extracted": extracted,
                "sources_already_present": skipped,
                "manifest": str((args.run_dir / "manifest.jsonl").resolve()),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def refresh_manifest_prompts(args: argparse.Namespace) -> int:
    path = args.run_dir / "manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run prepare first")
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for expected_index, record in enumerate(records):
        if record.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported manifest schema at row {expected_index}")
        if record.get("index") != expected_index:
            raise ValueError(f"Non-contiguous manifest index at row {expected_index}")
        prompt = build_edit_prompt(
            record["structural_tags"], record["human_spec"], record["texture_spec"]
        )
        record["prompt_version"] = PROMPT_VERSION
        record["prompt"] = prompt
        record["prompt_sha256"] = stable_digest(prompt)
    write_jsonl(path, records)

    config_path = args.run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["prompt_version"] = PROMPT_VERSION
    config["prompts_refreshed_at"] = utc_now()
    atomic_write_text(
        config_path, json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "manifest": str(path.resolve()),
                "records_refreshed": len(records),
                "prompt_version": PROMPT_VERSION,
            },
            indent=2,
        )
    )
    return 0


def refresh_manifest_styles(args: argparse.Namespace) -> int:
    """Reassign decoupled styles and preserve all outputs from the old scheme."""

    manifest_path = args.run_dir / "manifest.jsonl"
    config_path = args.run_dir / "config.json"
    if not manifest_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("Missing manifest.jsonl or config.json; run prepare first")
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed = int(config.get("selection_seed", DEFAULT_SEED))

    for expected_index, record in enumerate(records):
        if record.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported manifest schema at row {expected_index}")
        if record.get("index") != expected_index:
            raise ValueError(f"Non-contiguous manifest index at row {expected_index}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive_root = args.run_dir / "superseded" / f"linear_style_v1_{stamp}"
    moved: dict[str, str] = {}
    for name in ("images", "metadata", "logs"):
        source = args.run_dir / name
        if source.exists():
            destination = archive_root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
            moved[name] = str(destination.relative_to(args.run_dir))
    status_path = args.run_dir / "status.json"
    if status_path.exists():
        destination = archive_root / "status.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(status_path, destination)
        moved["status"] = str(destination.relative_to(args.run_dir))

    for index, record in enumerate(records):
        style = style_for_index(index, seed=seed)
        prompt = build_edit_prompt(
            record["structural_tags"], style["human"], style["texture"]
        )
        record.update(
            {
                "style_version": style["style_version"],
                "style_code": style["style_code"],
                "human_spec": style["human"],
                "texture_spec": style["texture"],
                "material": style["material"],
                "palette": style["palette"],
                "motif": style["motif"],
                "prompt_version": PROMPT_VERSION,
                "prompt": prompt,
                "prompt_sha256": stable_digest(prompt),
            }
        )

    write_jsonl(manifest_path, records)
    config["style_version"] = STYLE_VERSION
    config["styles_refreshed_at"] = utc_now()
    config["superseded_artifacts"] = moved
    atomic_write_text(
        config_path, json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    migration = {
        "created_at": utc_now(),
        "reason": "replace correlated linear style strides with unique affine Cartesian assignments",
        "new_style_version": STYLE_VERSION,
        "records_refreshed": len(records),
        "moved": moved,
    }
    atomic_write_text(
        archive_root / "migration.json",
        json.dumps(migration, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(migration | {"archive_root": str(archive_root.resolve())}, indent=2))
    return 0


def expected_dimensions(size: str) -> tuple[int, int]:
    try:
        width, height = size.lower().split("x", 1)
        return int(width), int(height)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid image size {size!r}; expected WIDTHxHEIGHT") from exc


def normalized_png_bytes(value: bytes, expected_size: tuple[int, int]) -> bytes:
    try:
        with Image.open(BytesIO(value)) as image:
            image.load()
            if image.size != expected_size:
                raise ValueError(
                    f"Gateway image size {image.size} does not match {expected_size}"
                )
            converted = image.convert("RGB")
            output = BytesIO()
            converted.save(output, format="PNG")
            return output.getvalue()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Gateway returned an undecodable image: {exc}") from exc


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def nonblocking_item_lock(path: Path) -> Iterator[bool]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def quarantine_invalid_output(run_dir: Path, output: Path, garment_id: str) -> Optional[Path]:
    if not output.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = run_dir / "quarantine" / f"{garment_id}.{stamp}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(output, destination)
    return destination


def metadata_for_result(
    record: dict[str, Any], args: argparse.Namespace, payload: dict[str, Any], attempt: int
) -> dict[str, Any]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "index": record["index"],
        "garment_id": record["garment_id"],
        "source_member": record["source_member"],
        "source_path": record["source_path"],
        "output_path": record["output_path"],
        "human_spec": record["human_spec"],
        "texture_spec": record["texture_spec"],
        "prompt_sha256": record["prompt_sha256"],
        "model": args.image_model,
        "quality": args.quality,
        "size": args.size,
        "attempt_in_process": attempt,
        "usage": usage,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def run_one_item(
    client: GatewayClient,
    run_dir: Path,
    record: dict[str, Any],
    args: argparse.Namespace,
    event_path: Path,
) -> str:
    garment_id = str(record["garment_id"])
    source = run_dir / str(record["source_path"])
    output = run_dir / str(record["output_path"])
    metadata_path = run_dir / str(record["metadata_path"])
    expected_size = expected_dimensions(args.size)
    lock_path = run_dir / "locks" / f"{garment_id}.lock"

    with nonblocking_item_lock(lock_path) as acquired:
        if not acquired:
            return "locked"
        valid, _ = validate_image(output, expected_size=expected_size)
        if valid:
            return "skipped"
        if not source.is_file():
            raise FileNotFoundError(f"Missing source render {source}")
        source_valid, source_reason = validate_image(source)
        if not source_valid:
            raise ValueError(f"Invalid source render {source}: {source_reason}")
        quarantined = quarantine_invalid_output(run_dir, output, garment_id)
        if quarantined is not None:
            append_event(
                event_path,
                {
                    "time": utc_now(),
                    "event": "quarantined_invalid_output",
                    "index": record["index"],
                    "garment_id": garment_id,
                    "path": str(quarantined.relative_to(run_dir)),
                },
            )

        last_error: Optional[Exception] = None
        for attempt in range(1, args.item_attempts + 1):
            started = time.monotonic()
            try:
                payload = client.edit_image(
                    source,
                    str(record["prompt"]),
                    n=1,
                    size=args.size,
                    quality=args.quality,
                )
                items = image_items(payload)
                if len(items) != 1:
                    raise GatewayError(f"Expected one image result, received {len(items)}")
                raw = client.image_bytes(items[0])
                png = normalized_png_bytes(raw, expected_size)
                atomic_write_bytes(output, png)
                valid, reason = validate_image(output, expected_size=expected_size)
                if not valid:
                    raise ValueError(f"Committed output did not validate: {reason}")
            except (GatewayError, OSError, ValueError) as exc:
                last_error = exc
                append_event(
                    event_path,
                    {
                        "time": utc_now(),
                        "event": "attempt_failed",
                        "index": record["index"],
                        "garment_id": garment_id,
                        "attempt": attempt,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "error": str(exc)[:2000],
                    },
                )
                if attempt < args.item_attempts:
                    time.sleep(min(120.0, 15.0 * (2 ** (attempt - 1))))
                continue

            # The paid result is durably committed at this point. Metadata or
            # event-ledger failures must never cause another image request.
            metadata = metadata_for_result(record, args, payload, attempt)
            try:
                atomic_write_text(
                    metadata_path, json.dumps(metadata, indent=2, sort_keys=True) + "\n"
                )
                append_event(
                    event_path,
                    {
                        "time": utc_now(),
                        "event": "completed",
                        "index": record["index"],
                        "garment_id": garment_id,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "usage": metadata["usage"],
                    },
                )
            except OSError as exc:
                print(
                    f"WARNING output committed but metadata logging failed for {garment_id}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            return "completed"
        assert last_error is not None
        raise last_error


def run_chunk(args: argparse.Namespace) -> int:
    records = load_manifest(args.run_dir)
    if args.chunk_size < 1 or args.chunk_index < 0:
        raise ValueError("chunk-size must be positive and chunk-index non-negative")
    start = args.chunk_index * args.chunk_size
    end = min(len(records), start + args.chunk_size)
    if start >= len(records):
        raise ValueError(
            f"Chunk {args.chunk_index} starts at {start}, beyond {len(records)} records"
        )
    selected = records[start:end]
    if args.max_items > 0:
        selected = selected[: args.max_items]
    event_path = (
        args.run_dir
        / "logs"
        / f"chunk_{args.chunk_index:04d}_size_{args.chunk_size}.events.jsonl"
    )
    client = GatewayClient(
        key_file=args.key_file,
        image_base_url=args.image_base_url,
        image_model=args.image_model,
        credential_index_from_end=args.credential_index_from_end,
        timeout=args.timeout,
        max_attempts=args.request_attempts,
    )
    counts = {"completed": 0, "skipped": 0, "locked": 0, "failed": 0}
    print(
        f"chunk={args.chunk_index} range=[{start},{end}) selected={len(selected)} model={args.image_model} quality={args.quality}",
        flush=True,
    )
    for position, record in enumerate(selected, 1):
        try:
            result = run_one_item(client, args.run_dir, record, args, event_path)
            counts[result] += 1
            print(
                f"chunk={args.chunk_index} progress={position}/{len(selected)} index={record['index']} id={record['garment_id']} result={result}",
                flush=True,
            )
        except (GatewayError, OSError, ValueError) as exc:
            counts["failed"] += 1
            append_event(
                event_path,
                {
                    "time": utc_now(),
                    "event": "item_failed",
                    "index": record["index"],
                    "garment_id": record["garment_id"],
                    "error": str(exc)[:2000],
                },
            )
            print(
                f"ERROR index={record['index']} id={record['garment_id']}: {exc}",
                file=sys.stderr,
                flush=True,
            )
    print(json.dumps(counts, sort_keys=True), flush=True)
    return 2 if counts["failed"] else 0


def usage_counts(metadata: dict[str, Any]) -> tuple[int, int]:
    usage = metadata.get("usage", {})
    if not isinstance(usage, dict):
        return 0, 0
    try:
        return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
    except (TypeError, ValueError):
        return 0, 0


def campaign_status(args: argparse.Namespace) -> int:
    records = load_manifest(args.run_dir)
    dimensions = expected_dimensions(args.size)
    complete = 0
    invalid = 0
    metadata_present = 0
    input_tokens = 0
    output_tokens = 0
    for record in records:
        output = args.run_dir / str(record["output_path"])
        if output.exists():
            valid, _ = validate_image(output, expected_size=dimensions)
            if valid:
                complete += 1
            else:
                invalid += 1
        metadata_path = args.run_dir / str(record["metadata_path"])
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                item_input, item_output = usage_counts(metadata)
                input_tokens += item_input
                output_tokens += item_output
                metadata_present += 1
            except (OSError, ValueError, TypeError):
                pass
    cost = input_tokens * args.input_token_rate + output_tokens * args.output_token_rate
    result = {
        "schema_version": SCHEMA_VERSION,
        "checked_at": utc_now(),
        "run_dir": str(args.run_dir.resolve()),
        "total": len(records),
        "complete": complete,
        "missing": len(records) - complete - invalid,
        "invalid": invalid,
        "metadata_present": metadata_present,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "exact_recorded_cost_usd": round(cost, 6),
        "input_token_rate": args.input_token_rate,
        "output_token_rate": args.output_token_rate,
        "progress_percent": round(100.0 * complete / len(records), 3) if records else 0.0,
    }
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(serialized, end="")
    if args.json_output is not None:
        atomic_write_text(args.json_output, serialized)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Select and extract source renders")
    prepare.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    prepare.add_argument("--captions", type=Path, default=DEFAULT_CAPTIONS)
    prepare.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    prepare.add_argument("--count", type=int, default=DEFAULT_COUNT)
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare.add_argument("--body-kind", default="default_body")
    prepare.add_argument(
        "--allow-separates",
        action="store_false",
        dest="require_complete_outfit",
        help=(
            "Allow records tagged 'no top' or 'no bottom'. By default they are "
            "excluded to avoid nudity, safety refusals, and invented base garments."
        ),
    )
    prepare.add_argument(
        "--no-extract-sources",
        action="store_false",
        dest="extract_sources",
        help="Write the manifest without extracting its selected PNGs.",
    )
    prepare.set_defaults(extract_sources=True, require_complete_outfit=True)

    refresh = commands.add_parser(
        "refresh-prompts",
        help="Rebuild frozen prompts without rescanning or reselecting sources",
    )
    refresh.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)

    refresh_styles = commands.add_parser(
        "refresh-styles",
        help="Apply decoupled style assignments and archive old completed outputs",
    )
    refresh_styles.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)

    run = commands.add_parser("run", help="Run one restart-safe manifest chunk")
    run.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    run.add_argument(
        "--chunk-index",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
    )
    run.add_argument("--chunk-size", type=int, default=250)
    run.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Process at most this many rows in the chunk; zero means all.",
    )
    run.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    run.add_argument(
        "--credential-index-from-end",
        type=int,
        default=DEFAULT_CREDENTIAL_INDEX_FROM_END,
    )
    run.add_argument("--image-base-url", default=DEFAULT_IMAGE_BASE_URL)
    run.add_argument("--image-model", default=DEFAULT_IMAGE_MODEL)
    run.add_argument("--quality", choices=("low", "medium", "high", "auto"), default=DEFAULT_QUALITY)
    run.add_argument("--size", default=DEFAULT_SIZE)
    run.add_argument("--timeout", type=float, default=600.0)
    run.add_argument("--request-attempts", type=int, default=4)
    run.add_argument("--item-attempts", type=int, default=2)

    status = commands.add_parser("status", help="Validate outputs and total exact usage")
    status.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    status.add_argument("--size", default=DEFAULT_SIZE)
    status.add_argument("--input-token-rate", type=float, default=INPUT_TOKEN_RATE)
    status.add_argument("--output-token-rate", type=float, default=OUTPUT_TOKEN_RATE)
    status.add_argument("--json-output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            return prepare_run(args)
        if args.command == "run":
            return run_chunk(args)
        if args.command == "refresh-prompts":
            return refresh_manifest_prompts(args)
        if args.command == "refresh-styles":
            return refresh_manifest_styles(args)
        if args.command == "status":
            return campaign_status(args)
        raise AssertionError(f"Unhandled command {args.command}")
    except (FileNotFoundError, GatewayError, OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
