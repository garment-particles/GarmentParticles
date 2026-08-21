#!/usr/bin/env python3
"""Generate the remaining GCDv2 realistic images into resumable Parquet shards.

The source PNGs are read directly from byte offsets in the original tar archive,
so this campaign does not create one source file per garment. Generated PNG
bytes and their usage metadata are grouped into atomic Parquet parts. A small
SQLite database per worker durably stages paid responses until their Parquet
part has been committed, preventing an interrupted worker from paying for the
same successful response again.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import tarfile
import time
from typing import Any, Iterable, Optional, Sequence

from PIL import Image, UnidentifiedImageError
import pyarrow as pa
import pyarrow.parquet as pq

from tools.fit_vto_api import (
    DEFAULT_CREDENTIAL_INDEX_FROM_END,
    DEFAULT_IMAGE_BASE_URL,
    DEFAULT_IMAGE_MODEL,
    DEFAULT_KEY_FILE,
    GatewayClient,
    GatewayError,
    image_items,
)
from tools.gcdv2_realistic_batch import (
    DEFAULT_ARCHIVE,
    DEFAULT_CAPTIONS,
    DEFAULT_QUALITY,
    DEFAULT_SEED,
    DEFAULT_SIZE,
    INPUT_TOKEN_RATE,
    OUTPUT_TOKEN_RATE,
    PNG_SIGNATURE,
    PROMPT_VERSION,
    STYLE_VERSION,
    append_event,
    atomic_write_text,
    build_edit_prompt,
    expected_dimensions,
    load_captions,
    normalized_png_bytes,
    source_member_info,
    stable_digest,
    style_for_index,
    utc_now,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = REPO_ROOT / "outputs/gcdv2_realistic_remaining_parquet"
DEFAULT_EXCLUDE_MANIFEST = (
    REPO_ROOT / "outputs/gcdv2_realistic_10k/manifest.jsonl"
)
PARQUET_SCHEMA_VERSION = "gcdv2_realistic_parquet_v1"
DEFAULT_ROWS_PER_SHARD = 64
DEFAULT_STYLE_INDEX_OFFSET = 10_000
DEFAULT_WORKER_COUNT = 20


def manifest_schema() -> pa.Schema:
    return pa.schema(
        [
            ("schema_version", pa.string()),
            ("prompt_version", pa.string()),
            ("style_version", pa.string()),
            ("style_code", pa.string()),
            ("index", pa.int64()),
            ("style_index", pa.int64()),
            ("garment_id", pa.string()),
            ("batch", pa.string()),
            ("body_kind", pa.string()),
            ("source_member", pa.string()),
            ("source_offset", pa.int64()),
            ("source_size", pa.int64()),
            ("structural_tags", pa.list_(pa.string())),
            ("human_spec", pa.string()),
            ("texture_spec", pa.string()),
            ("material", pa.string()),
            ("palette", pa.string()),
            ("motif", pa.string()),
            ("prompt", pa.string()),
            ("prompt_sha256", pa.string()),
            ("selection_seed", pa.int64()),
        ],
        metadata={
            b"schema_version": PARQUET_SCHEMA_VERSION.encode(),
            b"content": b"GCDv2 remaining-campaign manifest",
        },
    )


def output_schema() -> pa.Schema:
    return pa.schema(
        [
            ("schema_version", pa.string()),
            ("manifest_index", pa.int64()),
            ("style_index", pa.int64()),
            ("garment_id", pa.string()),
            ("batch", pa.string()),
            ("body_kind", pa.string()),
            ("source_member", pa.string()),
            ("source_offset", pa.int64()),
            ("source_size", pa.int64()),
            ("structural_tags", pa.list_(pa.string())),
            ("prompt_version", pa.string()),
            ("style_version", pa.string()),
            ("style_code", pa.string()),
            ("human_spec", pa.string()),
            ("texture_spec", pa.string()),
            ("material", pa.string()),
            ("palette", pa.string()),
            ("motif", pa.string()),
            ("prompt", pa.string()),
            ("prompt_sha256", pa.string()),
            ("request_prompt_sha256", pa.string()),
            ("model", pa.string()),
            ("quality", pa.string()),
            ("requested_size", pa.string()),
            ("status", pa.string()),
            ("image_bytes", pa.binary()),
            ("image_mime_type", pa.string()),
            ("image_width", pa.int32()),
            ("image_height", pa.int32()),
            ("created_at", pa.string()),
            ("attempt_in_process", pa.int32()),
            ("input_tokens", pa.int64()),
            ("output_tokens", pa.int64()),
            ("input_image_tokens", pa.int64()),
            ("input_text_tokens", pa.int64()),
            ("output_image_tokens", pa.int64()),
            ("output_text_tokens", pa.int64()),
            ("usage_json", pa.string()),
            ("error", pa.string()),
            ("slurm_job_id", pa.string()),
            ("slurm_array_job_id", pa.string()),
            ("slurm_array_task_id", pa.string()),
        ],
        metadata={
            b"schema_version": PARQUET_SCHEMA_VERSION.encode(),
            b"image_column": b"image_bytes",
            b"image_encoding": b"PNG",
        },
    )


def atomic_write_parquet(
    path: Path,
    table: pa.Table,
    *,
    row_group_size: Optional[int] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            compression_level=3,
            # Image payloads and prompts are effectively unique. Disabling
            # dictionary encoding avoids building a large in-memory dictionary
            # of 64 multi-megabyte PNG blobs per output part.
            use_dictionary=False,
            row_group_size=row_group_size,
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_excluded_ids(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing exclusion manifest {path}")
    if path.suffix == ".parquet":
        table = pq.read_table(path, columns=["garment_id"])
        values = table.column("garment_id").to_pylist()
        return {str(value) for value in values if value}
    result: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            garment_id = payload.get("garment_id")
            if not isinstance(garment_id, str) or not garment_id:
                raise ValueError(
                    f"Missing garment_id at {path}:{line_number}"
                )
            result.add(garment_id)
    return result


def scan_archive_remaining(
    archive_path: Path,
    captions: dict[str, list[str]],
    excluded_ids: set[str],
    body_kind: str,
    require_complete_outfit: bool = True,
) -> tuple[list[dict[str, Any]], int, int, int]:
    """Return every eligible front render not present in ``excluded_ids``."""

    remaining: list[dict[str, Any]] = []
    excluded_seen: set[str] = set()
    seen_ids: set[str] = set()
    eligible = 0
    scanned = 0
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
                    lowered = {tag.strip().lower() for tag in captions[garment_id]}
                    if require_complete_outfit and lowered.intersection(
                        {"no top", "no bottom"}
                    ):
                        archive.members.clear()
                        continue
                    eligible += 1
                    if garment_id in excluded_ids:
                        excluded_seen.add(garment_id)
                    else:
                        remaining.append(
                            {
                                "garment_id": garment_id,
                                "batch": batch,
                                "body_kind": body_kind,
                                "source_member": member.name,
                                "source_offset": int(member.offset_data),
                                "source_size": int(member.size),
                                "structural_tags": captions[garment_id],
                            }
                        )
            if scanned % 100_000 == 0:
                print(
                    " ".join(
                        [
                            f"tar_members_scanned={scanned}",
                            f"eligible_front_renders={eligible}",
                            f"remaining_front_renders={len(remaining)}",
                        ]
                    ),
                    flush=True,
                )
            archive.members.clear()

    missing_exclusions = excluded_ids - excluded_seen
    if missing_exclusions:
        examples = ", ".join(sorted(missing_exclusions)[:5])
        raise ValueError(
            f"{len(missing_exclusions)} excluded IDs were not eligible archive rows; "
            f"examples: {examples}"
        )
    remaining.sort(key=lambda item: int(item["source_offset"]))
    return remaining, eligible, scanned, len(excluded_seen)


def make_remaining_manifest_records(
    sources: Sequence[dict[str, Any]],
    seed: int,
    style_index_offset: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        style_index = style_index_offset + index
        style = style_for_index(style_index, seed=seed)
        prompt = build_remaining_edit_prompt(
            source["structural_tags"], style["human"], style["texture"]
        )
        records.append(
            {
                "schema_version": PARQUET_SCHEMA_VERSION,
                "prompt_version": PROMPT_VERSION,
                "style_version": STYLE_VERSION,
                "style_code": style["style_code"],
                "index": index,
                "style_index": style_index,
                **source,
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


def build_remaining_edit_prompt(
    structural_tags: Sequence[str], human_spec: str, texture_spec: str
) -> str:
    """Build the normal prompt plus explicit safe clothing for separates."""

    prompt = build_edit_prompt(structural_tags, human_spec, texture_spec)
    lowered = {str(tag).strip().lower() for tag in structural_tags}
    if "no top" in lowered:
        prompt += (
            "\n\nThis record contains a lower-body target garment only. Preserve and "
            "retexture that target garment exactly. Fully dress the adult model above "
            "it in a plain opaque neutral long-sleeve crew-neck top, with no print, "
            "logo, exposed torso, or added fashion detail. The neutral coverage top is "
            "not part of the target garment and must not obscure its waistband."
        )
    elif "no bottom" in lowered:
        prompt += (
            "\n\nThis record contains an upper-body target garment only. Preserve and "
            "retexture that target garment exactly. Fully dress the adult model below "
            "it in plain opaque neutral straight-leg trousers, with no print, logo, "
            "exposed hips, or added fashion detail. The neutral coverage trousers are "
            "not part of the target garment and must not obscure its hem."
        )
    return prompt


def prepare_run(args: argparse.Namespace) -> int:
    manifest_path = args.run_dir / "manifest.parquet"
    config_path = args.run_dir / "config.json"
    if manifest_path.is_file() and config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "already_prepared": True,
                    "manifest": str(manifest_path.resolve()),
                    "selected_count": config.get("selected_count"),
                    "num_shards": config.get("num_shards"),
                },
                indent=2,
            ),
            flush=True,
        )
        return 0
    if config_path.exists() and not manifest_path.is_file():
        raise ValueError(
            f"Configuration exists without {manifest_path}; inspect before retrying"
        )
    if manifest_path.exists() and any((args.run_dir / "parquet").glob("*.parquet")):
        raise ValueError(
            "Manifest preparation is incomplete but output parts already exist; "
            "inspect the run directory before retrying"
        )

    args.run_dir.mkdir(parents=True, exist_ok=True)
    captions = load_captions(args.captions)
    excluded_ids = load_excluded_ids(args.exclude_manifest)
    print(
        f"Scanning {args.archive} for all eligible rows except "
        f"{len(excluded_ids)} excluded IDs",
        flush=True,
    )
    sources, eligible, scanned, excluded_seen = scan_archive_remaining(
        args.archive,
        captions,
        excluded_ids,
        args.body_kind,
        require_complete_outfit=args.require_complete_outfit,
    )
    records = make_remaining_manifest_records(
        sources, args.seed, args.style_index_offset
    )
    table = pa.Table.from_pylist(records, schema=manifest_schema())
    atomic_write_parquet(
        manifest_path,
        table,
        row_group_size=args.rows_per_shard,
    )
    parquet_file = pq.ParquetFile(manifest_path)
    expected_shards = math.ceil(len(records) / args.rows_per_shard)
    if parquet_file.metadata.num_rows != len(records):
        raise RuntimeError("Committed manifest row count does not match preparation")
    if parquet_file.num_row_groups != expected_shards:
        raise RuntimeError("Committed manifest row-group count does not match shards")

    archive_stat = args.archive.stat()
    config = {
        "schema_version": PARQUET_SCHEMA_VERSION,
        "created_at": utc_now(),
        "archive": str(args.archive.resolve()),
        "archive_size": archive_stat.st_size,
        "archive_mtime_ns": archive_stat.st_mtime_ns,
        "captions": str(args.captions.resolve()),
        "caption_count": len(captions),
        "body_kind": args.body_kind,
        "require_complete_outfit": args.require_complete_outfit,
        "selection_seed": args.seed,
        "style_index_offset": args.style_index_offset,
        "selected_count": len(records),
        "eligible_count": eligible,
        "excluded_count": excluded_seen,
        "exclude_manifest": str(args.exclude_manifest.resolve()),
        "exclude_manifest_sha256": stable_digest(
            args.exclude_manifest.read_text(encoding="utf-8")
        ),
        "tar_members_scanned": scanned,
        "rows_per_shard": args.rows_per_shard,
        "num_shards": expected_shards,
        "planned_worker_count": DEFAULT_WORKER_COUNT,
        "image_model": DEFAULT_IMAGE_MODEL,
        "quality": DEFAULT_QUALITY,
        "size": DEFAULT_SIZE,
        "prompt_version": PROMPT_VERSION,
        "style_version": STYLE_VERSION,
        "input_token_rate": INPUT_TOKEN_RATE,
        "output_token_rate": OUTPUT_TOKEN_RATE,
        "output_format": "parquet",
        "image_storage": "PNG bytes in image_bytes",
    }
    atomic_write_text(
        config_path, json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "selected": len(records),
                "eligible": eligible,
                "excluded": excluded_seen,
                "tar_members_scanned": scanned,
                "rows_per_shard": args.rows_per_shard,
                "num_shards": expected_shards,
                "manifest": str(manifest_path.resolve()),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


class StagingStore:
    """Durable per-worker staging for paid results awaiting Parquet commit."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path, timeout=60.0)
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                manifest_index INTEGER PRIMARY KEY,
                row_json TEXT NOT NULL,
                image_bytes BLOB,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "StagingStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def put(self, row: dict[str, Any]) -> None:
        payload = dict(row)
        image = payload.pop("image_bytes", None)
        self.connection.execute(
            """
            INSERT INTO results(manifest_index, row_json, image_bytes, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(manifest_index) DO UPDATE SET
                row_json=excluded.row_json,
                image_bytes=excluded.image_bytes,
                updated_at=excluded.updated_at
            """,
            (
                int(row["manifest_index"]),
                json.dumps(payload, sort_keys=True),
                image,
                utc_now(),
            ),
        )
        self.connection.commit()

    def load(self, indices: Iterable[int]) -> dict[int, dict[str, Any]]:
        values = [int(index) for index in indices]
        if not values:
            return {}
        placeholders = ",".join("?" for _ in values)
        rows = self.connection.execute(
            f"SELECT manifest_index, row_json, image_bytes FROM results "
            f"WHERE manifest_index IN ({placeholders})",
            values,
        )
        result: dict[int, dict[str, Any]] = {}
        for index, row_json, image in rows:
            row = json.loads(row_json)
            row["image_bytes"] = bytes(image) if image is not None else None
            result[int(index)] = row
        return result

    def delete(self, indices: Iterable[int]) -> None:
        values = [int(index) for index in indices]
        if not values:
            return
        placeholders = ",".join("?" for _ in values)
        self.connection.execute(
            f"DELETE FROM results WHERE manifest_index IN ({placeholders})", values
        )
        self.connection.commit()

    def iter_rows(self) -> Iterable[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT row_json, image_bytes FROM results ORDER BY manifest_index"
        )
        for row_json, image in rows:
            row = json.loads(row_json)
            row["image_bytes"] = bytes(image) if image is not None else None
            yield row

    def vacuum_if_empty(self) -> None:
        count = self.connection.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        if int(count) == 0:
            self.connection.execute("VACUUM")


def read_source_bytes(archive: Any, record: dict[str, Any]) -> bytes:
    expected = int(record["source_size"])
    archive.seek(int(record["source_offset"]))
    value = archive.read(expected)
    if len(value) != expected or not value.startswith(PNG_SIGNATURE):
        raise ValueError(
            f"Could not read complete source PNG for {record['garment_id']} "
            f"at offset {record['source_offset']}"
        )
    try:
        with Image.open(BytesIO(value)) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(
            f"Source PNG for {record['garment_id']} is invalid: {exc}"
        ) from exc
    return value


def safety_retry_prompt(prompt: str) -> str:
    return (
        prompt
        + "\n\nSafety retry requirements: This is a non-sensual commercial catalog "
        "photograph of one clearly adult model in an upright neutral pose. Keep the "
        "person fully and opaquely clothed. Where source coverage is ambiguous, add "
        "a plain color-matched opaque base layer underneath without changing the "
        "visible garment geometry. No nudity, lingerie, swimwear, sheer exposure, "
        "provocative pose, or emphasis on intimate anatomy."
    )


def usage_fields(payload: dict[str, Any]) -> dict[str, Any]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    input_details = usage.get("input_tokens_details")
    output_details = usage.get("output_tokens_details")
    if not isinstance(input_details, dict):
        input_details = {}
    if not isinstance(output_details, dict):
        output_details = {}

    def as_int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    return {
        "input_tokens": as_int(usage.get("input_tokens")),
        "output_tokens": as_int(usage.get("output_tokens")),
        "input_image_tokens": as_int(input_details.get("image_tokens")),
        "input_text_tokens": as_int(input_details.get("text_tokens")),
        "output_image_tokens": as_int(output_details.get("image_tokens")),
        "output_text_tokens": as_int(output_details.get("text_tokens")),
        "usage_json": json.dumps(usage, sort_keys=True),
    }


def base_output_row(
    record: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    return {
        "schema_version": PARQUET_SCHEMA_VERSION,
        "manifest_index": int(record["index"]),
        "style_index": int(record["style_index"]),
        "garment_id": str(record["garment_id"]),
        "batch": str(record["batch"]),
        "body_kind": str(record["body_kind"]),
        "source_member": str(record["source_member"]),
        "source_offset": int(record["source_offset"]),
        "source_size": int(record["source_size"]),
        "structural_tags": [str(value) for value in record["structural_tags"]],
        "prompt_version": str(record["prompt_version"]),
        "style_version": str(record["style_version"]),
        "style_code": str(record["style_code"]),
        "human_spec": str(record["human_spec"]),
        "texture_spec": str(record["texture_spec"]),
        "material": str(record["material"]),
        "palette": str(record["palette"]),
        "motif": str(record["motif"]),
        "prompt": str(record["prompt"]),
        "prompt_sha256": str(record["prompt_sha256"]),
        "request_prompt_sha256": str(record["prompt_sha256"]),
        "model": args.image_model,
        "quality": args.quality,
        "requested_size": args.size,
        "status": "failed",
        "image_bytes": None,
        "image_mime_type": None,
        "image_width": None,
        "image_height": None,
        "created_at": utc_now(),
        "attempt_in_process": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "input_image_tokens": 0,
        "input_text_tokens": 0,
        "output_image_tokens": 0,
        "output_text_tokens": 0,
        "usage_json": "{}",
        "error": None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def generate_one(
    client: GatewayClient,
    archive: Any,
    record: dict[str, Any],
    args: argparse.Namespace,
    event_path: Path,
) -> dict[str, Any]:
    source = read_source_bytes(archive, record)
    expected_size = expected_dimensions(args.size)
    last_error: Optional[Exception] = None
    for attempt in range(1, args.item_attempts + 1):
        prompt = str(record["prompt"])
        if attempt > 1 and args.safety_retry:
            prompt = safety_retry_prompt(prompt)
        started = time.monotonic()
        try:
            payload = client.edit_image_bytes(
                source,
                prompt,
                filename=f"{record['garment_id']}.png",
                media_type="image/png",
                n=1,
                size=args.size,
                quality=args.quality,
            )
            items = image_items(payload)
            if len(items) != 1:
                raise GatewayError(
                    f"Expected one image result, received {len(items)}"
                )
            png = normalized_png_bytes(client.image_bytes(items[0]), expected_size)
        except (GatewayError, OSError, ValueError) as exc:
            last_error = exc
            append_event(
                event_path,
                {
                    "time": utc_now(),
                    "event": "attempt_failed",
                    "index": record["index"],
                    "garment_id": record["garment_id"],
                    "attempt": attempt,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "error": str(exc)[:4000],
                },
            )
            if attempt < args.item_attempts:
                time.sleep(min(120.0, 15.0 * (2 ** (attempt - 1))))
            continue

        row = base_output_row(record, args)
        row.update(
            {
                "request_prompt_sha256": stable_digest(prompt),
                "status": "completed",
                "image_bytes": png,
                "image_mime_type": "image/png",
                "image_width": expected_size[0],
                "image_height": expected_size[1],
                "created_at": utc_now(),
                "attempt_in_process": attempt,
                "error": None,
                **usage_fields(payload),
            }
        )
        append_event(
            event_path,
            {
                "time": row["created_at"],
                "event": "completed",
                "index": record["index"],
                "garment_id": record["garment_id"],
                "attempt": attempt,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "usage": json.loads(row["usage_json"]),
            },
        )
        return row

    assert last_error is not None
    row = base_output_row(record, args)
    row["attempt_in_process"] = args.item_attempts
    row["error"] = str(last_error)[:4000]
    append_event(
        event_path,
        {
            "time": row["created_at"],
            "event": "item_failed",
            "index": record["index"],
            "garment_id": record["garment_id"],
            "error": row["error"],
        },
    )
    return row


def load_output_rows(
    path: Path, expected_records: Sequence[dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    expected = {int(record["index"]): str(record["garment_id"]) for record in expected_records}
    table = pq.read_table(path)
    required = {field.name for field in output_schema()}
    missing_columns = required - set(table.column_names)
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")
    rows: dict[int, dict[str, Any]] = {}
    for row in table.to_pylist():
        index = int(row["manifest_index"])
        if index not in expected:
            raise ValueError(f"{path} contains unexpected manifest index {index}")
        if index in rows:
            raise ValueError(f"{path} contains duplicate manifest index {index}")
        if str(row["garment_id"]) != expected[index]:
            raise ValueError(f"{path} garment ID does not match manifest index {index}")
        if row["status"] not in {"completed", "failed"}:
            raise ValueError(f"{path} has invalid status at manifest index {index}")
        if row["status"] == "completed":
            image = row.get("image_bytes")
            if not isinstance(image, bytes) or not image.startswith(PNG_SIGNATURE):
                raise ValueError(f"{path} has invalid image bytes at index {index}")
            row["image_bytes"] = image
        rows[index] = row
    return rows


def quarantine_invalid_parquet(run_dir: Path, path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = run_dir / "quarantine" / f"{path.stem}.{stamp}.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(path, destination)
    return destination


def write_output_rows(path: Path, rows: dict[int, dict[str, Any]]) -> None:
    ordered = [rows[index] for index in sorted(rows)]
    table = pa.Table.from_pylist(ordered, schema=output_schema())
    atomic_write_parquet(path, table, row_group_size=len(ordered) or None)
    committed = pq.ParquetFile(path)
    if committed.metadata.num_rows != len(ordered):
        raise RuntimeError(f"Committed row count mismatch in {path}")


def read_manifest_shard(
    parquet_file: pq.ParquetFile, shard_id: int
) -> list[dict[str, Any]]:
    if shard_id < 0 or shard_id >= parquet_file.num_row_groups:
        raise ValueError(f"Manifest shard {shard_id} is out of range")
    rows = parquet_file.read_row_group(shard_id).to_pylist()
    expected_start = sum(
        parquet_file.metadata.row_group(index).num_rows for index in range(shard_id)
    )
    for offset, row in enumerate(rows):
        if row.get("schema_version") != PARQUET_SCHEMA_VERSION:
            raise ValueError(f"Unsupported manifest schema in shard {shard_id}")
        if row.get("prompt_version") != PROMPT_VERSION:
            raise ValueError(f"Stale prompt version in shard {shard_id}")
        if row.get("style_version") != STYLE_VERSION:
            raise ValueError(f"Stale style version in shard {shard_id}")
        if int(row.get("index", -1)) != expected_start + offset:
            raise ValueError(f"Non-contiguous manifest index in shard {shard_id}")
        if stable_digest(str(row.get("prompt", ""))) != row.get("prompt_sha256"):
            raise ValueError(f"Prompt hash mismatch in shard {shard_id}")
    return rows


def shard_ids_for_worker(
    num_shards: int, worker_index: int, worker_count: int
) -> list[int]:
    if worker_count < 1:
        raise ValueError("worker-count must be positive")
    if worker_index < 0 or worker_index >= worker_count:
        raise ValueError("worker-index must be in [0, worker-count)")
    return list(range(worker_index, num_shards, worker_count))


def load_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run prepare first")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != PARQUET_SCHEMA_VERSION:
        raise ValueError(f"Unsupported run schema in {path}")
    return config


def nearing_slurm_deadline(stop_before_seconds: int) -> bool:
    if stop_before_seconds <= 0:
        return False
    raw = os.environ.get("SLURM_JOB_END_TIME")
    if not raw:
        return False
    try:
        return time.time() >= int(raw) - stop_before_seconds
    except ValueError:
        return False


def run_worker(args: argparse.Namespace) -> int:
    config = load_config(args.run_dir)
    manifest_path = args.run_dir / "manifest.parquet"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing {manifest_path}; run prepare first")
    archive_path = Path(config["archive"])
    archive_stat = archive_path.stat()
    if archive_stat.st_size != int(config["archive_size"]):
        raise ValueError("Source archive size changed after manifest preparation")
    manifest = pq.ParquetFile(manifest_path)
    assigned = shard_ids_for_worker(
        manifest.num_row_groups, args.worker_index, args.worker_count
    )
    event_path = (
        args.run_dir / "logs" / f"worker_{args.worker_index:03d}.events.jsonl"
    )
    staging_path = (
        args.run_dir / "staging" / f"worker_{args.worker_index:03d}.sqlite3"
    )
    client = GatewayClient(
        key_file=args.key_file,
        image_base_url=args.image_base_url,
        image_model=args.image_model,
        credential_index_from_end=args.credential_index_from_end,
        timeout=args.timeout,
        max_attempts=args.request_attempts,
    )
    counters = {
        "completed": 0,
        "failed": 0,
        "already_completed": 0,
        "already_failed": 0,
        "shards_written": 0,
        "shards_complete": 0,
    }
    remaining_budget = args.max_items if args.max_items > 0 else None
    needs_requeue = False
    worker_started = time.monotonic()
    print(
        " ".join(
            [
                f"worker={args.worker_index}/{args.worker_count}",
                f"assigned_shards={len(assigned)}",
                f"manifest_rows={manifest.metadata.num_rows}",
                f"model={args.image_model}",
                f"quality={args.quality}",
            ]
        ),
        flush=True,
    )

    with StagingStore(staging_path) as staging, archive_path.open("rb") as archive:
        for shard_id in assigned:
            records = read_manifest_shard(manifest, shard_id)
            expected_indices = [int(record["index"]) for record in records]
            output_path = args.run_dir / "parquet" / f"part-{shard_id:06d}.parquet"
            try:
                rows = load_output_rows(output_path, records)
            except (OSError, ValueError, pa.ArrowException) as exc:
                quarantined = quarantine_invalid_parquet(args.run_dir, output_path)
                append_event(
                    event_path,
                    {
                        "time": utc_now(),
                        "event": "quarantined_invalid_parquet",
                        "shard_id": shard_id,
                        "path": str(quarantined.relative_to(args.run_dir)),
                        "error": str(exc)[:4000],
                    },
                )
                rows = {}
            rows.update(staging.load(expected_indices))

            pending: list[dict[str, Any]] = []
            for record in records:
                index = int(record["index"])
                existing = rows.get(index)
                if existing is None:
                    pending.append(record)
                elif existing["status"] == "completed":
                    counters["already_completed"] += 1
                elif args.retry_failed:
                    pending.append(record)
                else:
                    counters["already_failed"] += 1

            changed = False
            for position, record in enumerate(pending, 1):
                if remaining_budget is not None and remaining_budget <= 0:
                    break
                if remaining_budget is None and nearing_slurm_deadline(
                    args.stop_before_job_end_seconds
                ):
                    needs_requeue = True
                    print(
                        f"worker={args.worker_index} approaching Slurm deadline; "
                        "committing the partial shard before requeue",
                        flush=True,
                    )
                    break
                if (
                    remaining_budget is None
                    and args.max_runtime_seconds > 0
                    and time.monotonic() - worker_started >= args.max_runtime_seconds
                ):
                    needs_requeue = True
                    print(
                        f"worker={args.worker_index} reached its safe runtime; "
                        "committing the partial shard before requeue",
                        flush=True,
                    )
                    break
                row = generate_one(client, archive, record, args, event_path)
                staging.put(row)
                rows[int(record["index"])] = row
                changed = True
                counters[str(row["status"])] += 1
                if remaining_budget is not None:
                    remaining_budget -= 1
                print(
                    " ".join(
                        [
                            f"worker={args.worker_index}",
                            f"shard={shard_id}",
                            f"progress={position}/{len(pending)}",
                            f"index={record['index']}",
                            f"id={record['garment_id']}",
                            f"status={row['status']}",
                        ]
                    ),
                    flush=True,
                )

            if changed or (rows and not output_path.exists()):
                write_output_rows(output_path, rows)
                staging.delete(rows)
                counters["shards_written"] += 1
            if len(rows) == len(records):
                counters["shards_complete"] += 1

            if remaining_budget is not None and remaining_budget <= 0:
                break
            if needs_requeue:
                break
        staging.vacuum_if_empty()

    print(
        json.dumps(counters | {"needs_requeue": needs_requeue}, sort_keys=True),
        flush=True,
    )
    return 75 if needs_requeue else 0


def rows_from_staging(path: Path) -> Iterable[dict[str, Any]]:
    try:
        connection = sqlite3.connect(
            f"file:{path.resolve()}?mode=ro", uri=True, timeout=5.0
        )
        try:
            rows = connection.execute(
                "SELECT row_json, image_bytes FROM results ORDER BY manifest_index"
            )
            for row_json, image in rows:
                row = json.loads(row_json)
                row["image_bytes"] = bytes(image) if image is not None else None
                yield row
        finally:
            connection.close()
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"Could not read staging database {path}: {exc}") from exc


def campaign_status(args: argparse.Namespace) -> int:
    config = load_config(args.run_dir)
    total = int(config["selected_count"])
    rows: dict[int, dict[str, Any]] = {}
    invalid_parquet: list[str] = []
    parquet_paths = sorted((args.run_dir / "parquet").glob("part-*.parquet"))
    columns = [
        "manifest_index",
        "status",
        "input_tokens",
        "output_tokens",
    ]
    for path in parquet_paths:
        try:
            for row in pq.read_table(path, columns=columns).to_pylist():
                rows[int(row["manifest_index"])] = row
        except (OSError, ValueError, pa.ArrowException):
            invalid_parquet.append(str(path.relative_to(args.run_dir)))
    for path in sorted((args.run_dir / "staging").glob("worker_*.sqlite3")):
        for row in rows_from_staging(path):
            rows[int(row["manifest_index"])] = row

    completed = sum(row.get("status") == "completed" for row in rows.values())
    failed = sum(row.get("status") == "failed" for row in rows.values())
    input_tokens = sum(int(row.get("input_tokens") or 0) for row in rows.values())
    output_tokens = sum(int(row.get("output_tokens") or 0) for row in rows.values())
    cost = input_tokens * args.input_token_rate + output_tokens * args.output_token_rate
    result = {
        "schema_version": PARQUET_SCHEMA_VERSION,
        "checked_at": utc_now(),
        "run_dir": str(args.run_dir.resolve()),
        "total": total,
        "processed": completed + failed,
        "completed": completed,
        "failed": failed,
        "missing": total - len(rows),
        "progress_percent": round(100.0 * len(rows) / total, 3) if total else 0.0,
        "valid_image_percent": round(100.0 * completed / total, 3) if total else 0.0,
        "parquet_files": len(parquet_paths),
        "expected_parquet_files": int(config["num_shards"]),
        "invalid_parquet_files": invalid_parquet,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "exact_recorded_cost_usd": round(cost, 6),
        "input_token_rate": args.input_token_rate,
        "output_token_rate": args.output_token_rate,
    }
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(serialized, end="")
    if args.json_output is not None:
        atomic_write_text(args.json_output, serialized)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare", help="Build a Parquet manifest for all remaining eligible garments"
    )
    prepare.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    prepare.add_argument("--captions", type=Path, default=DEFAULT_CAPTIONS)
    prepare.add_argument("--exclude-manifest", type=Path, default=DEFAULT_EXCLUDE_MANIFEST)
    prepare.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare.add_argument("--style-index-offset", type=int, default=DEFAULT_STYLE_INDEX_OFFSET)
    prepare.add_argument("--rows-per-shard", type=int, default=DEFAULT_ROWS_PER_SHARD)
    prepare.add_argument("--body-kind", default="default_body")
    prepare.add_argument(
        "--allow-separates",
        action="store_false",
        dest="require_complete_outfit",
        help="Include records tagged no-top or no-bottom (not recommended for human edits)",
    )
    prepare.set_defaults(require_complete_outfit=True)

    run = commands.add_parser("run", help="Run one resumable Parquet worker")
    run.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    run.add_argument(
        "--worker-index",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
    )
    run.add_argument(
        "--worker-count",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1")),
    )
    run.add_argument("--max-items", type=int, default=0)
    run.add_argument("--retry-failed", action="store_true")
    run.add_argument("--stop-before-job-end-seconds", type=int, default=900)
    run.add_argument("--max-runtime-seconds", type=int, default=0)
    run.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    run.add_argument(
        "--credential-index-from-end",
        type=int,
        default=DEFAULT_CREDENTIAL_INDEX_FROM_END,
    )
    run.add_argument("--image-base-url", default=DEFAULT_IMAGE_BASE_URL)
    run.add_argument("--image-model", default=DEFAULT_IMAGE_MODEL)
    run.add_argument(
        "--quality",
        choices=("low", "medium", "high", "auto"),
        default=DEFAULT_QUALITY,
    )
    run.add_argument("--size", default=DEFAULT_SIZE)
    run.add_argument("--timeout", type=float, default=600.0)
    run.add_argument("--request-attempts", type=int, default=4)
    run.add_argument("--item-attempts", type=int, default=2)
    run.add_argument(
        "--no-safety-retry",
        action="store_false",
        dest="safety_retry",
    )
    run.set_defaults(safety_retry=True)

    status = commands.add_parser("status", help="Summarize committed and staged rows")
    status.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    status.add_argument("--input-token-rate", type=float, default=INPUT_TOKEN_RATE)
    status.add_argument("--output-token-rate", type=float, default=OUTPUT_TOKEN_RATE)
    status.add_argument("--json-output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            if args.rows_per_shard < 1:
                raise ValueError("rows-per-shard must be positive")
            if args.style_index_offset < 0:
                raise ValueError("style-index-offset must be non-negative")
            return prepare_run(args)
        if args.command == "run":
            if args.item_attempts < 1 or args.request_attempts < 1:
                raise ValueError("attempt counts must be positive")
            return run_worker(args)
        if args.command == "status":
            return campaign_status(args)
        raise AssertionError(f"Unhandled command {args.command}")
    except (
        FileNotFoundError,
        GatewayError,
        OSError,
        ValueError,
        RuntimeError,
        sqlite3.DatabaseError,
        pa.ArrowException,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
