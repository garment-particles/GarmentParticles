from __future__ import annotations

import argparse
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tools.gcdv2_realistic_parquet import (  # noqa: E402
    PARQUET_SCHEMA_VERSION,
    StagingStore,
    atomic_write_parquet,
    base_output_row,
    build_remaining_edit_prompt,
    generate_one,
    load_output_rows,
    make_remaining_manifest_records,
    manifest_schema,
    read_manifest_shard,
    read_source_bytes,
    scan_archive_remaining,
    shard_ids_for_worker,
    write_output_rows,
)


def png_bytes(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(output, format="PNG")
    return output.getvalue()


class GCDv2RealisticParquetTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def make_archive(self) -> tuple[Path, dict[str, list[str]]]:
        archive_path = self.tmp_path / "tiny.tar"
        tags = (
            ["fitted top", "straight skirt"],
            ["no top", "straight skirt"],
            ["fitted top", "no bottom"],
            ["fitted top", "flare skirt"],
            ["fitted top", "trousers"],
        )
        captions: dict[str, list[str]] = {}
        with tarfile.open(archive_path, "w") as archive:
            for index, structural_tags in enumerate(tags):
                garment_id = f"rand_PQ{index:05d}"
                captions[garment_id] = structural_tags
                value = png_bytes((index * 20, 30, 40))
                member = tarfile.TarInfo(
                    "garmentcodedatav2/garments_5000_0/default_body/"
                    f"{garment_id}/{garment_id}_render_front.png"
                )
                member.size = len(value)
                archive.addfile(member, io.BytesIO(value))
        return archive_path, captions

    def test_scan_all_remaining_includes_separates(self):
        archive, captions = self.make_archive()
        excluded = {"rand_PQ00000"}

        remaining, eligible, _, excluded_seen = scan_archive_remaining(
            archive,
            captions,
            excluded,
            "default_body",
            require_complete_outfit=False,
        )

        self.assertEqual(eligible, 5)
        self.assertEqual(excluded_seen, 1)
        self.assertEqual(len(remaining), 4)
        self.assertEqual(
            [record["source_offset"] for record in remaining],
            sorted(record["source_offset"] for record in remaining),
        )

    def test_scan_complete_outfits_retains_old_filter(self):
        archive, captions = self.make_archive()
        remaining, eligible, _, excluded_seen = scan_archive_remaining(
            archive,
            captions,
            {"rand_PQ00000"},
            "default_body",
            require_complete_outfit=True,
        )
        self.assertEqual((eligible, excluded_seen, len(remaining)), (3, 1, 2))

    def test_separate_prompts_add_opaque_base_clothing(self):
        lower = build_remaining_edit_prompt(
            ["no top", "straight skirt"], "an adult", "navy wool"
        )
        upper = build_remaining_edit_prompt(
            ["fitted top", "no bottom"], "an adult", "navy wool"
        )
        self.assertIn("plain opaque neutral long-sleeve", lower)
        self.assertIn("must not obscure its waistband", lower)
        self.assertIn("plain opaque neutral straight-leg trousers", upper)
        self.assertIn("must not obscure its hem", upper)

    def test_manifest_parquet_has_one_row_group_per_shard(self):
        archive, captions = self.make_archive()
        sources, _, _, _ = scan_archive_remaining(
            archive, captions, set(), "default_body", require_complete_outfit=False
        )
        records = make_remaining_manifest_records(sources, seed=123, style_index_offset=10)
        path = self.tmp_path / "manifest.parquet"
        atomic_write_parquet(
            path,
            pa.Table.from_pylist(records, schema=manifest_schema()),
            row_group_size=2,
        )

        parquet_file = pq.ParquetFile(path)
        self.assertEqual(parquet_file.num_row_groups, 3)
        self.assertEqual(len(read_manifest_shard(parquet_file, 0)), 2)
        self.assertEqual(records[0]["style_index"], 10)
        self.assertEqual(records[-1]["style_index"], 14)

    def test_source_bytes_are_read_directly_from_tar_offset(self):
        archive_path, captions = self.make_archive()
        sources, _, _, _ = scan_archive_remaining(
            archive_path,
            captions,
            set(),
            "default_body",
            require_complete_outfit=False,
        )
        with archive_path.open("rb") as archive:
            value = read_source_bytes(archive, sources[2])
        with Image.open(io.BytesIO(value)) as image:
            self.assertEqual(image.size, (64, 64))

    def test_sqlite_staging_survives_reopen(self):
        path = self.tmp_path / "staging.sqlite3"
        row = {
            "manifest_index": 7,
            "status": "completed",
            "image_bytes": png_bytes((1, 2, 3)),
        }
        with StagingStore(path) as store:
            store.put(row)
        with StagingStore(path) as store:
            restored = store.load([7])[7]
            self.assertEqual(restored["status"], "completed")
            self.assertEqual(restored["image_bytes"], row["image_bytes"])
            store.delete([7])
            self.assertEqual(store.load([7]), {})

    def test_generate_one_logs_and_returns_a_complete_row(self):
        archive_path, captions = self.make_archive()
        sources, _, _, _ = scan_archive_remaining(
            archive_path,
            captions,
            set(),
            "default_body",
            require_complete_outfit=False,
        )
        record = make_remaining_manifest_records(sources[:1], 123, 10)[0]
        generated = png_bytes((9, 8, 7))

        class FakeClient:
            def edit_image_bytes(self, *_args, **_kwargs):
                return {
                    "data": [{"fake": True}],
                    "usage": {"input_tokens": 100, "output_tokens": 200},
                }

            def image_bytes(self, _item):
                return generated

        args = argparse.Namespace(
            image_model="openai/openai/gpt-image-2",
            quality="medium",
            size="64x64",
            item_attempts=2,
            safety_retry=True,
        )
        event_path = self.tmp_path / "worker.events.jsonl"
        with archive_path.open("rb") as archive:
            row = generate_one(FakeClient(), archive, record, args, event_path)

        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["input_tokens"], 100)
        self.assertEqual(row["output_tokens"], 200)
        self.assertEqual(row["image_bytes"], generated)
        self.assertIn('"event": "completed"', event_path.read_text())

    def test_output_parquet_round_trip_contains_image_bytes(self):
        archive, captions = self.make_archive()
        sources, _, _, _ = scan_archive_remaining(
            archive, captions, set(), "default_body", require_complete_outfit=False
        )
        records = make_remaining_manifest_records(sources[:2], 123, 10)
        args = argparse.Namespace(
            image_model="openai/openai/gpt-image-2",
            quality="medium",
            size="1024x1024",
        )
        completed = base_output_row(records[0], args)
        completed.update(
            {
                "status": "completed",
                "image_bytes": png_bytes((4, 5, 6)),
                "image_mime_type": "image/png",
                "image_width": 64,
                "image_height": 64,
            }
        )
        failed = base_output_row(records[1], args)
        failed["error"] = "moderation blocked"
        path = self.tmp_path / "part-000000.parquet"

        write_output_rows(path, {0: completed, 1: failed})
        restored = load_output_rows(path, records)

        self.assertEqual(len(restored), 2)
        self.assertEqual(restored[0]["image_bytes"], completed["image_bytes"])
        self.assertEqual(restored[1]["status"], "failed")
        self.assertEqual(
            pq.ParquetFile(path).schema_arrow.metadata[b"schema_version"].decode(),
            PARQUET_SCHEMA_VERSION,
        )

    def test_worker_shards_are_disjoint_and_complete(self):
        assignments = [set(shard_ids_for_worker(17, worker, 4)) for worker in range(4)]
        self.assertEqual(set.union(*assignments), set(range(17)))
        for left in range(4):
            for right in range(left + 1, 4):
                self.assertTrue(assignments[left].isdisjoint(assignments[right]))


if __name__ == "__main__":
    unittest.main()
