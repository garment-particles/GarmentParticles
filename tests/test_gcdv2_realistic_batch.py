from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tools.gcdv2_realistic_batch import (  # noqa: E402
    HUMAN_SPECS,
    build_edit_prompt,
    extract_selected_sources,
    make_manifest_records,
    scan_archive_sample,
    source_member_info,
    style_for_index,
    validate_image,
)


def png_bytes(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(output, format="PNG")
    return output.getvalue()


class GCDv2RealisticBatchTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def make_archive(self, count: int = 6) -> tuple[Path, dict[str, list[str]]]:
        archive_path = self.tmp_path / "tiny.tar"
        captions: dict[str, list[str]] = {}
        with tarfile.open(archive_path, "w") as archive:
            for index in range(count):
                garment_id = f"rand_TEST{index:05d}"
                captions[garment_id] = ["fitted top", "straight skirt"]
                value = png_bytes((index * 20, 30, 40))
                member = tarfile.TarInfo(
                    "garmentcodedatav2/garments_5000_0/default_body/"
                    f"{garment_id}/{garment_id}_render_front.png"
                )
                member.size = len(value)
                archive.addfile(member, io.BytesIO(value))
        return archive_path, captions

    def test_source_member_info_accepts_only_front_render(self):
        path = (
            "garmentcodedatav2/garments_5000_6/default_body/rand_ABC123/"
            "rand_ABC123_render_front.png"
        )
        self.assertEqual(source_member_info(path), ("garments_5000_6", "rand_ABC123"))
        self.assertIsNone(source_member_info(path.replace("front", "back")))
        self.assertIsNone(source_member_info(path.replace("default_body", "random_body")))

    def test_prompt_contains_diversity_and_geometry_constraints(self):
        prompt = build_edit_prompt(
            ["fitted waistband", "tiered skirt"],
            "an older adult with deep skin and short gray curls",
            "emerald velvet with botanical embroidery",
        )
        self.assertIn("older adult with deep skin", prompt)
        self.assertIn("emerald velvet", prompt)
        self.assertIn("fitted waistband, tiered skirt", prompt)
        self.assertIn("Preserve exactly the source garment geometry", prompt)
        self.assertIn("Do not add, remove, shorten, lengthen, or redesign", prompt)
        self.assertIn("If the source crop omits the head, legs, hands, or feet", prompt)

    def test_style_assignment_is_deterministic_and_cycles_humans(self):
        self.assertEqual(style_for_index(7), style_for_index(7))
        humans = {style_for_index(index)["human"] for index in range(len(HUMAN_SPECS))}
        self.assertEqual(len(humans), len(HUMAN_SPECS))

    def test_first_10k_style_packages_are_unique_and_balanced(self):
        styles = [style_for_index(index) for index in range(10_000)]
        self.assertEqual(len({style["style_code"] for style in styles}), 10_000)
        for key in ("human", "material", "palette", "motif"):
            counts = {}
            for style in styles:
                counts[style[key]] = counts.get(style[key], 0) + 1
            expected = len(styles) / len(counts)
            largest_relative_error = max(
                abs(count - expected) / expected for count in counts.values()
            )
            self.assertLess(largest_relative_error, 0.02)

    def test_scan_sample_and_offset_extraction(self):
        archive, captions = self.make_archive()
        selected, eligible, scanned = scan_archive_sample(
            archive, captions, count=4, seed=123, body_kind="default_body"
        )
        self.assertEqual(eligible, 6)
        self.assertGreaterEqual(scanned, 6)
        self.assertEqual(len(selected), 4)
        self.assertEqual(
            [item["garment_id"] for item in selected],
            [
                item["garment_id"]
                for item in scan_archive_sample(
                    archive, captions, count=4, seed=123, body_kind="default_body"
                )[0]
            ],
        )

        records = make_manifest_records(selected, seed=123)
        extracted, skipped = extract_selected_sources(archive, self.tmp_path / "run", records)
        self.assertEqual((extracted, skipped), (4, 0))
        for record in records:
            path = self.tmp_path / "run" / record["source_path"]
            self.assertEqual(validate_image(path), (True, "ok"))

        extracted, skipped = extract_selected_sources(archive, self.tmp_path / "run", records)
        self.assertEqual((extracted, skipped), (0, 4))

    def test_scan_excludes_incomplete_outfits_by_default(self):
        archive, captions = self.make_archive()
        first_id = next(iter(captions))
        captions[first_id] = ["no top", "pants"]

        _, eligible, _ = scan_archive_sample(
            archive, captions, count=5, seed=123, body_kind="default_body"
        )
        self.assertEqual(eligible, 5)

        _, eligible_with_separates, _ = scan_archive_sample(
            archive,
            captions,
            count=6,
            seed=123,
            body_kind="default_body",
            require_complete_outfit=False,
        )
        self.assertEqual(eligible_with_separates, 6)


if __name__ == "__main__":
    unittest.main()
