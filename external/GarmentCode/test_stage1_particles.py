import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from pygarment.meshgen.stage1_particles import write_stage1_particles


class Stage1ParticleWriterTest(unittest.TestCase):
    def setUp(self):
        self.particles = {
            "front": {
                "front_panel": {
                    "boundary_verts": np.arange(20).reshape(4, 5),
                    "interior_verts": np.arange(10).reshape(2, 5),
                },
            },
            "back": {
                "back_panel": {
                    "boundary_verts": np.arange(15).reshape(3, 5),
                    "interior_verts": np.empty((0, 5)),
                },
            },
        }
        self.metadata = {
            "strategy": "fine_to_coarse",
            "padding_cm": 3.0,
            "max_iterations": 500,
            "sides": {
                "front": {"iterations": 4, "has_overlap": False},
                "back": {"iterations": 2, "has_overlap": False},
            },
        }
        self.offsets = {
            "front_panel": [1.25, -2.0],
            "back_panel": [0.0, 3.5],
        }

    def test_writes_stage_one_loader_schema_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "garment_particles_example.h5"
            write_stage1_particles(
                output,
                self.particles,
                self.metadata,
                self.offsets,
            )

            with h5py.File(output, "r") as dataset:
                self.assertEqual(
                    dataset.attrs["format"],
                    "garmentparticles_stage1",
                )
                self.assertEqual(dataset.attrs["format_version"], 1)
                self.assertEqual(
                    dataset.attrs["packing_strategy"],
                    "fine_to_coarse",
                )
                np.testing.assert_array_equal(
                    dataset["front"]["front_panel"]["boundary_verts"],
                    self.particles["front"]["front_panel"][
                        "boundary_verts"
                    ],
                )
                np.testing.assert_allclose(
                    dataset["back"]["back_panel"].attrs[
                        "packing_offset"
                    ],
                    self.offsets["back_panel"],
                )

    def test_rejects_wrong_particle_width_without_publishing_file(self):
        self.particles["front"]["front_panel"]["boundary_verts"] = (
            np.zeros((4, 4))
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "invalid.h5"
            with self.assertRaisesRegex(ValueError, "shape \\(N, 5\\)"):
                write_stage1_particles(
                    output,
                    self.particles,
                    self.metadata,
                    self.offsets,
                )
            self.assertFalse(output.exists())
            self.assertFalse((Path(directory) / ".invalid.h5.tmp").exists())


if __name__ == "__main__":
    unittest.main()
