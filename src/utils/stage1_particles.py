"""Serialization helpers for first-stage GarmentParticles training data."""

import os
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np


SIDES = ("front", "back")
PARTICLE_DATASETS = ("boundary_verts", "interior_verts")


def _validated_particles(panel_name, particles):
    validated = {}
    for dataset_name in PARTICLE_DATASETS:
        if dataset_name not in particles:
            raise KeyError(
                f"Panel {panel_name!r} is missing {dataset_name!r}"
            )
        values = np.asarray(particles[dataset_name])
        if values.ndim != 2 or values.shape[1] != 5:
            raise ValueError(
                f"{panel_name}/{dataset_name} must have shape (N, 5); "
                f"got {values.shape}"
            )
        validated[dataset_name] = values
    return validated


def write_stage1_particles(
    output_path,
    panel_particles: Mapping,
    packing_metadata: Mapping,
    panel_offsets: Mapping,
):
    """Atomically write the HDF5 schema consumed by stage-one datasets.

    Every particle is ``[packed_u, packed_v, simulated_x, simulated_y,
    simulated_z]``. Packing configuration is stored as attributes alongside
    the particle arrays.
    """

    output_path = Path(output_path)
    missing_sides = [side for side in SIDES if side not in panel_particles]
    if missing_sides:
        raise KeyError(f"Missing particle sides: {missing_sides}")

    strategy = str(packing_metadata["strategy"])
    padding = float(packing_metadata["padding_cm"])
    max_iterations = int(packing_metadata["max_iterations"])
    side_metadata = packing_metadata["sides"]

    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with h5py.File(temporary_path, "w", track_order=True) as dataset:
            dataset.attrs["format"] = "garmentparticles_stage1"
            dataset.attrs["format_version"] = 1
            dataset.attrs["packing_strategy"] = strategy
            dataset.attrs["packing_padding_cm"] = padding
            dataset.attrs["packing_max_iterations"] = max_iterations

            for side in SIDES:
                side_group = dataset.create_group(side, track_order=True)
                side_group.attrs["packing_iterations"] = int(
                    side_metadata[side]["iterations"]
                )
                side_group.attrs["packing_has_overlap"] = bool(
                    side_metadata[side]["has_overlap"]
                )

                for panel_name, particles in panel_particles[side].items():
                    if panel_name not in panel_offsets:
                        raise KeyError(
                            f"Missing packing offset for panel {panel_name!r}"
                        )
                    offset = np.asarray(panel_offsets[panel_name], dtype=float)
                    if offset.shape != (2,):
                        raise ValueError(
                            f"Packing offset for {panel_name!r} must have "
                            f"shape (2,); got {offset.shape}"
                        )

                    validated = _validated_particles(panel_name, particles)
                    panel_group = side_group.create_group(
                        panel_name,
                        track_order=True,
                    )
                    panel_group.attrs["packing_offset"] = offset
                    for dataset_name, values in validated.items():
                        panel_group.create_dataset(dataset_name, data=values)

        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


__all__ = ["write_stage1_particles"]
