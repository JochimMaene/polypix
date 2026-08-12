"""Small executable examples mirrored by the task-oriented documentation."""

from __future__ import annotations

import numpy as np

import polypix as px


def test_direction_recipe_round_trips_cell_centers() -> None:
    directions = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [-2.0, 1.0, 0.5]])
    cells = px.cell_at(directions, resolution=8)
    assert cells.shape == (3,)

    sample = np.asarray([0, 17, 123], dtype=np.uint64)
    np.testing.assert_array_equal(px.cell_at(px.centers(sample, 3), 3), sample)


def test_region_and_coverage_recipes() -> None:
    cap = px.cover_cap([1.0, 0.0, 0.0], np.deg2rad(5.0), resolution=8)
    assert cap.segment_count == 1

    footprints = [
        np.asarray([[1.0, -0.1, -0.1], [1.0, 0.1, -0.1], [1.0, 0.0, 0.1]]),
        np.asarray(
            [
                [1.0, -0.2, -0.1],
                [1.0, 0.2, -0.1],
                [1.0, 0.2, 0.1],
                [1.0, -0.2, 0.1],
            ]
        ),
    ]
    coverage = px.cover_footprint(footprints, resolution=8)
    assert len(coverage) == 2
    assert all(not segment.flags.writeable for segment in coverage)

    imported = px.Coverage.from_arrays(
        cells=[2, 7, 9],
        offsets=[0, 2, 3],
        resolution=1,
    )
    assert imported.segment_count == 2
    assert not imported.cells.flags.writeable


def test_sweep_and_occupancy_recipes() -> None:
    left = np.asarray([[1.0, -0.1, -0.1], [1.0, -0.1, 0.1], [1.0, -0.1, 0.3]])
    right = np.asarray([[1.0, 0.1, -0.1], [1.0, 0.1, 0.1], [1.0, 0.1, 0.3]])
    assert px.cover_sweep(left, right, resolution=8).segment_count == 2

    source_a = px.Coverage.from_arrays([7, 7], [0, 1, 2, 2, 2, 2], resolution=1)
    source_b = px.Coverage.from_arrays([7, 7], [0, 0, 0, 1, 1, 2], resolution=1)
    summary = px.summarize_occupancy([source_a, source_b])
    np.testing.assert_array_equal(summary.run_counts, [3])
    np.testing.assert_array_equal(summary.merged_gap_counts, [1])
    np.testing.assert_array_equal(summary.merged_gap_steps_sum, [1])


def test_cap_count_recipe() -> None:
    cap_centers = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    radii_rad = np.deg2rad([5.0, 8.0])
    counts = px.count_caps_per_cell(cap_centers, radii_rad, resolution=8)
    assert counts.shape == (12 * 4**8,)
    assert counts.dtype == np.int64
