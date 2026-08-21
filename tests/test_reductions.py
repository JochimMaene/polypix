"""Coverage reductions: Count and Sum, fused and by covering then reducing."""

from __future__ import annotations

import math

import numpy as np
import pytest

import polypix as px


def _xyz(longitude_deg: float, latitude_deg: float) -> tuple[float, float, float]:
    longitude = math.radians(longitude_deg)
    latitude = math.radians(latitude_deg)
    cos_latitude = math.cos(latitude)
    return (
        cos_latitude * math.cos(longitude),
        cos_latitude * math.sin(longitude),
        math.sin(latitude),
    )


def test_coverage_reductions_match_the_cell_lists_they_reduce() -> None:
    coverage = px.Coverage.from_arrays(
        cells=[1, 4, 4, 1, 2],
        offsets=[0, 2, 3, 3, 5],
        resolution=1,
    )

    dense_counts = (coverage).reduce(px.Count())
    assert dense_counts.dtype == np.int64
    assert dense_counts.shape == (48,)
    np.testing.assert_array_equal(dense_counts[[1, 2, 4, 7]], [2, 1, 2, 0])

    requested = np.asarray([4, 1, 4, 7, 2], dtype=np.int64)
    np.testing.assert_array_equal(
        (coverage).reduce(px.Count(), cells=requested),
        [2, 2, 2, 0, 1],
    )

    values = np.asarray([0.5, 2.0, 100.0, -0.25])
    dense_sums = (coverage).reduce(px.Sum(values))
    np.testing.assert_allclose(dense_sums[[1, 2, 4, 7]], [0.25, -0.25, 2.5, 0.0])
    np.testing.assert_allclose(
        (coverage).reduce(px.Sum(values), cells=requested),
        [2.5, 0.25, 2.5, 0.0, -0.25],
    )

    with pytest.raises(ValueError, match="one value per coverage segment"):
        (coverage).reduce(px.Sum([1.0]))

    empty = px.Coverage.from_arrays([], [0], resolution=29)
    for invalid in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="finite"):
            (empty).reduce(px.Sum(invalid), cells=[])
    with pytest.raises(ValueError, match="finite"):
        (coverage).reduce(px.Sum([np.nan, 2.0, 3.0, 4.0]), cells=[])
    assert (empty).reduce(px.Count(), cells=[]).dtype == np.int64
    assert (empty).reduce(px.Sum(1.0), cells=[]).dtype == np.float64


def test_fused_cap_counts_match_covering_then_reducing() -> None:
    centers = np.asarray([_xyz(-20, 10), _xyz(0, 0), _xyz(42, -12)], dtype=np.float64)
    radii = np.asarray([0.2, 0.3, 0.15])
    coverage = px.cover_cap(centers, radii, resolution=5, threads=1)

    np.testing.assert_array_equal(
        px.cover_cap(centers, radii, resolution=5, reduce=px.Count(), threads=1),
        (coverage).reduce(px.Count()),
    )
    requested = np.asarray([0, 8, 8, 100, 3000], dtype=np.int64)
    np.testing.assert_array_equal(
        px.cover_cap(
            centers,
            radii,
            resolution=5,
            candidate_cells=requested,
            reduce=px.Count(),
            threads=1,
        ),
        (coverage).reduce(px.Count(), cells=requested),
    )


def test_cap_counts_agree_on_both_sides_of_the_selected_work_estimate() -> None:
    """Both sides of the fuse/cover decision must return the same counts.

    ``cover_cap(reduce=Count(cells=...))`` fuses a small request and covers first
    for a large one, because fusing costs one cap test per requested cell.
    """
    rng = np.random.default_rng(20240819)
    centers = rng.normal(size=(400, 3))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    radii = np.full(400, 0.05)
    resolution = 7
    coverage = px.cover_cap(centers, radii, resolution, threads=1)

    total = px.cell_count(resolution)
    for requested in (
        np.asarray([], dtype=np.int64),
        np.asarray([0, 5, 5, total - 1], dtype=np.int64),
        rng.integers(0, total, size=40_000),
    ):
        np.testing.assert_array_equal(
            px.cover_cap(
                centers,
                radii,
                resolution,
                candidate_cells=requested,
                reduce=px.Count(),
                threads=1,
            ),
            coverage.reduce(px.Count(), cells=requested),
        )


def test_fused_cap_counts_stay_available_for_caps_too_large_to_store() -> None:
    """A whole-sphere cap at resolution 29 can only be answered by fusing."""
    requested = np.asarray([12 * 4**29 - 1, 0, 6 * 4**29, 0], dtype=np.uint64)
    np.testing.assert_array_equal(
        px.cover_cap(
            [-1.0, 0.0, 0.0], math.pi, 29, candidate_cells=requested, reduce=px.Count()
        ),
        np.ones(requested.size, dtype=np.int64),
    )


def test_selected_reductions_agree_on_both_sides_of_the_scratch_grid_choice() -> None:
    """Dense-scratch and hash accumulation must return identical results.

    A few hits with a tiny query take the hash path even where the dense grid
    would fit, so both paths have to be exercised at the same resolution.
    """
    rng = np.random.default_rng(20240820)
    resolution = 8
    total = px.cell_count(resolution)
    sparse = px.Coverage.from_arrays(
        cells=np.asarray([3, 700_000, total - 1, 3], dtype=np.int64),
        offsets=np.asarray([0, 2, 3, 4], dtype=np.int64),
        resolution=resolution,
    )
    values = np.asarray([1.5, -0.25, 2.0])
    for requested in (
        np.asarray([3, 3, total - 1, 0], dtype=np.int64),
        rng.integers(0, total, size=50_000),
    ):
        dense_counts = sparse.reduce(px.Count())
        dense_sums = sparse.reduce(px.Sum(values))
        np.testing.assert_array_equal(
            sparse.reduce(px.Count(), cells=requested), dense_counts[requested]
        )
        np.testing.assert_allclose(
            sparse.reduce(px.Sum(values), cells=requested), dense_sums[requested]
        )


def test_queried_reductions_match_the_dense_result_across_paths() -> None:
    # Resolution 6 is served from a dense scratch grid; resolution 20 is far
    # above the scratch budget and falls back to the hash path. Both must agree
    # with the dense reduction, preserving query order and duplicates.
    cells = np.asarray([0, 3, 17, 3, 400, 400, 401], dtype=np.int64)
    offsets = np.asarray([0, 3, 5, 7], dtype=np.int64)
    values = np.asarray([0.5, 2.0, -1.25], dtype=np.float64)
    queried = np.asarray([400, 3, 1, 3, 0, 17], dtype=np.int64)

    dense_reference = (px.Coverage.from_arrays(cells, offsets, resolution=6)).reduce(
        px.Count()
    )
    expected_counts = dense_reference[queried]

    for resolution in (6, 20):
        coverage = px.Coverage.from_arrays(cells, offsets, resolution=resolution)
        counts = (coverage).reduce(px.Count(), cells=queried)
        np.testing.assert_array_equal(counts, expected_counts)

        sums = (coverage).reduce(px.Sum(values), cells=queried)
        assert sums.dtype == np.float64
        np.testing.assert_allclose(sums, [0.75, 2.5, 0.0, 2.5, 0.5, 0.5])


def test_native_errors_map_to_distinct_python_exception_types() -> None:
    """Invalid input raises ValueError; an unsatisfiable allocation raises MemoryError."""
    coverage = px.Coverage.from_arrays([0, 1], [0, 1, 2], resolution=1)
    with pytest.raises(ValueError, match="one value per coverage segment"):
        coverage.reduce(px.Sum([1.0, 2.0, 3.0]))

    # A dense grid at the maximum resolution cannot be allocated on any machine,
    # so this exercises the allocation category rather than the input category.
    huge = px.Coverage.from_arrays([0], [0, 1], resolution=29)
    with pytest.raises(MemoryError, match="too large to fit in memory"):
        huge.reduce(px.Count())
