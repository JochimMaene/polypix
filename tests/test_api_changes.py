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


def test_canonical_polygon_api_accepts_packed_ragged_vertices() -> None:
    first = np.asarray(
        [_xyz(-8, -5), _xyz(6, -5), _xyz(6, 7), _xyz(-8, 7)],
        dtype=np.float64,
    )
    second = np.asarray(
        [_xyz(32, 11), _xyz(43, 11), _xyz(43, 20), _xyz(32, 20)],
        dtype=np.float64,
    )

    ragged = px.cover_convex_polygon([first, second], resolution=4)
    packed = px.cover_convex_polygon(
        np.vstack((first, second)),
        resolution=4,
        vertex_offsets=[0, len(first), len(first) + len(second)],
    )

    np.testing.assert_array_equal(packed.cells, ragged.cells)
    np.testing.assert_array_equal(packed.offsets, ragged.offsets)

    with pytest.raises(ValueError, match="polygons_xyz"):
        px.cover_convex_polygon(np.ones((2, 2)), resolution=1)
    for invalid_offsets in ([], [1, 8], [0, 9], [0, 6, 5, 8]):
        with pytest.raises(ValueError, match="vertex_offsets"):
            px.cover_convex_polygon(
                np.vstack((first, second)),
                resolution=4,
                vertex_offsets=invalid_offsets,
            )


def test_canonical_grid_names_and_cell_count() -> None:
    assert px.cell_count(0) == 12
    assert px.cell_count(6) == 12 * 4**6
    with pytest.raises(ValueError):
        px.cell_count(30)

    cells = np.asarray([0, 7, 31], dtype=np.int64)
    assert px.cell_centers(cells, 1).shape == (3, 3)
    assert px.cell_corners(cells, 1).shape == (3, 4, 3)


def test_coverage_helpers_preserve_segment_alignment() -> None:
    coverage = px.Coverage.from_arrays(
        cells=[1, 4, 4, 2, 7],
        offsets=[0, 2, 2, 3, 5],
        resolution=1,
    )

    assert coverage.cells.dtype == np.int64
    assert coverage.offsets.dtype == np.int64
    np.testing.assert_array_equal(coverage.segment_sizes, [2, 0, 1, 2])
    np.testing.assert_array_equal(coverage.segment_indices(), [0, 0, 2, 3, 3])

    filtered = coverage.filter_hits([True, False, True, False, True])
    np.testing.assert_array_equal(filtered.cells, [1, 4, 7])
    np.testing.assert_array_equal(filtered.offsets, [0, 1, 1, 2, 3])
    assert filtered.resolution == coverage.resolution

    with pytest.raises(ValueError, match="one value per covered cell"):
        coverage.filter_hits([True])
    with pytest.raises(TypeError, match="boolean"):
        coverage.filter_hits([1, 0, 1, 0, 1])
    empty = px.Coverage.from_arrays([], [0], resolution=1)
    with pytest.raises(TypeError, match="boolean"):
        empty.filter_hits(np.empty(0, dtype=np.int64))


def test_generic_coverage_reductions_match_membership_semantics() -> None:
    coverage = px.Coverage.from_arrays(
        cells=[1, 4, 4, 1, 2],
        offsets=[0, 2, 3, 3, 5],
        resolution=1,
    )

    dense_counts = px.count_coverage_per_cell(coverage)
    assert dense_counts.dtype == np.int64
    assert dense_counts.shape == (48,)
    np.testing.assert_array_equal(dense_counts[[1, 2, 4, 7]], [2, 1, 2, 0])

    requested = np.asarray([4, 1, 4, 7, 2], dtype=np.int64)
    np.testing.assert_array_equal(
        px.count_coverage_per_cell(coverage, cells=requested),
        [2, 2, 2, 0, 1],
    )

    values = np.asarray([0.5, 2.0, 100.0, -0.25])
    dense_sums = px.sum_coverage_per_cell(coverage, values)
    np.testing.assert_allclose(dense_sums[[1, 2, 4, 7]], [0.25, -0.25, 2.5, 0.0])
    np.testing.assert_allclose(
        px.sum_coverage_per_cell(coverage, values, cells=requested),
        [2.5, 0.25, 2.5, 0.0, -0.25],
    )

    with pytest.raises(ValueError, match="one value per coverage segment"):
        px.sum_coverage_per_cell(coverage, [1.0])

    empty = px.Coverage.from_arrays([], [0], resolution=29)
    for invalid in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="finite"):
            px.sum_coverage_per_cell(empty, invalid, cells=[])
    with pytest.raises(ValueError, match="finite"):
        px.sum_coverage_per_cell(coverage, [np.nan, 2.0, 3.0, 4.0], cells=[])
    assert px.count_coverage_per_cell(empty, cells=[]).dtype == np.int64
    assert px.sum_coverage_per_cell(empty, 1.0, cells=[]).dtype == np.float64


def test_empty_cap_batches_still_validate_scalar_radii() -> None:
    empty = np.empty((0, 3), dtype=np.float64)
    for invalid in (-1.0, math.nextafter(math.pi, math.inf), np.nan, np.inf):
        with pytest.raises(ValueError, match="radii_rad"):
            px.cover_cap(empty, invalid, resolution=1)
        with pytest.raises(ValueError, match="radii_rad"):
            px.count_caps_per_cell(empty, invalid, resolution=1)


def test_fused_cap_counts_match_generic_reduction() -> None:
    centers = np.asarray([_xyz(-20, 10), _xyz(0, 0), _xyz(42, -12)], dtype=np.float64)
    radii = np.asarray([0.2, 0.3, 0.15])
    coverage = px.cover_cap(centers, radii, resolution=5, threads=1)

    np.testing.assert_array_equal(
        px.count_caps_per_cell(centers, radii, resolution=5, threads=1),
        px.count_coverage_per_cell(coverage),
    )
    requested = np.asarray([0, 8, 8, 100, 3000], dtype=np.int64)
    np.testing.assert_array_equal(
        px.count_caps_per_cell(
            centers,
            radii,
            resolution=5,
            cells=requested,
            threads=1,
        ),
        px.count_coverage_per_cell(coverage, cells=requested),
    )


def test_occupancy_runs_preserve_complete_ordinal_windows() -> None:
    first = px.Coverage.from_arrays(
        cells=[1, 1, 2, 2, 1],
        offsets=[0, 1, 3, 4, 4, 5],
        resolution=1,
    )
    second = px.Coverage.from_arrays(
        cells=[1, 1, 2, 2, 1],
        offsets=[0, 0, 1, 3, 4, 5],
        resolution=1,
    )

    union = px.occupancy_runs([first, second])
    assert union.cells.dtype == np.int64
    assert union.offsets.dtype == np.int64
    assert union.starts.dtype == np.int64
    assert union.stops.dtype == np.int64
    assert union.resolution == 1
    assert union.segment_count == 5
    assert union.minimum_sources == 1
    assert union.source_count == 2
    np.testing.assert_array_equal(union.cells, [1, 2])
    np.testing.assert_array_equal(union.offsets, [0, 2, 3])
    np.testing.assert_array_equal(union.starts, [0, 4, 1])
    np.testing.assert_array_equal(union.stops, [3, 5, 4])
    np.testing.assert_array_equal(union.run_counts, [2, 1])

    coincident = px.occupancy_runs([first, second], minimum_sources=2)
    np.testing.assert_array_equal(coincident.cells, [1, 2])
    np.testing.assert_array_equal(coincident.offsets, [0, 2, 3])
    np.testing.assert_array_equal(coincident.starts, [1, 4, 2])
    np.testing.assert_array_equal(coincident.stops, [2, 5, 3])

    impossible = px.occupancy_runs([first, second], minimum_sources=3)
    np.testing.assert_array_equal(impossible.cells, [])
    np.testing.assert_array_equal(impossible.offsets, [0])
    np.testing.assert_array_equal(impossible.starts, [])
    np.testing.assert_array_equal(impossible.stops, [])
    assert impossible.minimum_sources == 3
    assert impossible.source_count == 2

    enormous = px.occupancy_runs([first, second], minimum_sources=10**100)
    np.testing.assert_array_equal(enormous.cells, [])
    np.testing.assert_array_equal(enormous.offsets, [0])
    assert enormous.minimum_sources == 10**100

    # Sequence positions are source entries; callers own source uniqueness.
    repeated = px.occupancy_runs([first, first], minimum_sources=2)
    np.testing.assert_array_equal(repeated.cells, [1, 2])
    assert repeated.source_count == 2


def test_occupancy_runs_validates_alignment_and_threshold() -> None:
    one = px.Coverage.from_arrays([1], [0, 1], resolution=1)
    two_segments = px.Coverage.from_arrays([1], [0, 1, 1], resolution=1)
    other_resolution = px.Coverage.from_arrays([1], [0, 1], resolution=2)

    with pytest.raises(ValueError, match="at least one"):
        px.occupancy_runs([])
    with pytest.raises(ValueError, match="same number of segments"):
        px.occupancy_runs([one, two_segments])
    with pytest.raises(ValueError, match="same resolution"):
        px.occupancy_runs([one, other_resolution])
    with pytest.raises(ValueError, match="positive integer"):
        px.occupancy_runs(one, minimum_sources=0)
    with pytest.raises(TypeError, match="positive integer"):
        px.occupancy_runs(one, minimum_sources=True)


def test_occupancy_runs_matches_random_boolean_oracle() -> None:
    random = np.random.default_rng(20260819)
    for _ in range(60):
        source_count = int(random.integers(1, 5))
        segment_count = int(random.integers(0, 11))
        occupancy = random.random((source_count, segment_count, 12)) < 0.22
        sources: list[px.Coverage] = []
        for source_occupancy in occupancy:
            segments = [
                np.flatnonzero(segment).astype(np.int64) for segment in source_occupancy
            ]
            sizes = np.asarray([segment.size for segment in segments], dtype=np.int64)
            offsets = np.concatenate(
                (np.zeros(1, dtype=np.int64), np.cumsum(sizes, dtype=np.int64))
            )
            cells = (
                np.concatenate(segments) if segments else np.empty(0, dtype=np.int64)
            )
            sources.append(px.Coverage.from_arrays(cells, offsets, resolution=0))

        threshold = int(random.integers(1, source_count + 2))
        actual = px.occupancy_runs(sources, minimum_sources=threshold)
        qualifying = occupancy.sum(axis=0) >= threshold
        expected_cells: list[int] = []
        expected_offsets = [0]
        expected_starts: list[int] = []
        expected_stops: list[int] = []
        for cell in range(12):
            values = qualifying[:, cell]
            padded = np.concatenate(([False], values, [False]))
            changes = np.diff(padded.astype(np.int8))
            starts = np.flatnonzero(changes == 1)
            stops = np.flatnonzero(changes == -1)
            if starts.size == 0:
                continue
            expected_cells.append(cell)
            expected_starts.extend(starts.tolist())
            expected_stops.extend(stops.tolist())
            expected_offsets.append(len(expected_starts))

        np.testing.assert_array_equal(actual.cells, expected_cells)
        np.testing.assert_array_equal(actual.offsets, expected_offsets)
        np.testing.assert_array_equal(actual.starts, expected_starts)
        np.testing.assert_array_equal(actual.stops, expected_stops)
        for values in (actual.cells, actual.offsets, actual.starts, actual.stops):
            assert not values.flags.writeable


def test_sweep_accepts_an_empty_interval_axis() -> None:
    empty = np.empty((0, 3), dtype=np.float64)
    one = np.asarray([_xyz(0, 0)], dtype=np.float64)

    for edge in (empty, one):
        coverage = px.cover_sweep(edge, edge, resolution=2)
        assert coverage.segment_count == 0
        np.testing.assert_array_equal(coverage.cells, [])
        np.testing.assert_array_equal(coverage.offsets, [0])

    for invalid in ([0.0, 0.0, 0.0], [np.nan, 0.0, 1.0]):
        edge = np.asarray([invalid], dtype=np.float64)
        with pytest.raises(ValueError, match="left_edge_xyz"):
            px.cover_sweep(edge, edge, resolution=2)


def test_occupancy_runs_accept_unsorted_hits_and_all_empty_segments() -> None:
    shuffled = px.Coverage.from_arrays(
        cells=[7, 2, 7, 2],
        offsets=[0, 2, 4],
        resolution=1,
    )
    runs = px.occupancy_runs(shuffled)
    np.testing.assert_array_equal(runs.cells, [2, 7])
    np.testing.assert_array_equal(runs.offsets, [0, 1, 2])
    np.testing.assert_array_equal(runs.starts, [0, 0])
    np.testing.assert_array_equal(runs.stops, [2, 2])

    all_empty = px.Coverage.from_arrays([], [0, 0, 0, 0], resolution=29)
    empty_runs = px.occupancy_runs(all_empty)
    assert empty_runs.segment_count == 3
    np.testing.assert_array_equal(empty_runs.cells, [])
    np.testing.assert_array_equal(empty_runs.offsets, [0])


def test_occupancy_runs_preserve_resolution_29_cell_ids() -> None:
    final_cell = px.cell_count(29) - 1
    coverage = px.Coverage.from_arrays(
        [final_cell, 0, final_cell],
        [0, 1, 2, 3],
        resolution=29,
    )

    runs = px.occupancy_runs(coverage)

    np.testing.assert_array_equal(runs.cells, [0, final_cell])
    np.testing.assert_array_equal(runs.offsets, [0, 1, 3])
    np.testing.assert_array_equal(runs.starts, [1, 0, 2])
    np.testing.assert_array_equal(runs.stops, [2, 1, 3])
