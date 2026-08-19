"""Occupancy of aligned coverage: ordinal runs and fused per-cell statistics."""

from __future__ import annotations

import numpy as np
import pytest

import polypix as px


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

    union = px.occupancy([first, second])
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

    coincident = px.occupancy([first, second], minimum_sources=2)
    np.testing.assert_array_equal(coincident.cells, [1, 2])
    np.testing.assert_array_equal(coincident.offsets, [0, 2, 3])
    np.testing.assert_array_equal(coincident.starts, [1, 4, 2])
    np.testing.assert_array_equal(coincident.stops, [2, 5, 3])

    impossible = px.occupancy([first, second], minimum_sources=3)
    np.testing.assert_array_equal(impossible.cells, [])
    np.testing.assert_array_equal(impossible.offsets, [0])
    np.testing.assert_array_equal(impossible.starts, [])
    np.testing.assert_array_equal(impossible.stops, [])
    assert impossible.minimum_sources == 3
    assert impossible.source_count == 2

    enormous = px.occupancy([first, second], minimum_sources=10**100)
    np.testing.assert_array_equal(enormous.cells, [])
    np.testing.assert_array_equal(enormous.offsets, [0])
    assert enormous.minimum_sources == 10**100

    # Sequence positions are source entries; callers own source uniqueness.
    repeated = px.occupancy([first, first], minimum_sources=2)
    np.testing.assert_array_equal(repeated.cells, [1, 2])
    assert repeated.source_count == 2


def test_occupancy_runs_validates_alignment_and_threshold() -> None:
    one = px.Coverage.from_arrays([1], [0, 1], resolution=1)
    two_segments = px.Coverage.from_arrays([1], [0, 1, 1], resolution=1)
    other_resolution = px.Coverage.from_arrays([1], [0, 1], resolution=2)

    with pytest.raises(ValueError, match="at least one"):
        px.occupancy([])
    with pytest.raises(ValueError, match="same number of segments"):
        px.occupancy([one, two_segments])
    with pytest.raises(ValueError, match="same resolution"):
        px.occupancy([one, other_resolution])
    with pytest.raises(ValueError, match="positive integer"):
        px.occupancy(one, minimum_sources=0)
    with pytest.raises(TypeError, match="positive integer"):
        px.occupancy(one, minimum_sources=True)


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
        actual = px.occupancy(sources, minimum_sources=threshold)
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


def test_occupancy_runs_accept_unsorted_hits_and_all_empty_segments() -> None:
    shuffled = px.Coverage.from_arrays(
        cells=[7, 2, 7, 2],
        offsets=[0, 2, 4],
        resolution=1,
    )
    runs = px.occupancy(shuffled)
    np.testing.assert_array_equal(runs.cells, [2, 7])
    np.testing.assert_array_equal(runs.offsets, [0, 1, 2])
    np.testing.assert_array_equal(runs.starts, [0, 0])
    np.testing.assert_array_equal(runs.stops, [2, 2])

    all_empty = px.Coverage.from_arrays([], [0, 0, 0, 0], resolution=29)
    empty_runs = px.occupancy(all_empty)
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

    runs = px.occupancy(coverage)

    np.testing.assert_array_equal(runs.cells, [0, final_cell])
    np.testing.assert_array_equal(runs.offsets, [0, 1, 3])
    np.testing.assert_array_equal(runs.starts, [1, 0, 2])
    np.testing.assert_array_equal(runs.stops, [2, 1, 3])


def _stats_from_runs(runs: px.OccupancyRuns) -> dict[str, np.ndarray]:
    """Derive the fused statistics from lossless runs, the slow way."""
    counts = runs.run_counts
    gap_sum = np.zeros(runs.cells.size, dtype=np.int64)
    gap_max = np.zeros(runs.cells.size, dtype=np.int64)
    for index in range(runs.cells.size):
        start, stop = runs.offsets[index], runs.offsets[index + 1]
        gaps = runs.starts[start + 1 : stop] - runs.stops[start : stop - 1]
        if gaps.size:
            gap_sum[index] = gaps.sum()
            gap_max[index] = gaps.max()
    return {
        "run_counts": counts,
        "internal_gap_steps_sum": gap_sum,
        "maximum_internal_gap_steps": gap_max,
        "first_start": runs.starts[runs.offsets[:-1]],
        "last_stop": runs.stops[runs.offsets[1:] - 1],
    }


@pytest.mark.parametrize("resolution", [0, 2, 29], ids=["r0", "r2", "r29_sparse"])
def test_occupancy_stats_match_statistics_derived_from_runs(resolution: int) -> None:
    rng = np.random.default_rng(20260819)
    cell_count = min(px.cell_count(resolution), 24)
    for _ in range(40):
        segments = int(rng.integers(1, 12))
        source_count = int(rng.integers(1, 4))
        sources = []
        for _ in range(source_count):
            cells, offsets = [], [0]
            for _ in range(segments):
                chosen = np.flatnonzero(rng.random(cell_count) < 0.4)
                cells.extend(int(cell) for cell in chosen)
                offsets.append(len(cells))
            sources.append(
                px.Coverage.from_arrays(
                    np.asarray(cells, dtype=np.int64),
                    np.asarray(offsets, dtype=np.int64),
                    resolution=resolution,
                )
            )
        threshold = int(rng.integers(1, source_count + 2))
        stats = px.occupancy(sources, minimum_sources=threshold, into=px.Stats())
        runs = px.occupancy(sources, minimum_sources=threshold)

        np.testing.assert_array_equal(stats.cells, runs.cells)
        for name, expected in _stats_from_runs(runs).items():
            np.testing.assert_array_equal(getattr(stats, name), expected, err_msg=name)
        np.testing.assert_array_equal(stats.internal_gap_counts, runs.run_counts - 1)
        assert stats.minimum_sources == threshold
        assert stats.source_count == source_count
        assert stats.segment_count == segments
