"""Occupancy of aligned coverage timelines: fused per-cell statistics."""

from __future__ import annotations

import re

import numpy as np
import pytest

import polypix as px


def _expected_stats(qualifying: np.ndarray) -> dict[str, list[int]]:
    """Statistics for a ``(segments, cells)`` boolean occupancy matrix.

    Derived from the occupancy itself rather than from anything Polypix
    returns, so the kernel's own run bookkeeping is never the reference.
    """
    expected: dict[str, list[int]] = {
        "cells": [],
        "run_counts": [],
        "internal_gap_steps_sum": [],
        "maximum_internal_gap_steps": [],
        "first_start": [],
        "last_stop": [],
    }
    for cell in range(qualifying.shape[1]):
        padded = np.concatenate(([False], qualifying[:, cell], [False]))
        changes = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1)
        if starts.size == 0:
            continue
        gaps = starts[1:] - stops[:-1]
        expected["cells"].append(cell)
        expected["run_counts"].append(int(starts.size))
        expected["internal_gap_steps_sum"].append(int(gaps.sum()))
        expected["maximum_internal_gap_steps"].append(
            int(gaps.max()) if gaps.size else 0
        )
        expected["first_start"].append(int(starts[0]))
        expected["last_stop"].append(int(stops[-1]))
    return expected


def test_occupancy_summarizes_complete_ordinal_windows() -> None:
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
    for values in (
        union.cells,
        union.run_counts,
        union.internal_gap_steps_sum,
        union.maximum_internal_gap_steps,
        union.first_start,
        union.last_stop,
    ):
        assert values.dtype == np.int64
        assert not values.flags.writeable
    assert union.resolution == 1
    assert union.segment_count == 5
    assert union.minimum_sources == 1
    assert union.source_count == 2
    # Cell 1 is occupied over [0, 3) and [4, 5), so one internal gap of one
    # segment; cell 2 over [1, 4) alone.
    np.testing.assert_array_equal(union.cells, [1, 2])
    np.testing.assert_array_equal(union.run_counts, [2, 1])
    np.testing.assert_array_equal(union.internal_gap_counts, [1, 0])
    np.testing.assert_array_equal(union.internal_gap_steps_sum, [1, 0])
    np.testing.assert_array_equal(union.maximum_internal_gap_steps, [1, 0])
    np.testing.assert_array_equal(union.first_start, [0, 1])
    np.testing.assert_array_equal(union.last_stop, [5, 4])

    coincident = px.occupancy([first, second], minimum_sources=2)
    np.testing.assert_array_equal(coincident.cells, [1, 2])
    np.testing.assert_array_equal(coincident.run_counts, [2, 1])
    np.testing.assert_array_equal(coincident.internal_gap_steps_sum, [2, 0])
    np.testing.assert_array_equal(coincident.maximum_internal_gap_steps, [2, 0])
    np.testing.assert_array_equal(coincident.first_start, [1, 2])
    np.testing.assert_array_equal(coincident.last_stop, [5, 3])

    impossible = px.occupancy([first, second], minimum_sources=3)
    np.testing.assert_array_equal(impossible.cells, [])
    np.testing.assert_array_equal(impossible.run_counts, [])
    np.testing.assert_array_equal(impossible.first_start, [])
    assert impossible.minimum_sources == 3
    assert impossible.source_count == 2

    enormous = px.occupancy([first, second], minimum_sources=10**100)
    np.testing.assert_array_equal(enormous.cells, [])
    assert enormous.minimum_sources == 10**100

    # Sequence positions are source entries; callers own source uniqueness.
    repeated = px.occupancy([first, first], minimum_sources=2)
    np.testing.assert_array_equal(repeated.cells, [1, 2])
    assert repeated.source_count == 2


def test_occupancy_validates_alignment_and_threshold() -> None:
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
    with pytest.raises(TypeError, match="timelines must be"):
        px.occupancy(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="timelines must contain"):
        px.occupancy([one, object()])  # type: ignore[list-item]


def test_occupancy_matches_random_boolean_oracle() -> None:
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
        expected = _expected_stats(occupancy.sum(axis=0) >= threshold)

        for name, values in expected.items():
            np.testing.assert_array_equal(getattr(actual, name), values, err_msg=name)
        np.testing.assert_array_equal(actual.internal_gap_counts, actual.run_counts - 1)


@pytest.mark.parametrize("resolution", [0, 2, 29], ids=["r0", "r2", "r29_sparse"])
def test_occupancy_matches_the_oracle_on_both_memory_profiles(
    resolution: int,
) -> None:
    """Resolution 29 cannot hold a dense state array, so it takes the hash path."""
    rng = np.random.default_rng(20260819)
    cell_count = min(px.cell_count(resolution), 24)
    for _ in range(40):
        segments = int(rng.integers(1, 12))
        source_count = int(rng.integers(1, 4))
        occupancy = rng.random((source_count, segments, cell_count)) < 0.4
        sources = []
        for source_occupancy in occupancy:
            cells: list[int] = []
            offsets = [0]
            for segment in source_occupancy:
                cells.extend(int(cell) for cell in np.flatnonzero(segment))
                offsets.append(len(cells))
            sources.append(
                px.Coverage.from_arrays(
                    np.asarray(cells, dtype=np.int64),
                    np.asarray(offsets, dtype=np.int64),
                    resolution=resolution,
                )
            )
        threshold = int(rng.integers(1, source_count + 2))
        stats = px.occupancy(sources, minimum_sources=threshold)
        expected = _expected_stats(occupancy.sum(axis=0) >= threshold)

        for name, values in expected.items():
            np.testing.assert_array_equal(getattr(stats, name), values, err_msg=name)
        assert stats.minimum_sources == threshold
        assert stats.source_count == source_count
        assert stats.segment_count == segments


def test_occupancy_accepts_unsorted_hits_and_all_empty_segments() -> None:
    shuffled = px.Coverage.from_arrays(
        cells=[7, 2, 7, 2],
        offsets=[0, 2, 4],
        resolution=1,
    )
    stats = px.occupancy(shuffled)
    np.testing.assert_array_equal(stats.cells, [2, 7])
    np.testing.assert_array_equal(stats.run_counts, [1, 1])
    np.testing.assert_array_equal(stats.first_start, [0, 0])
    np.testing.assert_array_equal(stats.last_stop, [2, 2])

    all_empty = px.Coverage.from_arrays([], [0, 0, 0, 0], resolution=29)
    empty = px.occupancy(all_empty)
    assert empty.segment_count == 3
    assert len(empty) == 0
    np.testing.assert_array_equal(empty.cells, [])


def test_occupancy_preserves_resolution_29_cell_ids() -> None:
    final_cell = px.cell_count(29) - 1
    coverage = px.Coverage.from_arrays(
        [final_cell, 0, final_cell],
        [0, 1, 2, 3],
        resolution=29,
    )

    stats = px.occupancy(coverage)

    # Cell 0 is occupied in segment 1 alone; the final cell in segments 0 and
    # 2, so two runs with one internal gap of one segment.
    np.testing.assert_array_equal(stats.cells, [0, final_cell])
    np.testing.assert_array_equal(stats.run_counts, [1, 2])
    np.testing.assert_array_equal(stats.first_start, [1, 0])
    np.testing.assert_array_equal(stats.last_stop, [2, 3])
    np.testing.assert_array_equal(stats.internal_gap_steps_sum, [0, 1])
    np.testing.assert_array_equal(stats.maximum_internal_gap_steps, [0, 1])


@pytest.mark.parametrize(
    ("offsets", "expected"),
    [
        ([0, 9], "offsets[-1] must equal"),
        ([4, 0], "must start at zero"),
        ([0, 4, 2], "must be nondecreasing"),
        ([2, 4], "must start at zero"),
        ([0, 2], "offsets[-1] must equal"),
    ],
)
def test_native_occupancy_rejects_malformed_offsets(
    offsets: list[int], expected: str
) -> None:
    """The native entry point indexes offsets directly, so it must check them.

    ``Coverage`` copies and validates what it is given, so the public API
    cannot reach this. The private function can be called with any arrays,
    and before validation these inputs either panicked on the slice index or,
    for a nonzero initial offset, silently dropped the leading hits.
    """
    from polypix._core import _occupancy_stats

    cells = np.array([1, 2, 3, 4], dtype=np.uint64)
    malformed = np.asarray(offsets, dtype=np.uint64)
    with pytest.raises(ValueError, match=r"^sources\[0\]: "):
        _occupancy_stats([cells], [malformed], 4, 1)
    with pytest.raises(ValueError, match=re.escape(expected)):
        _occupancy_stats([cells], [malformed], 4, 1)
