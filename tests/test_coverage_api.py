"""Input shapes, grid helpers, and segment alignment of the covering calls."""

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


def test_polygon_api_accepts_packed_and_ragged_vertices() -> None:
    first = np.asarray(
        [_xyz(-8, -5), _xyz(6, -5), _xyz(6, 7), _xyz(-8, 7)],
        dtype=np.float64,
    )
    second = np.asarray(
        [_xyz(32, 11), _xyz(43, 11), _xyz(43, 20), _xyz(32, 20)],
        dtype=np.float64,
    )

    # A uniform sequence, a dense array, and a ragged sequence containing the
    # same two polygons must all agree.
    sequence = px.cover_convex_polygon([first, second], resolution=4)
    dense = px.cover_convex_polygon(np.stack((first, second)), resolution=4)
    triangle = second[:3]
    ragged = px.cover_convex_polygon([first, triangle], resolution=4)

    np.testing.assert_array_equal(dense.cells, sequence.cells)
    np.testing.assert_array_equal(dense.offsets, sequence.offsets)
    assert len(ragged) == 2
    np.testing.assert_array_equal(ragged[0], sequence[0])

    with pytest.raises(ValueError, match="polygons_xyz"):
        px.cover_convex_polygon(np.ones((2, 2)), resolution=1)


def test_malformed_ragged_polygon_uses_the_public_validation_error() -> None:
    quad = np.asarray(
        [_xyz(-8, -5), _xyz(6, -5), _xyz(6, 7), _xyz(-8, 7)],
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match=r"polygons_xyz\[1\].*shape"):
        px.cover_convex_polygon([quad, 1], resolution=4)


def test_grid_names_and_cell_count() -> None:
    assert px.cell_count(0) == 12
    assert px.cell_count(6) == 12 * 4**6
    with pytest.raises(ValueError):
        px.cell_count(30)

    cells = np.asarray([0, 7, 31], dtype=np.int64)
    assert px.cell_centers(cells, 1).shape == (3, 3)
    assert px.cell_corners(cells, 1).shape == (3, 4, 3)


@pytest.mark.parametrize("dtype", [np.float64, "<U3"])
def test_typed_empty_cell_arrays_require_an_integer_dtype(dtype: object) -> None:
    with pytest.raises(TypeError, match="integers"):
        px.cell_centers(np.empty(0, dtype=dtype), resolution=4)

    with pytest.raises(TypeError, match="integers"):
        px.Coverage.from_arrays(np.empty(0, dtype=dtype), [0], resolution=4)


def test_coverage_helpers_preserve_segment_alignment() -> None:
    coverage = px.Coverage.from_arrays(
        cells=[1, 4, 4, 2, 7],
        offsets=[0, 2, 2, 3, 5],
        resolution=1,
    )

    assert coverage.cells.dtype == np.int64
    assert coverage.offsets.dtype == np.int64
    np.testing.assert_array_equal(np.diff(coverage.offsets), [2, 0, 1, 2])
    np.testing.assert_array_equal(
        np.repeat(np.arange(len(coverage)), np.diff(coverage.offsets)), [0, 0, 2, 3, 3]
    )

    empty = px.Coverage.from_arrays([], [0], resolution=1)
    np.testing.assert_array_equal(np.diff(empty.offsets), [])
    np.testing.assert_array_equal(
        np.repeat(np.arange(len(empty)), np.diff(empty.offsets)), []
    )


def test_empty_cap_batches_validate_scalar_radii() -> None:
    empty = np.empty((0, 3), dtype=np.float64)
    for invalid in (-1.0, math.nextafter(math.pi, math.inf), np.nan, np.inf):
        with pytest.raises(ValueError, match="radii_rad"):
            px.cover_cap(empty, invalid, resolution=1)
        with pytest.raises(ValueError, match="radii_rad"):
            px.cover_cap(empty, invalid, resolution=1, reduce=px.Count())


def test_reducer_tokens_use_identity_equality() -> None:
    for first, second in (
        (px.Count(), px.Count()),
        (px.Sum(np.asarray([1.0, 2.0])), px.Sum(np.asarray([1.0, 2.0]))),
    ):
        assert first == first
        assert first != second


def test_sweep_accepts_an_empty_interval_axis() -> None:
    empty = np.empty((0, 3), dtype=np.float64)
    one = np.asarray([_xyz(0, 0)], dtype=np.float64)

    for edge in (empty, one):
        coverage = px.cover_sweep(edge, edge, resolution=2)
        assert len(coverage) == 0
        np.testing.assert_array_equal(coverage.cells, [])
        np.testing.assert_array_equal(coverage.offsets, [0])

    for invalid in ([0.0, 0.0, 0.0], [np.nan, 0.0, 1.0]):
        edge = np.asarray([invalid], dtype=np.float64)
        with pytest.raises(ValueError, match="left_edge_xyz"):
            px.cover_sweep(edge, edge, resolution=2)
