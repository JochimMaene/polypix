from __future__ import annotations

import math

import numpy as np
import pytest

import polypix as px


def _xyz(lon_deg: float, lat_deg: float) -> np.ndarray:
    lon = math.radians(lon_deg)
    lat = math.radians(lat_deg)
    cos_lat = math.cos(lat)
    return np.asarray(
        [cos_lat * math.cos(lon), cos_lat * math.sin(lon), math.sin(lat)],
        dtype=np.float64,
    )


def _footprint(
    vertices: list[tuple[float, float]],
    *,
    scale: float = 1.0,
) -> np.ndarray:
    return scale * np.asarray([_xyz(lon, lat) for lon, lat in vertices])


def _segments(coverage: px.Coverage) -> list[np.ndarray]:
    return [
        coverage.cells[start:stop]
        for start, stop in zip(coverage.offsets[:-1], coverage.offsets[1:], strict=True)
    ]


def test_coverage_uses_standard_ring_indices_and_stores_resolution() -> None:
    resolution = 3
    polygon = _footprint([(-15.0, -8.0), (15.0, -8.0), (15.0, 8.0), (-15.0, 8.0)])

    coverage = px.cover_footprint(polygon, resolution)

    assert coverage.resolution == resolution
    assert coverage.cells.dtype == np.uint64
    assert np.all(coverage.cells < 12 * 4**resolution)
    np.testing.assert_array_equal(
        coverage.offsets,
        np.asarray([0, coverage.cells.size], dtype=np.uint64),
    )
    np.testing.assert_array_equal(
        coverage.counts,
        np.asarray([coverage.cells.size], dtype=np.intp),
    )


def test_cover_footprint_accepts_ragged_batches_and_normalizes_vectors() -> None:
    first = _footprint(
        [(-15.0, -8.0), (15.0, -8.0), (15.0, 8.0), (-15.0, 8.0)],
        scale=1e300,
    )
    second = _footprint(
        [(20.0, -8.0), (43.0, -8.0), (47.0, 0.0), (43.0, 8.0), (20.0, 8.0)],
        scale=1e-300,
    )

    actual = px.cover_footprint([first, second], resolution=4)
    expected = [
        px.cover_footprint(first / 1e300, resolution=4).cells,
        px.cover_footprint(second / 1e-300, resolution=4).cells,
    ]

    assert actual.offsets.shape == (3,)
    for actual_segment, expected_segment in zip(
        _segments(actual), expected, strict=True
    ):
        np.testing.assert_array_equal(actual_segment, expected_segment)


def test_empty_dense_batch_preserves_one_initial_offset() -> None:
    coverage = px.cover_footprint(np.empty((0, 4, 3)), resolution=4)

    assert coverage.cells.shape == (0,)
    np.testing.assert_array_equal(coverage.offsets, [0])
    assert coverage.counts.shape == (0,)


def test_candidate_cells_are_standard_indices_with_set_semantics() -> None:
    polygon = _footprint([(-20.0, -10.0), (20.0, -10.0), (20.0, 10.0), (-20.0, 10.0)])
    full = px.cover_footprint(polygon, resolution=4)
    candidates = np.concatenate((full.cells[::2][::-1], full.cells[:2]))

    restricted = px.cover_footprint(
        polygon,
        resolution=4,
        candidate_cells=candidates,
    )

    np.testing.assert_array_equal(
        np.sort(restricted.cells),
        np.intersect1d(full.cells, candidates),
    )
    assert np.unique(restricted.cells).size == restricted.cells.size


def test_empty_candidate_cells_return_empty_segments() -> None:
    polygon = _footprint([(-20.0, -10.0), (20.0, -10.0), (20.0, 10.0), (-20.0, 10.0)])

    restricted = px.cover_footprint(
        np.repeat(polygon[np.newaxis, :, :], 2, axis=0),
        resolution=4,
        candidate_cells=[],
    )

    assert restricted.cells.shape == (0,)
    np.testing.assert_array_equal(restricted.offsets, [0, 0, 0])


def test_cover_strip_preserves_interval_segments() -> None:
    left = np.asarray([_xyz(-5.0, -5.0), _xyz(-4.0, 0.0), _xyz(-3.0, 5.0)])
    right = 7.0 * np.asarray([_xyz(5.0, -5.0), _xyz(4.0, 0.0), _xyz(3.0, 5.0)])
    footprints = np.asarray(
        [
            [left[0], right[0], right[1], left[1]],
            [left[1], right[1], right[2], left[2]],
        ]
    )

    actual = px.cover_strip(left, right, resolution=3)
    expected = px.cover_footprint(footprints, resolution=3)

    np.testing.assert_array_equal(actual.offsets, expected.offsets)
    np.testing.assert_array_equal(actual.cells, expected.cells)
    assert not hasattr(px, "cover_swath")


def test_cell_geometry_accepts_standard_indices_and_returns_xyz() -> None:
    cells = np.asarray([0, 17, 123], dtype=np.uint64)

    center_vectors = px.centers(cells, resolution=3)
    boundary_vectors = px.boundaries(cells, resolution=3)

    assert center_vectors.shape == (3, 3)
    assert boundary_vectors.shape == (3, 4, 3)
    np.testing.assert_allclose(np.linalg.norm(center_vectors, axis=1), 1.0)
    np.testing.assert_allclose(np.linalg.norm(boundary_vectors, axis=2), 1.0)


def test_empty_cell_geometry_has_stable_shapes() -> None:
    assert px.centers([], resolution=3).shape == (0, 3)
    assert px.boundaries([], resolution=3).shape == (0, 4, 3)


def test_cells_are_validated_against_resolution() -> None:
    with pytest.raises(ValueError, match="valid RING indices at resolution 2"):
        px.centers([12 * 4**2], resolution=2)

    polygon = _footprint([(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)])
    with pytest.raises(ValueError, match="valid RING indices at resolution 2"):
        px.cover_footprint(
            polygon,
            resolution=2,
            candidate_cells=[12 * 4**2],
        )


@pytest.mark.parametrize("threads", [0, -1])
def test_threads_must_be_positive(threads: int) -> None:
    polygon = _footprint([(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)])
    with pytest.raises(ValueError, match="positive integer"):
        px.cover_footprint(polygon, resolution=2, threads=threads)


def test_threads_produce_identical_membership_and_order() -> None:
    polygon = _footprint([(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)])
    batch = np.repeat(polygon[np.newaxis, :, :], 300, axis=0)

    sequential = px.cover_footprint(batch, resolution=3, threads=1)
    parallel = px.cover_footprint(batch, resolution=3, threads=2)
    automatic = px.cover_footprint(batch, resolution=3)

    np.testing.assert_array_equal(parallel.cells, sequential.cells)
    np.testing.assert_array_equal(parallel.offsets, sequential.offsets)
    np.testing.assert_array_equal(automatic.cells, sequential.cells)
    np.testing.assert_array_equal(automatic.offsets, sequential.offsets)


def test_geometry_rejects_nonfinite_zero_and_complex_vectors() -> None:
    valid = _footprint([(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)])

    nonfinite = valid.copy()
    nonfinite[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        px.cover_footprint(nonfinite, resolution=2)

    zero = valid.copy()
    zero[0] = 0.0
    with pytest.raises(ValueError, match="zero-length"):
        px.cover_footprint(zero, resolution=2)

    with pytest.raises(TypeError, match="complex"):
        px.cover_footprint(valid.astype(np.complex128), resolution=2)


def test_public_surface_is_minimal() -> None:
    assert px.__all__ == [
        "Coverage",
        "__version__",
        "boundaries",
        "centers",
        "cover_footprint",
        "cover_strip",
    ]
