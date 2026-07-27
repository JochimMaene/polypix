from __future__ import annotations

import math

import numpy as np
import pytest

import polypix as px


def _lonlat_to_xyz(lon_deg: float, lat_deg: float) -> tuple[float, float, float]:
    lon = math.radians(lon_deg)
    lat = math.radians(lat_deg)
    cos_lat = math.cos(lat)
    return cos_lat * math.cos(lon), cos_lat * math.sin(lon), math.sin(lat)


@pytest.fixture(scope="module")
def footprints() -> np.ndarray:
    rows: list[list[tuple[float, float, float]]] = []
    for lat in np.linspace(-55.0, 55.0, 20):
        for lon in np.linspace(-160.0, 160.0, 25):
            rows.append(
                [
                    _lonlat_to_xyz(float(lon - 0.35), float(lat - 0.25)),
                    _lonlat_to_xyz(float(lon + 0.35), float(lat - 0.25)),
                    _lonlat_to_xyz(float(lon + 0.35), float(lat + 0.25)),
                    _lonlat_to_xyz(float(lon - 0.35), float(lat + 0.25)),
                ]
            )
    return np.asarray(rows, dtype=np.float64)


@pytest.fixture(scope="module")
def large_footprints() -> np.ndarray:
    longitudes = np.linspace(-170.0, 170.0, 64)
    latitudes = np.linspace(-60.0, 60.0, 64)
    return np.asarray(
        [
            [
                _lonlat_to_xyz(float(lon - 0.05), float(lat - 0.05)),
                _lonlat_to_xyz(float(lon + 0.05), float(lat - 0.05)),
                _lonlat_to_xyz(float(lon + 0.05), float(lat + 0.05)),
                _lonlat_to_xyz(float(lon - 0.05), float(lat + 0.05)),
            ]
            for lat in latitudes
            for lon in longitudes
        ],
        dtype=np.float64,
    )


@pytest.fixture(scope="module")
def strip_edges() -> tuple[np.ndarray, np.ndarray]:
    latitudes = np.linspace(-40.0, 40.0, 501)
    left = np.asarray(
        [_lonlat_to_xyz(-5.0, float(lat)) for lat in latitudes], dtype=np.float64
    )
    right = np.asarray(
        [_lonlat_to_xyz(5.0, float(lat)) for lat in latitudes], dtype=np.float64
    )
    return left, right


@pytest.fixture(scope="module")
def large_strip_edges() -> tuple[np.ndarray, np.ndarray]:
    latitudes = np.linspace(-60.0, 60.0, 4097)
    left = np.asarray(
        [_lonlat_to_xyz(-2.0, float(lat)) for lat in latitudes], dtype=np.float64
    )
    right = np.asarray(
        [_lonlat_to_xyz(2.0, float(lat)) for lat in latitudes], dtype=np.float64
    )
    return left, right


@pytest.fixture(scope="module")
def cells(footprints: np.ndarray) -> np.ndarray:
    return px.cover_footprint(footprints, resolution=7, threads=1).cells


@pytest.fixture(scope="module")
def sparse_resolution_12_cells() -> np.ndarray:
    resolution = 12
    pixel_count = 12 * (4**resolution)
    ring_indices = np.arange(1024, dtype=np.uint64) * np.uint64(pixel_count // 1024)
    return ring_indices


@pytest.fixture(scope="module")
def large_sparse_resolution_12_cells() -> np.ndarray:
    resolution = 12
    pixel_count = 12 * 4**resolution
    return np.arange(65536, dtype=np.uint64) * np.uint64(pixel_count // 65536)


@pytest.mark.parametrize(
    "resolution", [4, 6, 7], ids=lambda value: f"resolution_{value}"
)
def test_cover_footprint_batch(
    benchmark, footprints: np.ndarray, resolution: int
) -> None:
    coverage = benchmark(px.cover_footprint, footprints, resolution, threads=1)

    assert coverage.offsets.shape == (footprints.shape[0] + 1,)
    assert coverage.cells.dtype == np.uint64


def test_cover_footprint_single_latency(benchmark, footprints: np.ndarray) -> None:
    coverage = benchmark(px.cover_footprint, footprints[0], 7, threads=1)

    assert coverage.offsets.shape == (2,)


def test_cover_footprint_automatic_parallel(
    benchmark, large_footprints: np.ndarray
) -> None:
    coverage = benchmark(px.cover_footprint, large_footprints, 9)

    assert coverage.offsets.shape == (large_footprints.shape[0] + 1,)


@pytest.mark.parametrize("resolution", [6, 12], ids=lambda value: f"resolution_{value}")
def test_cover_footprint_automatic_light_batch(
    benchmark, footprints: np.ndarray, resolution: int
) -> None:
    coverage = benchmark(px.cover_footprint, footprints[:300], resolution)

    assert coverage.offsets.shape == (301,)


def test_cover_footprint_explicit_pool_reuse(
    benchmark, large_footprints: np.ndarray
) -> None:
    coverage = benchmark(px.cover_footprint, large_footprints, 9, threads=4)

    assert coverage.offsets.shape == (large_footprints.shape[0] + 1,)


def test_cover_footprint_explicit_pool_varied_batches(
    benchmark,
    footprints: np.ndarray,
) -> None:
    def cover_two_sizes() -> px.Coverage:
        px.cover_footprint(footprints[:3], 6, threads=4)
        return px.cover_footprint(footprints[:64], 6, threads=4)

    coverage = benchmark(cover_two_sizes)

    assert coverage.offsets.shape == (65,)


def test_cover_footprint_mixed_ragged_batch(
    benchmark, large_footprints: np.ndarray
) -> None:
    midpoint = large_footprints[-1, 1] + large_footprints[-1, 2]
    midpoint /= np.linalg.norm(midpoint)
    pentagon = np.vstack((large_footprints[-1, :2], midpoint, large_footprints[-1, 2:]))
    ragged = tuple(large_footprints[:-1]) + (pentagon,)

    coverage = benchmark(px.cover_footprint, ragged, 9, threads=1)

    assert coverage.offsets.shape == (large_footprints.shape[0] + 1,)


def test_cover_footprint_with_large_sparse_candidate_set(
    benchmark,
    large_footprints: np.ndarray,
    large_sparse_resolution_12_cells: np.ndarray,
) -> None:
    resolution = 12

    coverage = benchmark(
        px.cover_footprint,
        large_footprints,
        resolution,
        candidate_cells=large_sparse_resolution_12_cells,
        threads=1,
    )

    assert coverage.offsets.shape == (large_footprints.shape[0] + 1,)


def test_cover_single_with_large_sparse_candidate_set(
    benchmark,
    large_footprints: np.ndarray,
    large_sparse_resolution_12_cells: np.ndarray,
) -> None:
    coverage = benchmark(
        px.cover_footprint,
        large_footprints[0],
        12,
        candidate_cells=large_sparse_resolution_12_cells,
        threads=1,
    )

    assert coverage.offsets.shape == (2,)


def test_cover_strip(benchmark, strip_edges: tuple[np.ndarray, np.ndarray]) -> None:
    left, right = strip_edges
    coverage = benchmark(px.cover_strip, left, right, 7, threads=1)

    assert coverage.offsets.shape == (left.shape[0],)
    assert coverage.cells.dtype == np.uint64


def test_cover_strip_automatic_parallel(
    benchmark,
    large_strip_edges: tuple[np.ndarray, np.ndarray],
) -> None:
    left, right = large_strip_edges
    coverage = benchmark(px.cover_strip, left, right, 9)

    assert coverage.offsets.shape == (left.shape[0],)


def test_cover_strip_with_sparse_high_resolution_candidates(
    benchmark,
    strip_edges: tuple[np.ndarray, np.ndarray],
    sparse_resolution_12_cells: np.ndarray,
) -> None:
    left, right = strip_edges
    coverage = benchmark(
        px.cover_strip,
        left,
        right,
        12,
        candidate_cells=sparse_resolution_12_cells,
        threads=1,
    )

    assert coverage.offsets.shape == (left.shape[0],)
    assert coverage.cells.dtype == np.uint64


def test_centers(benchmark, cells: np.ndarray) -> None:
    centers = benchmark(px.centers, cells, 7)

    assert centers.shape == (cells.size, 3)
    assert centers.dtype == np.float64


def test_boundaries(benchmark, cells: np.ndarray) -> None:
    boundaries = benchmark(px.boundaries, cells[:256], 7)

    assert boundaries.shape == (min(cells.size, 256), 4, 3)
    assert boundaries.dtype == np.float64
