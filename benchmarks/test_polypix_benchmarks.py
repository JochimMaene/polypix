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
def swath_edges() -> tuple[np.ndarray, np.ndarray]:
    latitudes = np.linspace(-40.0, 40.0, 501)
    left = np.asarray([_lonlat_to_xyz(-5.0, float(lat)) for lat in latitudes], dtype=np.float64)
    right = np.asarray([_lonlat_to_xyz(5.0, float(lat)) for lat in latitudes], dtype=np.float64)
    return left, right


@pytest.fixture(scope="module")
def cell_ids(footprints: np.ndarray) -> np.ndarray:
    return px.cover_footprint(footprints, resolution=7).cell_ids


@pytest.mark.parametrize("resolution", [4, 6, 7], ids=lambda value: f"resolution_{value}")
def test_cover_footprint_batch(benchmark, footprints: np.ndarray, resolution: int) -> None:
    coverage = benchmark(px.cover_footprint, footprints, resolution)

    assert coverage.offsets.shape == (footprints.shape[0] + 1,)
    assert coverage.cell_ids.dtype == np.uint64


def test_cover_swath(benchmark, swath_edges: tuple[np.ndarray, np.ndarray]) -> None:
    left, right = swath_edges
    coverage = benchmark(px.cover_swath, left, right, 7)

    assert coverage.offsets.shape == (left.shape[0],)
    assert coverage.cell_ids.dtype == np.uint64


def test_centers(benchmark, cell_ids: np.ndarray) -> None:
    centers = benchmark(px.centers, cell_ids)

    assert centers.shape == (cell_ids.size, 2)
    assert centers.dtype == np.float64


def test_boundaries(benchmark, cell_ids: np.ndarray) -> None:
    boundaries = benchmark(px.boundaries, cell_ids[:256])

    assert boundaries.shape == (min(cell_ids.size, 256), 4, 2)
    assert boundaries.dtype == np.float64
