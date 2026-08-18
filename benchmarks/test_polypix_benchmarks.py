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


def _regular_spherical_quad(
    axis: tuple[float, float, float], radius: float
) -> np.ndarray:
    center = np.asarray(axis, dtype=np.float64)
    center /= np.linalg.norm(center)
    seed = (
        np.asarray([1.0, 0.0, 0.0])
        if abs(center[2]) > 0.9
        else np.asarray([0.0, 0.0, 1.0])
    )
    tangent_x = np.cross(seed, center)
    tangent_x /= np.linalg.norm(tangent_x)
    tangent_y = np.cross(center, tangent_x)
    angles = np.arange(4) * (math.pi / 2.0)
    return math.cos(radius) * center + math.sin(radius) * (
        np.cos(angles)[:, np.newaxis] * tangent_x
        + np.sin(angles)[:, np.newaxis] * tangent_y
    )


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
def few_large_footprints() -> np.ndarray:
    axes = (
        _lonlat_to_xyz(float(longitude), float(latitude))
        for latitude in np.linspace(-50.0, 50.0, 8)
        for longitude in np.linspace(-160.0, 160.0, 8)
    )
    return np.asarray(
        [_regular_spherical_quad(axis, 0.3) for axis in axes],
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
def constellation_caps() -> tuple[np.ndarray, np.ndarray]:
    random = np.random.default_rng(20260811)
    centers = random.normal(size=(10_771, 3))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    radii = random.uniform(0.06, 0.155, size=centers.shape[0])
    return centers, radii


@pytest.fixture(scope="module")
def eo_shaped_coverage() -> px.Coverage:
    interval_count = 14_400
    cells_per_interval = 64
    cell_count = 12 * 4**6
    interval = np.arange(interval_count, dtype=np.uint64)[:, np.newaxis]
    within = np.arange(cells_per_interval, dtype=np.uint64)[np.newaxis, :]
    cells = ((131 * interval + 17 * within) % cell_count).reshape(-1)
    offsets = np.arange(
        0,
        cells.size + 1,
        cells_per_interval,
        dtype=np.uint64,
    )
    return px.Coverage.from_arrays(cells=cells, offsets=offsets, resolution=6)


@pytest.fixture(scope="module")
def many_sparse_sources() -> list[px.Coverage]:
    empty = px.Coverage.from_arrays(
        cells=np.empty(0, dtype=np.uint64),
        offsets=np.asarray([0, 0], dtype=np.uint64),
        resolution=8,
    )
    populated = px.Coverage.from_arrays(
        cells=np.asarray([123], dtype=np.uint64),
        offsets=np.asarray([0, 1], dtype=np.uint64),
        resolution=8,
    )
    return [empty] * 2_048 + [populated]


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


@pytest.fixture(scope="module")
def multi_million_sorted_resolution_12_cells() -> np.ndarray:
    resolution = 12
    pixel_count = 12 * 4**resolution
    candidate_count = 2_000_000
    return np.arange(candidate_count, dtype=np.uint64) * np.uint64(
        pixel_count // candidate_count
    )


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


def test_cover_footprint_single_automatic_latency(
    benchmark, footprints: np.ndarray
) -> None:
    coverage = benchmark(px.cover_footprint, footprints[0], 7)

    assert coverage.offsets.shape == (2,)


@pytest.mark.parametrize(
    "resolution", [6, 7, 9], ids=lambda value: f"resolution_{value}"
)
@pytest.mark.parallel
def test_cover_footprint_automatic_parallel(
    benchmark, large_footprints: np.ndarray, resolution: int
) -> None:
    coverage = benchmark(px.cover_footprint, large_footprints, resolution)

    assert coverage.offsets.shape == (large_footprints.shape[0] + 1,)


@pytest.mark.parametrize(
    "threads",
    [1, pytest.param(None, marks=pytest.mark.parallel)],
    ids=["serial", "automatic"],
)
def test_cover_few_large_footprints(
    benchmark,
    few_large_footprints: np.ndarray,
    threads: int | None,
) -> None:
    coverage = benchmark(
        px.cover_footprint,
        few_large_footprints,
        10,
        threads=threads,
    )

    assert coverage.offsets.shape == (few_large_footprints.shape[0] + 1,)


@pytest.mark.parametrize("resolution", [6, 12], ids=lambda value: f"resolution_{value}")
def test_cover_footprint_automatic_light_batch(
    benchmark, footprints: np.ndarray, resolution: int
) -> None:
    coverage = benchmark(px.cover_footprint, footprints[:300], resolution)

    assert coverage.offsets.shape == (301,)


@pytest.mark.parametrize("threads", [1, None], ids=["serial", "automatic"])
def test_cover_footprint_automatic_prepass_cost(
    benchmark,
    large_footprints: np.ndarray,
    threads: int | None,
) -> None:
    footprints = large_footprints[:2_000]
    coverage = benchmark(px.cover_footprint, footprints, 2, threads=threads)

    assert coverage.offsets.shape == (2_001,)


@pytest.mark.parallel
def test_cover_footprint_explicit_pool_reuse(
    benchmark, large_footprints: np.ndarray
) -> None:
    coverage = benchmark(px.cover_footprint, large_footprints, 9, threads=4)

    assert coverage.offsets.shape == (large_footprints.shape[0] + 1,)


def test_cover_footprint_dense_nested_list(
    benchmark, large_footprints: np.ndarray
) -> None:
    nested = large_footprints.tolist()
    coverage = benchmark(px.cover_footprint, nested, 7, threads=1)

    assert coverage.offsets.shape == (large_footprints.shape[0] + 1,)


def test_cover_footprint_mixed_ragged_batch(
    benchmark, large_footprints: np.ndarray
) -> None:
    midpoint = large_footprints[-1, 1] + large_footprints[-1, 2]
    midpoint /= np.linalg.norm(midpoint)
    pentagon = np.vstack((large_footprints[-1, :2], midpoint, large_footprints[-1, 2:]))
    ragged = tuple(large_footprints[:-1]) + (pentagon,)

    coverage = benchmark(px.cover_footprint, ragged, 9, threads=1)

    assert coverage.offsets.shape == (large_footprints.shape[0] + 1,)


@pytest.mark.parallel
def test_cover_footprint_mixed_ragged_batch_automatic(
    benchmark, large_footprints: np.ndarray
) -> None:
    midpoint = large_footprints[-1, 1] + large_footprints[-1, 2]
    midpoint /= np.linalg.norm(midpoint)
    pentagon = np.vstack((large_footprints[-1, :2], midpoint, large_footprints[-1, 2:]))
    ragged = tuple(large_footprints[:-1]) + (pentagon,)

    coverage = benchmark(px.cover_footprint, ragged, 9)

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


@pytest.mark.parametrize(
    "threads",
    [1, pytest.param(None, marks=pytest.mark.parallel)],
    ids=["serial", "automatic"],
)
def test_cover_footprint_with_candidates_parallel_scaling(
    benchmark,
    large_footprints: np.ndarray,
    multi_million_sorted_resolution_12_cells: np.ndarray,
    threads: int | None,
) -> None:
    coverage = benchmark(
        px.cover_footprint,
        large_footprints,
        12,
        candidate_cells=multi_million_sorted_resolution_12_cells,
        threads=threads,
    )

    assert coverage.offsets.shape == (large_footprints.shape[0] + 1,)


@pytest.mark.parametrize(
    "threads",
    [1, pytest.param(None, marks=pytest.mark.parallel)],
    ids=["serial", "automatic"],
)
def test_cover_footprint_with_small_sparse_candidate_set(
    benchmark,
    large_footprints: np.ndarray,
    sparse_resolution_12_cells: np.ndarray,
    threads: int | None,
) -> None:
    coverage = benchmark(
        px.cover_footprint,
        large_footprints,
        12,
        candidate_cells=sparse_resolution_12_cells,
        threads=threads,
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


def test_cover_with_multi_million_sorted_candidates(
    benchmark,
    large_footprints: np.ndarray,
    multi_million_sorted_resolution_12_cells: np.ndarray,
) -> None:
    coverage = benchmark(
        px.cover_footprint,
        large_footprints[:64],
        12,
        candidate_cells=multi_million_sorted_resolution_12_cells,
        threads=1,
    )

    assert coverage.offsets.shape == (65,)


def test_cover_sweep(benchmark, strip_edges: tuple[np.ndarray, np.ndarray]) -> None:
    left, right = strip_edges
    coverage = benchmark(px.cover_sweep, left, right, 7, threads=1)

    assert coverage.offsets.shape == (left.shape[0],)
    assert coverage.cells.dtype == np.uint64


@pytest.mark.parallel
def test_cover_sweep_automatic_parallel(
    benchmark,
    large_strip_edges: tuple[np.ndarray, np.ndarray],
) -> None:
    left, right = large_strip_edges
    coverage = benchmark(px.cover_sweep, left, right, 9)

    assert coverage.offsets.shape == (left.shape[0],)


def test_cover_sweep_with_sparse_high_resolution_candidates(
    benchmark,
    strip_edges: tuple[np.ndarray, np.ndarray],
    sparse_resolution_12_cells: np.ndarray,
) -> None:
    left, right = strip_edges
    coverage = benchmark(
        px.cover_sweep,
        left,
        right,
        12,
        candidate_cells=sparse_resolution_12_cells,
        threads=1,
    )

    assert coverage.offsets.shape == (left.shape[0],)
    assert coverage.cells.dtype == np.uint64


@pytest.mark.parametrize(
    "threads",
    [1, pytest.param(None, marks=pytest.mark.parallel)],
    ids=["serial", "automatic"],
)
def test_cover_cap_constellation_batch(
    benchmark,
    constellation_caps: tuple[np.ndarray, np.ndarray],
    threads: int | None,
) -> None:
    centers, radii = constellation_caps
    coverage = benchmark(px.cover_cap, centers, radii, 6, threads=threads)

    assert coverage.offsets.shape == (centers.shape[0] + 1,)
    assert coverage.cells.dtype == np.uint64
    assert coverage.cells.size == 1_629_277


@pytest.mark.parametrize(
    "threads",
    [1, pytest.param(None, marks=pytest.mark.parallel)],
    ids=["serial", "automatic"],
)
def test_count_caps_per_cell_constellation_batch(
    benchmark,
    constellation_caps: tuple[np.ndarray, np.ndarray],
    threads: int | None,
) -> None:
    centers, radii = constellation_caps
    counts = benchmark(px.count_caps_per_cell, centers, radii, 6, threads=threads)

    assert counts.shape == (12 * 4**6,)
    assert counts.dtype == np.int64
    assert int(counts.sum()) == 1_629_277


def test_coverage_from_arrays_eo_shape(
    benchmark,
    eo_shaped_coverage: px.Coverage,
) -> None:
    imported = benchmark(
        px.Coverage.from_arrays,
        eo_shaped_coverage.cells,
        eo_shaped_coverage.offsets,
        eo_shaped_coverage.resolution,
    )

    assert imported.cells.size == 921_600
    assert imported.segment_count == 14_400
    assert not np.shares_memory(imported.cells, eo_shaped_coverage.cells)
    assert not imported.cells.flags.writeable


def test_summarize_occupancy_eo_shape(
    benchmark,
    eo_shaped_coverage: px.Coverage,
) -> None:
    summary = benchmark(px.summarize_occupancy, [eo_shaped_coverage] * 10)

    assert summary.segment_count == 14_400
    assert summary.cells.dtype == np.uint64
    assert summary.run_counts.dtype == np.uint64
    assert summary.cells.size == 49_152
    assert int(summary.run_counts.sum()) == 9_216_000
    assert int(summary.merged_gap_steps_sum.sum()) == 641_429_424
    assert int(summary.merged_gap_counts.sum()) == 872_448


def test_summarize_occupancy_many_sparse_sources(
    benchmark,
    many_sparse_sources: list[px.Coverage],
) -> None:
    summary = benchmark(px.summarize_occupancy, many_sparse_sources)

    np.testing.assert_array_equal(summary.cells, [123])
    np.testing.assert_array_equal(summary.run_counts, [1])


def test_centers(benchmark, cells: np.ndarray) -> None:
    centers = benchmark(px.centers, cells, 7)

    assert centers.shape == (cells.size, 3)
    assert centers.dtype == np.float64


def test_corners(benchmark, cells: np.ndarray) -> None:
    corners = benchmark(px.corners, cells[:256], 7)

    assert corners.shape == (min(cells.size, 256), 4, 3)
    assert corners.dtype == np.float64


@pytest.mark.parametrize(
    "count",
    [1_000, pytest.param(1_000_000, marks=pytest.mark.parallel)],
    ids=["small", "large"],
)
def test_centers_transform_scaling(benchmark, count: int) -> None:
    cells = np.arange(count, dtype=np.uint64)
    centers = benchmark(px.centers, cells, 12)

    assert centers.shape == (count, 3)


@pytest.mark.parametrize(
    "count",
    [1_000, pytest.param(1_000_000, marks=pytest.mark.parallel)],
    ids=["small", "large"],
)
def test_cell_at_transform_scaling(benchmark, count: int) -> None:
    cells = np.arange(count, dtype=np.uint64)
    vectors = px.centers(cells, 12)
    actual = benchmark(px.cell_at, vectors, 12)

    np.testing.assert_array_equal(actual, cells)


@pytest.mark.parametrize(
    "count",
    [1_000, pytest.param(1_000_000, marks=pytest.mark.parallel)],
    ids=["small", "large"],
)
def test_corners_transform_scaling(benchmark, count: int) -> None:
    cells = np.arange(count, dtype=np.uint64)
    corners = benchmark(px.corners, cells, 12)

    assert corners.shape == (count, 4, 3)
