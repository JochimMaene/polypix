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


def _diagonal_edges(samples: int) -> tuple[np.ndarray, np.ndarray]:
    """A 4-degree-wide swath crossing longitude as fast as latitude."""
    latitudes = np.linspace(-60.0, 60.0, samples)
    longitudes = np.linspace(-120.0, 120.0, samples)
    return tuple(
        np.asarray(
            [
                _lonlat_to_xyz(float(lon) + across, float(lat))
                for lon, lat in zip(longitudes, latitudes, strict=True)
            ],
            dtype=np.float64,
        )
        for across in (-2.0, 2.0)
    )


@pytest.fixture(scope="module")
def coarse_diagonal_strip_edges() -> tuple[np.ndarray, np.ndarray]:
    """A diagonal swath sampled coarsely enough that segments span longitude.

    Every other strip fixture here runs along a meridian, where a segment spans
    almost no longitude and the per-footprint longitude envelope is already
    tight. The envelope only becomes loose when one footprint covers a wide
    longitude range, which needs a diagonal track *and* coarse sampling: at 20
    samples this costs about 2.7x the meridian's time per emitted cell, at 5
    samples about 9x, and by 100 samples the penalty is gone. This is the
    fixture that moves if the scan stops testing every centre in the envelope.
    """
    return _diagonal_edges(20)


@pytest.fixture(scope="module")
def diagonal_strip_edges() -> tuple[np.ndarray, np.ndarray]:
    """The same track sampled densely: a control that must not regress.

    Segments here are small enough that their envelopes are already tight, so
    this pays no envelope penalty today and must not start paying one.
    """
    return _diagonal_edges(501)


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
def sparse_high_resolution_reduction(
    eo_shaped_coverage: px.Coverage,
) -> tuple[px.Coverage, np.ndarray]:
    coverage = px.Coverage.from_arrays(
        eo_shaped_coverage.cells,
        eo_shaped_coverage.offsets,
        resolution=29,
    )
    queried = np.concatenate(
        (
            np.arange(12 * 4**6, dtype=np.int64),
            1_000_000_000_000 + np.arange(16_384, dtype=np.int64),
        )
    )
    return coverage, queried


@pytest.fixture(scope="module")
def small_resolution_8_coverage() -> px.Coverage:
    """A handful of hits on a grid large enough for a dense scratch array."""
    centers = np.asarray(
        [_lonlat_to_xyz(lon, 12.0) for lon in (0.0, 90.0, 180.0, 270.0)],
        dtype=np.float64,
    )
    return px.cover_cap(centers, 0.01, 8)


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
    return px.cover_convex_polygon(footprints, resolution=7, threads=1).cells


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
def test_cover_convex_polygon_batch(
    benchmark, footprints: np.ndarray, resolution: int
) -> None:
    coverage = benchmark(px.cover_convex_polygon, footprints, resolution, threads=1)

    assert coverage.offsets.shape == (footprints.shape[0] + 1,)
    assert coverage.cells.dtype == np.int64


def test_cover_convex_polygon_single_latency(benchmark, footprints: np.ndarray) -> None:
    coverage = benchmark(px.cover_convex_polygon, footprints[0], 7, threads=1)

    assert coverage.offsets.shape == (2,)


def test_cover_convex_polygon_single_automatic_latency(
    benchmark, footprints: np.ndarray
) -> None:
    coverage = benchmark(px.cover_convex_polygon, footprints[0], 7)

    assert coverage.offsets.shape == (2,)


@pytest.mark.parametrize(
    "resolution", [6, 7, 9], ids=lambda value: f"resolution_{value}"
)
@pytest.mark.parallel
def test_cover_convex_polygon_automatic_parallel(
    benchmark, large_footprints: np.ndarray, resolution: int
) -> None:
    coverage = benchmark(px.cover_convex_polygon, large_footprints, resolution)

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
        px.cover_convex_polygon,
        few_large_footprints,
        10,
        threads=threads,
    )

    assert coverage.offsets.shape == (few_large_footprints.shape[0] + 1,)


@pytest.mark.parametrize("resolution", [6, 12], ids=lambda value: f"resolution_{value}")
def test_cover_convex_polygon_automatic_light_batch(
    benchmark, footprints: np.ndarray, resolution: int
) -> None:
    coverage = benchmark(px.cover_convex_polygon, footprints[:300], resolution)

    assert coverage.offsets.shape == (301,)


@pytest.mark.parametrize("threads", [1, None], ids=["serial", "automatic"])
def test_cover_convex_polygon_automatic_prepass_cost(
    benchmark,
    large_footprints: np.ndarray,
    threads: int | None,
) -> None:
    footprints = large_footprints[:2_000]
    coverage = benchmark(px.cover_convex_polygon, footprints, 2, threads=threads)

    assert coverage.offsets.shape == (2_001,)


@pytest.mark.parallel
def test_cover_convex_polygon_explicit_pool_reuse(
    benchmark, large_footprints: np.ndarray
) -> None:
    coverage = benchmark(px.cover_convex_polygon, large_footprints, 9, threads=4)

    assert coverage.offsets.shape == (large_footprints.shape[0] + 1,)


def test_cover_convex_polygon_dense_nested_list(
    benchmark, large_footprints: np.ndarray
) -> None:
    nested = large_footprints.tolist()
    coverage = benchmark(px.cover_convex_polygon, nested, 7, threads=1)

    assert coverage.offsets.shape == (large_footprints.shape[0] + 1,)


def test_cover_convex_polygon_mixed_ragged_batch(
    benchmark, large_footprints: np.ndarray
) -> None:
    midpoint = large_footprints[-1, 1] + large_footprints[-1, 2]
    midpoint /= np.linalg.norm(midpoint)
    pentagon = np.vstack((large_footprints[-1, :2], midpoint, large_footprints[-1, 2:]))
    ragged = tuple(large_footprints[:-1]) + (pentagon,)

    coverage = benchmark(px.cover_convex_polygon, ragged, 9, threads=1)

    assert coverage.offsets.shape == (large_footprints.shape[0] + 1,)


@pytest.mark.parallel
def test_cover_convex_polygon_mixed_ragged_batch_automatic(
    benchmark, large_footprints: np.ndarray
) -> None:
    midpoint = large_footprints[-1, 1] + large_footprints[-1, 2]
    midpoint /= np.linalg.norm(midpoint)
    pentagon = np.vstack((large_footprints[-1, :2], midpoint, large_footprints[-1, 2:]))
    ragged = tuple(large_footprints[:-1]) + (pentagon,)

    coverage = benchmark(px.cover_convex_polygon, ragged, 9)

    assert coverage.offsets.shape == (large_footprints.shape[0] + 1,)


@pytest.fixture(scope="module")
def packed_ragged_polygons() -> tuple[np.ndarray, np.ndarray]:
    """A columnar ragged batch: flat vertices plus offsets, mixed 3/4/5-gons.

    This is GeoArrow's polygon encoding, and what Arrow, Parquet, and database
    geometry columns hand over, so it is the shape `vertex_offsets=` exists for.
    """
    random = np.random.default_rng(20260821)
    count = 200_000
    vertex_counts = random.choice([3, 4, 5], size=count)
    longitudes = random.uniform(-170.0, 170.0, count)
    latitudes = random.uniform(-60.0, 60.0, count)
    rows = []
    for vertices, longitude, latitude in zip(
        vertex_counts, longitudes, latitudes, strict=True
    ):
        angles = np.linspace(0.0, 2.0 * np.pi, vertices, endpoint=False)
        rows.append(
            np.asarray(
                [
                    _lonlat_to_xyz(
                        float(longitude + 0.6 * np.cos(angle)),
                        float(latitude + 0.6 * np.sin(angle)),
                    )
                    for angle in angles
                ]
            )
        )
    packed = np.ascontiguousarray(np.concatenate(rows))
    offsets = np.concatenate(([0], np.cumsum(vertex_counts))).astype(np.int64)
    return packed, offsets


def test_cover_convex_polygon_packed_ragged_batch(
    benchmark, packed_ragged_polygons: tuple[np.ndarray, np.ndarray]
) -> None:
    """Columnar input must reach the kernel without being taken apart.

    The alternative a caller has without `vertex_offsets=` is splitting the
    buffer into one array per polygon, which was measured at 2.0x this call and
    62 MiB of peak for the concatenate that puts it back together.
    """
    packed, offsets = packed_ragged_polygons
    coverage = benchmark(
        px.cover_convex_polygon, packed, 7, vertex_offsets=offsets, threads=1
    )

    assert coverage.offsets.shape == (offsets.size,)


def test_cover_convex_polygon_split_packed_batch(
    benchmark, packed_ragged_polygons: tuple[np.ndarray, np.ndarray]
) -> None:
    """The same geometry taken apart into a sequence, for comparison."""
    packed, offsets = packed_ragged_polygons
    coverage = benchmark(
        lambda: px.cover_convex_polygon(
            [packed[offsets[i] : offsets[i + 1]] for i in range(offsets.size - 1)],
            7,
            threads=1,
        )
    )

    assert coverage.offsets.shape == (offsets.size,)


def test_cover_convex_polygon_with_large_sparse_candidate_set(
    benchmark,
    large_footprints: np.ndarray,
    large_sparse_resolution_12_cells: np.ndarray,
) -> None:
    resolution = 12

    coverage = benchmark(
        px.cover_convex_polygon,
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
def test_cover_convex_polygon_with_candidates_parallel_scaling(
    benchmark,
    large_footprints: np.ndarray,
    multi_million_sorted_resolution_12_cells: np.ndarray,
    threads: int | None,
) -> None:
    coverage = benchmark(
        px.cover_convex_polygon,
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
def test_cover_convex_polygon_with_small_sparse_candidate_set(
    benchmark,
    large_footprints: np.ndarray,
    sparse_resolution_12_cells: np.ndarray,
    threads: int | None,
) -> None:
    coverage = benchmark(
        px.cover_convex_polygon,
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
        px.cover_convex_polygon,
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
        px.cover_convex_polygon,
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
    assert coverage.cells.dtype == np.int64


@pytest.mark.parallel
def test_cover_sweep_automatic_parallel(
    benchmark,
    large_strip_edges: tuple[np.ndarray, np.ndarray],
) -> None:
    left, right = large_strip_edges
    coverage = benchmark(px.cover_sweep, left, right, 9)

    assert coverage.offsets.shape == (left.shape[0],)


def test_cover_sweep_coarse_diagonal(
    benchmark, coarse_diagonal_strip_edges: tuple[np.ndarray, np.ndarray]
) -> None:
    left, right = coarse_diagonal_strip_edges
    coverage = benchmark(px.cover_sweep, left, right, 9, threads=1)

    assert coverage.offsets.shape == (left.shape[0],)


def test_cover_sweep_coarse_diagonal_count(
    benchmark, coarse_diagonal_strip_edges: tuple[np.ndarray, np.ndarray]
) -> None:
    left, right = coarse_diagonal_strip_edges
    counts = benchmark(px.cover_sweep, left, right, 9, threads=1, reduce=px.Count())

    assert counts.shape == (px.cell_count(9),)


def test_cover_sweep_dense_diagonal(
    benchmark, diagonal_strip_edges: tuple[np.ndarray, np.ndarray]
) -> None:
    left, right = diagonal_strip_edges
    coverage = benchmark(px.cover_sweep, left, right, 9, threads=1)

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
    assert coverage.cells.dtype == np.int64


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
    assert coverage.cells.dtype == np.int64
    assert coverage.cells.size == 1_629_277


@pytest.mark.parametrize(
    "threads",
    [1, pytest.param(None, marks=pytest.mark.parallel)],
    ids=["serial", "automatic"],
)
def test_cover_cap_dense_count_constellation_batch(
    benchmark,
    constellation_caps: tuple[np.ndarray, np.ndarray],
    threads: int | None,
) -> None:
    centers, radii = constellation_caps
    counts = benchmark(
        px.cover_cap, centers, radii, 6, reduce=px.Count(), threads=threads
    )

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
    assert len(imported) == 14_400
    assert not np.shares_memory(imported.cells, eo_shaped_coverage.cells)
    assert not imported.cells.flags.writeable


def test_count_coverage_per_cell_eo_shape(
    benchmark,
    eo_shaped_coverage: px.Coverage,
) -> None:
    counts = benchmark(eo_shaped_coverage.reduce, px.Count())

    assert counts.shape == (12 * 4**6,)
    assert counts.dtype == np.int64
    assert int(counts.sum()) == 921_600


def test_sum_coverage_per_cell_eo_shape(
    benchmark,
    eo_shaped_coverage: px.Coverage,
) -> None:
    values = np.linspace(0.25, 1.25, len(eo_shaped_coverage))
    sums = benchmark(eo_shaped_coverage.reduce, px.Sum(values))

    assert sums.shape == (12 * 4**6,)
    assert sums.dtype == np.float64
    assert np.isclose(sums.sum(), 64 * values.sum())


def test_cover_cap_selected_count_small_request(
    benchmark,
    constellation_caps: tuple[np.ndarray, np.ndarray],
) -> None:
    """A small request must stay on the fused per-cell cap kernel."""
    centers, radii = constellation_caps
    requested = np.arange(64, dtype=np.int64)
    counts = benchmark(
        px.cover_cap,
        centers,
        radii,
        6,
        candidate_cells=requested,
        reduce=px.Count(),
        threads=1,
    )

    assert counts.shape == requested.shape
    np.testing.assert_array_equal(
        counts,
        px.cover_cap(centers, radii, 6).reduce(px.Count(), cells=requested),
    )


def test_cover_polygon_selected_count_small_request(
    benchmark, footprints: np.ndarray
) -> None:
    """A small request must be answered by testing those cells, not by scanning.

    Covering these footprints at resolution 9 and gathering 1000 cells out of
    the result was measured at 220x the cost of handing the request to the
    kernel as its candidate set, and it peaked at the full cell list besides.
    """
    requested = np.arange(1000, dtype=np.int64) * 3001
    counts = benchmark(
        px.cover_convex_polygon,
        footprints,
        9,
        candidate_cells=requested,
        reduce=px.Count(),
        threads=1,
    )

    assert counts.shape == requested.shape
    np.testing.assert_array_equal(
        counts,
        px.cover_convex_polygon(footprints, 9).reduce(px.Count(), cells=requested),
    )


def test_cover_polygon_selected_count_large_request(
    benchmark, footprints: np.ndarray
) -> None:
    """A large request must keep scanning once and gathering.

    Testing every requested cell against every footprint loses by up to 50x
    once the request stops being a small share of the grid, so the substitution
    has to decline here.
    """
    requested = np.arange(200_000, dtype=np.int64)
    counts = benchmark(
        px.cover_convex_polygon,
        footprints,
        9,
        candidate_cells=requested,
        reduce=px.Count(),
        threads=1,
    )

    assert counts.shape == requested.shape
    np.testing.assert_array_equal(
        counts,
        px.cover_convex_polygon(footprints, 9).reduce(px.Count(), cells=requested),
    )


def test_cover_sweep_selected_sum_small_request(
    benchmark, strip_edges: tuple[np.ndarray, np.ndarray]
) -> None:
    """Sums follow counts: a small request is a question about those cells."""
    left, right = strip_edges
    requested = np.arange(1000, dtype=np.int64) * 3001
    values = np.linspace(0.5, 1.5, left.shape[0] - 1)
    sums = benchmark(
        px.cover_sweep,
        left,
        right,
        9,
        candidate_cells=requested,
        reduce=px.Sum(values),
        threads=1,
    )

    assert sums.shape == requested.shape
    np.testing.assert_array_equal(
        sums,
        px.cover_sweep(left, right, 9).reduce(px.Sum(values), cells=requested),
    )


def test_cover_cap_selected_count_large_request(
    benchmark,
    constellation_caps: tuple[np.ndarray, np.ndarray],
) -> None:
    """A large request must not degrade to one cap test per requested cell.

    Fusing this shape costs ``cells * caps`` cap tests and was measured at 47x
    the cost of covering once and reducing, so the reducer has to decline.
    """
    centers, radii = constellation_caps
    requested = np.arange(100_000, dtype=np.int64)
    counts = benchmark(
        px.cover_cap,
        centers,
        radii,
        8,
        candidate_cells=requested,
        reduce=px.Count(),
        threads=1,
    )

    assert counts.shape == requested.shape
    np.testing.assert_array_equal(
        counts,
        px.cover_cap(centers, radii, 8).reduce(px.Count(), cells=requested),
    )


def test_count_coverage_selected_small_work(
    benchmark,
    small_resolution_8_coverage: px.Coverage,
) -> None:
    """A few hits and a tiny query must not zero the whole 6 MiB scratch grid.

    The dense scratch grid fits at resolution 8, but zeroing it for a handful
    of probes was measured at 18x the cost of the hash path.
    """
    requested = np.asarray(small_resolution_8_coverage.cells[:4])
    counts = benchmark(small_resolution_8_coverage.reduce, px.Count(), cells=requested)

    assert counts.shape == requested.shape
    assert int(counts.sum()) >= 4


def test_sum_coverage_selected_sparse_high_resolution(
    benchmark,
    sparse_high_resolution_reduction: tuple[px.Coverage, np.ndarray],
) -> None:
    coverage, queried = sparse_high_resolution_reduction
    values = np.linspace(0.25, 1.25, len(coverage))
    sums = benchmark(coverage.reduce, px.Sum(values), cells=queried)

    assert sums.shape == queried.shape
    assert sums.dtype == np.float64


def test_count_coverage_selected_sparse_high_resolution(
    benchmark,
    sparse_high_resolution_reduction: tuple[px.Coverage, np.ndarray],
) -> None:
    coverage, queried = sparse_high_resolution_reduction
    counts = benchmark(coverage.reduce, px.Count(), cells=queried)

    assert counts.shape == queried.shape
    assert counts.dtype == np.int64
    assert int(counts.sum()) == coverage.cells.size


def test_occupancy_many_sources_eo_shape(
    benchmark,
    eo_shaped_coverage: px.Coverage,
) -> None:
    """The statistics pass must not build the runs it summarizes.

    This shape is why: 9.28 million hits produce 921,600 runs, almost one
    boundary pair per represented cell per source.
    """
    stats = benchmark(px.occupancy, [eo_shaped_coverage] * 10)

    assert stats.cells.dtype == np.int64
    assert stats.run_counts.dtype == np.int64
    assert stats.cells.size == 49_152
    assert int(stats.run_counts.sum()) == 921_600


def test_occupancy_output_heavy_eo_shape(
    benchmark,
    eo_shaped_coverage: px.Coverage,
) -> None:
    stats = benchmark(px.occupancy, eo_shaped_coverage)

    assert stats.cells.size == 49_152
    assert int(stats.run_counts.sum()) == 921_600


def test_occupancy_sparse_high_resolution(
    benchmark,
    many_sparse_sources: list[px.Coverage],
) -> None:
    stats = benchmark(px.occupancy, many_sparse_sources)

    np.testing.assert_array_equal(stats.cells, [123])
    np.testing.assert_array_equal(stats.run_counts, [1])
    np.testing.assert_array_equal(stats.first_start, [0])
    np.testing.assert_array_equal(stats.last_stop, [1])


def test_centers(benchmark, cells: np.ndarray) -> None:
    centers = benchmark(px.cell_centers, cells, 7)

    assert centers.shape == (cells.size, 3)
    assert centers.dtype == np.float64


def test_corners(benchmark, cells: np.ndarray) -> None:
    corners = benchmark(px.cell_corners, cells[:256], 7)

    assert corners.shape == (min(cells.size, 256), 4, 3)
    assert corners.dtype == np.float64


@pytest.mark.parametrize(
    "count",
    [1_000, pytest.param(1_000_000, marks=pytest.mark.parallel)],
    ids=["small", "large"],
)
def test_centers_transform_scaling(benchmark, count: int) -> None:
    cells = np.arange(count, dtype=np.uint64)
    centers = benchmark(px.cell_centers, cells, 12)

    assert centers.shape == (count, 3)


@pytest.mark.parametrize(
    "count",
    [1_000, pytest.param(1_000_000, marks=pytest.mark.parallel)],
    ids=["small", "large"],
)
def test_cell_at_transform_scaling(benchmark, count: int) -> None:
    cells = np.arange(count, dtype=np.uint64)
    vectors = px.cell_centers(cells, 12)
    actual = benchmark(px.cell_at, vectors, 12)

    np.testing.assert_array_equal(actual, cells)


@pytest.mark.parametrize(
    "count",
    [1_000, pytest.param(1_000_000, marks=pytest.mark.parallel)],
    ids=["small", "large"],
)
def test_corners_transform_scaling(benchmark, count: int) -> None:
    cells = np.arange(count, dtype=np.uint64)
    corners = benchmark(px.cell_corners, cells, 12)

    assert corners.shape == (count, 4, 3)
