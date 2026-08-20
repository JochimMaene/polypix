"""Differential tests: the kernel against independent brute-force oracles.

These pin the kernel's *output*, not its method, so a rewrite of how cells are
found has something to be judged against. Every oracle here evaluates the
authoritative containment predicate on every cell of the grid, which is only
affordable at low resolutions - that is the point: exhaustive at small nside
beats sampled at large nside for catching a missed cell.

The oracles are deliberately independent of the covering kernel. They reach the
grid through :func:`polypix.cell_centers`, a separate code path, and apply the
predicate in NumPy.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

import polypix as px

# The kernel's containment tolerance, from rust/geometry.rs.
CONTAINMENT_EPSILON = 1.0e-14
# A cell whose predicate margin sits this close to the threshold is decided by
# the last bits of two different evaluation orders, so the oracle may disagree
# with the kernel about it and neither is wrong. Random inputs produce none;
# the tests assert that any disagreement is confined to this band.
AMBIGUOUS_MARGIN = 1.0e-12


def _unit(vectors: np.ndarray) -> np.ndarray:
    """Normalize like ``geometry::normalize``: scale down, then divide."""
    vectors = np.asarray(vectors, dtype=np.float64)
    scale = np.max(np.abs(vectors), axis=-1, keepdims=True)
    scaled = vectors / scale
    return scaled / np.sqrt(np.sum(scaled * scaled, axis=-1, keepdims=True))


def _edge_normals(vertices: np.ndarray) -> np.ndarray:
    """Outward-consistent unit edge normals, oriented as the kernel orients."""
    vertices = _unit(vertices)

    def normals(vertices: np.ndarray) -> np.ndarray:
        raw = np.cross(vertices, np.roll(vertices, -1, axis=0))
        length = np.hypot(np.hypot(raw[:, 0], raw[:, 1]), raw[:, 2])
        return raw / length[:, None]

    interior = vertices.sum(axis=0)
    interior /= np.hypot(np.hypot(interior[0], interior[1]), interior[2])
    if float((normals(vertices) @ interior).sum()) < 0.0:
        vertices = vertices[::-1]
    return normals(vertices)


@functools.cache
def _grid_centers(resolution: int) -> np.ndarray:
    """Every cell center of the grid, from a code path the kernel does not use."""
    return px.cell_centers(np.arange(px.cell_count(resolution)), resolution)


def _polygon_margins(vertices: np.ndarray, resolution: int) -> np.ndarray:
    """Per-cell containment margin for every cell of the grid."""
    return (_grid_centers(resolution) @ _edge_normals(vertices).T).min(axis=1)


def _cap_margins(center: np.ndarray, radius: float, resolution: int) -> np.ndarray:
    """Squared-chord slack for every cell, positive inside, as the kernel tests."""
    centers = _grid_centers(resolution)
    axis = _unit(np.asarray(center, dtype=np.float64))
    effective = min(radius + CONTAINMENT_EPSILON, np.pi)
    squared_chord = 4.0 * np.sin(0.5 * effective) ** 2
    return squared_chord - np.sum((centers - axis) ** 2, axis=1)


def _assert_matches(
    hits: np.ndarray, margins: np.ndarray, threshold: float, label: str
) -> None:
    """Compare a kernel hit list against an oracle margin, band-aware."""
    expected = np.flatnonzero(margins >= threshold)
    actual = np.sort(np.asarray(hits, dtype=np.int64))
    assert actual.size == np.unique(actual).size, f"{label}: duplicate cells"
    disagreed = np.setxor1d(expected, actual)
    ambiguous = np.abs(margins[disagreed] - threshold) < AMBIGUOUS_MARGIN
    assert ambiguous.all(), (
        f"{label}: cells {disagreed[~ambiguous].tolist()} disagree outside the "
        f"ambiguous band, margins {margins[disagreed[~ambiguous]].tolist()}"
    )


def _convex_polygon(
    rng: np.random.Generator, sides: int, radius_rad: float
) -> np.ndarray:
    """A convex spherical polygon inscribed in a cap, at a random attitude."""
    axis = _unit(rng.normal(size=3))
    east = _unit(np.cross(axis, _unit(rng.normal(size=3))))
    north = np.cross(axis, east)
    azimuths = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=sides))
    while np.any(np.diff(np.append(azimuths, azimuths[0] + 2 * np.pi)) < 0.3):
        azimuths = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=sides))
    offsets = np.cos(azimuths)[:, None] * east + np.sin(azimuths)[:, None] * north
    return _unit(np.cos(radius_rad) * axis + np.sin(radius_rad) * offsets)


def _lonlat(lon_deg: float, lat_deg: float) -> list[float]:
    lon, lat = np.radians(lon_deg), np.radians(lat_deg)
    return [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]


def _named_footprints() -> dict[str, np.ndarray]:
    """Footprints whose geometry stresses the ring and longitude bounds."""
    return {
        "north pole": np.asarray(
            [
                _lonlat(0.0, 80.0),
                _lonlat(90.0, 80.0),
                _lonlat(180.0, 80.0),
                _lonlat(270.0, 80.0),
            ]
        ),
        "south pole": np.asarray(
            [
                _lonlat(0.0, -80.0),
                _lonlat(270.0, -80.0),
                _lonlat(180.0, -80.0),
                _lonlat(90.0, -80.0),
            ]
        ),
        "antimeridian": np.asarray(
            [
                _lonlat(175.0, -5.0),
                _lonlat(-175.0, -5.0),
                _lonlat(-175.0, 5.0),
                _lonlat(175.0, 5.0),
            ]
        ),
        "thin diagonal": np.asarray(
            [
                _lonlat(-40.4, -40.0),
                _lonlat(39.6, 40.0),
                _lonlat(40.4, 40.0),
                _lonlat(-39.6, -40.0),
            ]
        ),
        "thin meridian": np.asarray(
            [
                _lonlat(-0.4, -40.0),
                _lonlat(0.4, -40.0),
                _lonlat(0.4, 40.0),
                _lonlat(-0.4, 40.0),
            ]
        ),
        "sub-cell": np.asarray(
            [
                _lonlat(10.0, 10.0),
                _lonlat(10.001, 10.0),
                _lonlat(10.001, 10.001),
                _lonlat(10.0, 10.001),
            ]
        ),
        "equator crossing": np.asarray(
            [
                _lonlat(-30.0, -1.0),
                _lonlat(30.0, -1.0),
                _lonlat(30.0, 1.0),
                _lonlat(-30.0, 1.0),
            ]
        ),
        "hemisphere octagon": _convex_polygon(
            np.random.default_rng(11), sides=8, radius_rad=1.4
        ),
    }


@pytest.mark.parametrize("resolution", [0, 1, 2, 3, 4])
def test_named_footprint_coverage_matches_brute_force_centers(
    resolution: int, subtests
) -> None:
    """Every cell the kernel emits, and only those, passes the predicate."""
    for name, vertices in _named_footprints().items():
        with subtests.test(footprint=name):
            coverage = px.cover_convex_polygon(vertices[None], resolution)
            _assert_matches(
                coverage.cells,
                _polygon_margins(vertices, resolution),
                -CONTAINMENT_EPSILON,
                f"{name} at resolution {resolution}",
            )


@pytest.mark.parametrize("sides", [3, 4, 5, 8])
@pytest.mark.parametrize("resolution", [1, 3, 5])
def test_random_polygon_coverage_matches_brute_force_centers(
    sides: int, resolution: int, subtests
) -> None:
    """Quads take the 4-half-space path; other counts take the polygon path."""
    rng = np.random.default_rng(20260820 + 100 * sides + resolution)
    for trial in range(12):
        radius = float(rng.uniform(0.01, 1.2))
        vertices = _convex_polygon(rng, sides, radius)
        with subtests.test(trial=trial, radius=radius):
            coverage = px.cover_convex_polygon(vertices[None], resolution)
            _assert_matches(
                coverage.cells,
                _polygon_margins(vertices, resolution),
                -CONTAINMENT_EPSILON,
                f"{sides}-gon trial {trial} at resolution {resolution}",
            )


@pytest.mark.parametrize("resolution", [0, 2, 4, 6])
def test_cap_coverage_matches_brute_force_centers(resolution: int, subtests) -> None:
    """The analytic cap ranges must agree with the cap predicate everywhere."""
    rng = np.random.default_rng(20260821 + resolution)
    radii = [1e-6, 0.01, 0.2, 1.0, 1.5, np.pi / 2, 3.0, np.pi]
    centers = np.vstack([_unit(rng.normal(size=3)) for _ in radii])
    # Axes on and beside both poles: the pole cases take their own branches.
    centers = np.vstack([centers, [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]])
    radii = radii + [0.3, 0.3]
    for index, (center, radius) in enumerate(zip(centers, radii, strict=True)):
        with subtests.test(index=index, radius=radius):
            coverage = px.cover_cap(center[None], radius, resolution)
            _assert_matches(
                coverage.cells,
                _cap_margins(center, radius, resolution),
                0.0,
                f"cap {index} radius {radius} at resolution {resolution}",
            )


@pytest.mark.parametrize("resolution", [1, 3, 5])
def test_sweep_segments_match_their_quads(resolution: int) -> None:
    """A sweep segment must cover exactly its four-vertex footprint."""
    latitudes = np.linspace(-70.0, 70.0, 25)
    longitudes = np.linspace(-100.0, 100.0, 25)
    left = np.asarray(
        [
            _lonlat(lon - 3.0, lat)
            for lon, lat in zip(longitudes, latitudes, strict=True)
        ]
    )
    right = np.asarray(
        [
            _lonlat(lon + 3.0, lat)
            for lon, lat in zip(longitudes, latitudes, strict=True)
        ]
    )
    coverage = px.cover_sweep(left, right, resolution)

    for segment in range(coverage.segment_count):
        quad = np.asarray(
            [left[segment], right[segment], right[segment + 1], left[segment + 1]]
        )
        start, stop = coverage.offsets[segment], coverage.offsets[segment + 1]
        _assert_matches(
            coverage.cells[start:stop],
            _polygon_margins(quad, resolution),
            -CONTAINMENT_EPSILON,
            f"sweep segment {segment} at resolution {resolution}",
        )


@pytest.mark.parametrize("resolution", [2, 4])
def test_candidate_filtering_matches_the_full_scan(resolution: int) -> None:
    """Selecting every cell must not change which cells come back.

    The two paths are different kernels - a ring scan against a per-candidate
    predicate - so this is the cheapest check that they agree.
    """
    rng = np.random.default_rng(20260822 + resolution)
    every_cell = np.arange(px.cell_count(resolution))
    polygons = np.stack(
        [_convex_polygon(rng, 4, float(rng.uniform(0.02, 0.9))) for _ in range(24)]
    )
    caps = np.vstack([_unit(rng.normal(size=3)) for _ in range(24)])
    radii = rng.uniform(0.01, 1.0, size=24)

    for name, scanned, filtered in [
        (
            "polygon",
            px.cover_convex_polygon(polygons, resolution),
            px.cover_convex_polygon(polygons, resolution, candidate_cells=every_cell),
        ),
        (
            "cap",
            px.cover_cap(caps, radii, resolution),
            px.cover_cap(caps, radii, resolution, candidate_cells=every_cell),
        ),
    ]:
        np.testing.assert_array_equal(scanned.offsets, filtered.offsets, err_msg=name)
        for segment in range(scanned.segment_count):
            start, stop = scanned.offsets[segment], scanned.offsets[segment + 1]
            np.testing.assert_array_equal(
                np.sort(scanned.cells[start:stop]),
                np.sort(filtered.cells[start:stop]),
                err_msg=f"{name} segment {segment}",
            )


@pytest.mark.parametrize("threads", [1, 2, 3, 8])
def test_results_are_invariant_across_thread_counts(threads: int) -> None:
    """Chunk boundaries must not reach the result.

    Covers the reduced forms as well as the materialized one: a dense
    ``Count``/``Sum`` reaches its result through per-worker accumulators that
    are merged, so a thread count that changes the chunking must not change
    the answer - bitwise, for ``Sum``.
    """
    rng = np.random.default_rng(20260823)
    resolution = 8
    polygons = np.stack(
        [_convex_polygon(rng, 4, float(rng.uniform(0.005, 0.05))) for _ in range(1500)]
    )
    caps = np.vstack([_unit(rng.normal(size=3)) for _ in range(4000)])
    radii = rng.uniform(0.01, 0.06, size=4000)
    polygon_values = rng.normal(size=1500)
    cap_values = rng.normal(size=4000)
    latitudes = np.linspace(-50.0, 50.0, 1501)
    longitudes = np.linspace(-150.0, 150.0, 1501)
    left = np.asarray(
        [
            _lonlat(lon - 1.0, lat)
            for lon, lat in zip(longitudes, latitudes, strict=True)
        ]
    )
    right = np.asarray(
        [
            _lonlat(lon + 1.0, lat)
            for lon, lat in zip(longitudes, latitudes, strict=True)
        ]
    )
    sweep_values = rng.normal(size=1500)

    reference = px.cover_convex_polygon(polygons, resolution, threads=1)
    actual = px.cover_convex_polygon(polygons, resolution, threads=threads)
    np.testing.assert_array_equal(reference.cells, actual.cells)
    np.testing.assert_array_equal(reference.offsets, actual.offsets)

    cap_reference = px.cover_cap(caps, radii, resolution, threads=1)
    cap_actual = px.cover_cap(caps, radii, resolution, threads=threads)
    np.testing.assert_array_equal(cap_reference.cells, cap_actual.cells)
    np.testing.assert_array_equal(cap_reference.offsets, cap_actual.offsets)

    for name, cover, kwargs, segment_values in [
        (
            "polygon",
            px.cover_convex_polygon,
            {"polygons_xyz": polygons},
            polygon_values,
        ),
        ("cap", px.cover_cap, {"centers_xyz": caps, "radii_rad": radii}, cap_values),
        (
            "sweep",
            px.cover_sweep,
            {"left_edge_xyz": left, "right_edge_xyz": right},
            sweep_values,
        ),
    ]:
        np.testing.assert_array_equal(
            cover(**kwargs, resolution=resolution, threads=1, into=px.Count()),
            cover(**kwargs, resolution=resolution, threads=threads, into=px.Count()),
            err_msg=f"{name} dense Count",
        )
        # Sums are bitwise-compared on purpose: a reassociated partial sum is
        # a behaviour change, not a rounding detail.
        np.testing.assert_array_equal(
            cover(
                **kwargs, resolution=resolution, threads=1, into=px.Sum(segment_values)
            ),
            cover(
                **kwargs,
                resolution=resolution,
                threads=threads,
                into=px.Sum(segment_values),
            ),
            err_msg=f"{name} dense Sum",
        )


@pytest.mark.parametrize("threads", [1, None])
def test_fused_reducers_match_materialize_then_reduce(threads: int | None) -> None:
    """``into=`` must be an optimization, never a different answer.

    ``cover_cap`` has a fused kernel for dense counts and for small
    selections; polygons and sweeps materialize a ``Coverage`` and reduce it,
    and a small selection on any of them is answered by testing those cells
    instead of scanning. Every one of those routes must equal reducing that
    ``Coverage`` by hand, so which route runs stays invisible - and fusing a
    further pair later cannot change a result silently.
    """
    rng = np.random.default_rng(20260824)
    resolution = 7
    caps = np.vstack([_unit(rng.normal(size=3)) for _ in range(600)])
    radii = rng.uniform(0.01, 0.15, size=600)
    polygons = np.stack(
        [_convex_polygon(rng, 4, float(rng.uniform(0.01, 0.1))) for _ in range(600)]
    )
    latitudes = np.linspace(-50.0, 50.0, 601)
    longitudes = np.linspace(-150.0, 150.0, 601)
    left = np.asarray(
        [
            _lonlat(lon - 1.0, lat)
            for lon, lat in zip(longitudes, latitudes, strict=True)
        ]
    )
    right = np.asarray(
        [
            _lonlat(lon + 1.0, lat)
            for lon, lat in zip(longitudes, latitudes, strict=True)
        ]
    )
    values = rng.normal(size=600)
    selected = rng.choice(px.cell_count(resolution), size=500, replace=False)

    for name, cover in [
        (
            "cap",
            lambda **kw: px.cover_cap(caps, radii, resolution, threads=threads, **kw),
        ),
        (
            "polygon",
            lambda **kw: px.cover_convex_polygon(
                polygons, resolution, threads=threads, **kw
            ),
        ),
        (
            "sweep",
            lambda **kw: px.cover_sweep(left, right, resolution, threads=threads, **kw),
        ),
    ]:
        coverage = cover()
        for reducer in [
            px.Count(),
            px.Count(cells=selected),
            px.Sum(values),
            px.Sum(values, cells=selected),
        ]:
            np.testing.assert_array_equal(
                cover(into=reducer),
                coverage.reduce(reducer),
                err_msg=f"{name} {type(reducer).__name__} threads={threads}",
            )


def test_sum_adds_in_segment_order_and_reports_overflow() -> None:
    """Pin the floating-point contract a reducer rewrite must not relax.

    Cancelling terms make the association observable: added in segment order
    the result is exactly zero, while any regrouping that pairs the large
    terms first leaves 1.0 behind. This is the test that a weighted difference
    array would fail.
    """
    cell = 7
    coverage = px.Coverage.from_arrays(
        cells=[cell, cell, cell], offsets=[0, 1, 2, 3], resolution=2
    )
    ordered = coverage.reduce(px.Sum([1e16, 1.0, -1e16], cells=[cell]))
    assert ordered[0] == 0.0
    assert (1e16 + 1.0) - 1e16 == 0.0
    assert (1e16 - 1e16) + 1.0 == 1.0

    # Dense and selected paths must agree on the same association.
    assert coverage.reduce(px.Sum([1e16, 1.0, -1e16]))[cell] == 0.0

    overflowing = px.Coverage.from_arrays(cells=[0, 0], offsets=[0, 1, 2], resolution=2)
    with pytest.raises(ValueError, match="overflow|finite|too large"):
        overflowing.reduce(px.Sum([1e308, 1e308]))


@pytest.mark.parametrize("minimum_sources", [1, 2, 3])
def test_occupancy_agrees_across_memory_profiles(minimum_sources: int) -> None:
    """The dense and sparse accumulators must be interchangeable.

    Runs and statistics depend only on cell identity and segment order, so the
    same cell lists reduced at a dense resolution and at a sparse one must
    agree on everything but the reported resolution. Resolution 6 always fits
    the dense accumulator; resolution 10 never does.
    """
    rng = np.random.default_rng(20260825 + minimum_sources)
    sources = []
    for _ in range(3):
        cells, offsets = [], [0]
        for _ in range(40):
            chosen = np.flatnonzero(rng.random(60) < 0.3)
            cells.extend(int(cell) for cell in chosen)
            offsets.append(len(cells))
        sources.append((np.asarray(cells, np.int64), np.asarray(offsets, np.int64)))

    def occupancy(resolution: int, **kwargs: object):
        return px.occupancy(
            [
                px.Coverage.from_arrays(cells, offsets, resolution=resolution)
                for cells, offsets in sources
            ],
            minimum_sources=minimum_sources,
            **kwargs,
        )

    dense, sparse = occupancy(6), occupancy(10)
    for name in ["cells", "offsets", "starts", "stops"]:
        np.testing.assert_array_equal(
            getattr(dense, name), getattr(sparse, name), err_msg=name
        )

    dense_stats, sparse_stats = (
        occupancy(6, into=px.Stats()),
        occupancy(10, into=px.Stats()),
    )
    for name in [
        "cells",
        "run_counts",
        "internal_gap_steps_sum",
        "maximum_internal_gap_steps",
        "first_start",
        "last_stop",
    ]:
        np.testing.assert_array_equal(
            getattr(dense_stats, name), getattr(sparse_stats, name), err_msg=name
        )


def _quad_through_point(
    point: np.ndarray, pole: np.ndarray, half_angle: float
) -> np.ndarray:
    """A quad with one edge plane passing exactly through ``point``.

    Constructed rather than sampled: the edge normal is built perpendicular to
    ``point``, so the containment margin for that cell is zero to the last bit
    and the longitude bound derived from the edge lands exactly on the cell's
    ring index. Random and even cell-corner-aligned footprints do not reach
    this state.
    """
    normal = _unit(np.cross(point, pole))
    along = _unit(np.cross(normal, point))
    first = _unit(np.cos(half_angle) * point - np.sin(half_angle) * along)
    second = _unit(np.cos(half_angle) * point + np.sin(half_angle) * along)
    return np.asarray(
        [first, second, _unit(second + 0.1 * normal), _unit(first + 0.1 * normal)]
    )


@pytest.mark.parametrize("resolution", [2, 4, 6])
def test_edge_lying_exactly_on_a_cell_center_is_covered(resolution: int) -> None:
    """A cell centre exactly on a footprint edge is inside, per the predicate.

    ``>= -CONTAINMENT_EPSILON`` makes an edge that grazes a centre inclusive,
    so the ring scan has to keep a longitude bound that rounds onto its own
    index. Dropping that widening loses the cell here - and for a single-cell
    footprint loses the whole result - while leaving every random and
    cell-corner-aligned case in this file untouched.
    """
    poles = np.asarray(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.6, -0.3, 0.75]]
    )
    centers = _grid_centers(resolution)
    every_cell = np.arange(px.cell_count(resolution))
    cells = np.arange(
        0, px.cell_count(resolution), max(1, px.cell_count(resolution) // 64)
    )
    checked = 0
    for cell in cells:
        for pole in poles:
            for half_angle in (0.05, 0.3):
                quad = _quad_through_point(centers[cell], pole, half_angle)
                coverage = px.cover_convex_polygon(quad[None], resolution)
                assert int(cell) in set(coverage.cells.tolist()), (
                    f"cell {cell} lies on the edge of its own footprint but was "
                    f"not covered at resolution {resolution}"
                )
                _assert_matches(
                    coverage.cells,
                    _polygon_margins(quad, resolution),
                    -CONTAINMENT_EPSILON,
                    f"grazing quad on cell {cell} at resolution {resolution}",
                )
                # The candidate path evaluates the predicate per cell instead of
                # solving longitude bounds, so it is the sharpest check that the
                # bounds admit everything the predicate accepts.
                np.testing.assert_array_equal(
                    np.sort(coverage.cells),
                    np.sort(
                        px.cover_convex_polygon(
                            quad[None], resolution, candidate_cells=every_cell
                        ).cells
                    ),
                    err_msg=f"scan and candidate paths differ on cell {cell}",
                )
                checked += 1
    assert checked >= 64, "the generator must actually produce footprints"


@pytest.mark.parametrize("source_resolution", [1, 2, 3])
def test_cell_corner_footprints_match_brute_force_centers(
    source_resolution: int,
) -> None:
    """Footprints whose edges are exactly cell boundaries.

    Every edge runs along a boundary between centres, so this is the densest
    supply of near-zero containment margins that ordinary geometry produces.
    """
    corners = px.cell_corners(
        np.arange(px.cell_count(source_resolution)), source_resolution
    )
    for target_resolution in (
        source_resolution,
        source_resolution + 1,
        source_resolution + 2,
    ):
        for cell in range(px.cell_count(source_resolution)):
            quad = corners[cell]
            coverage = px.cover_convex_polygon(quad[None], target_resolution)
            _assert_matches(
                coverage.cells,
                _polygon_margins(quad, target_resolution),
                -CONTAINMENT_EPSILON,
                f"corners of cell {cell} at resolution {target_resolution}",
            )


def _grid_quads(rng: np.random.Generator, count: int, half_deg: float) -> np.ndarray:
    return np.asarray(
        [
            [
                _lonlat(lon - half_deg, lat - half_deg),
                _lonlat(lon + half_deg, lat - half_deg),
                _lonlat(lon + half_deg, lat + half_deg),
                _lonlat(lon - half_deg, lat + half_deg),
            ]
            for lat, lon in zip(
                rng.uniform(-55.0, 55.0, count),
                rng.uniform(-180.0, 180.0, count),
                strict=True,
            )
        ]
    )


@pytest.mark.parametrize("resolution", [5, 8, 10])
def test_selected_reducers_ignore_how_the_kernel_reaches_the_cells(
    resolution: int, subtests
) -> None:
    """A cell selection may be answered by testing or by scanning, never both ways.

    At or below an internal size bound the selection is handed to the kernel as
    its candidate set, so it tests those cells directly rather than scanning
    every ring the footprints cross. That is a dispatch decision and must be
    invisible: identical counts, and bitwise-identical sums, whichever side of
    the bound a selection falls on. The sizes are derived from the bound rather
    than hard-coded so they keep straddling it if it moves.
    """
    rng = np.random.default_rng(20260826 + resolution)
    quads = _grid_quads(rng, 400, 1.5)
    values = rng.normal(size=quads.shape[0]) * 1e6
    grid = px.cell_count(resolution)
    bound = min(
        grid // px._SELECTED_CANDIDATE_GRID_DIVISOR, px._SELECTED_CANDIDATE_MAXIMUM
    )
    sizes = sorted({0, 1, max(bound, 1), bound + 1, min(grid, 4 * bound + 8)})
    for size in sizes:
        selections = {
            "sorted": np.sort(rng.choice(grid, size=size, replace=False)),
            "unsorted": rng.permutation(grid)[:size],
            "duplicated": np.repeat(
                np.sort(rng.choice(grid, size=size // 2, replace=False)), 2
            ),
        }
        coverage = px.cover_convex_polygon(quads, resolution, threads=1)
        for name, selected in selections.items():
            for reducer in (px.Count(cells=selected), px.Sum(values, cells=selected)):
                with subtests.test(
                    size=size, selection=name, reducer=type(reducer).__name__
                ):
                    np.testing.assert_array_equal(
                        px.cover_convex_polygon(
                            quads, resolution, threads=1, into=reducer
                        ),
                        coverage.reduce(reducer),
                    )


def test_explicit_candidates_still_bound_a_selected_reduction() -> None:
    """``candidate_cells`` is a restriction the caller asked for, not a hint.

    A selection outside it must stay zero, so the internal substitution has to
    leave an explicit candidate set alone.
    """
    rng = np.random.default_rng(20260827)
    resolution = 6
    quads = _grid_quads(rng, 200, 2.0)
    dense = px.cover_convex_polygon(quads, resolution, threads=1, into=px.Count())
    covered = np.flatnonzero(dense)
    inside, outside = covered[:8], covered[8:16]
    selected = np.concatenate([inside, outside])

    restricted = px.cover_convex_polygon(
        quads,
        resolution,
        threads=1,
        candidate_cells=inside,
        into=px.Count(cells=selected),
    )
    np.testing.assert_array_equal(restricted[: inside.size], dense[inside])
    np.testing.assert_array_equal(
        restricted[inside.size :], np.zeros(outside.size, dtype=np.int64)
    )


@pytest.mark.parametrize("resolution", [6, 9])
def test_selected_cap_and_sweep_reductions_match_materialize_then_reduce(
    resolution: int,
) -> None:
    """The substitution applies to every covering call, not only polygons."""
    rng = np.random.default_rng(20260828 + resolution)
    caps = np.vstack([_unit(rng.normal(size=3)) for _ in range(300)])
    radii = rng.uniform(0.02, 0.2, size=300)
    latitudes = np.linspace(-60.0, 60.0, 40)
    longitudes = np.linspace(-120.0, 120.0, 40)
    left = np.asarray(
        [
            _lonlat(lon - 2.0, lat)
            for lon, lat in zip(longitudes, latitudes, strict=True)
        ]
    )
    right = np.asarray(
        [
            _lonlat(lon + 2.0, lat)
            for lon, lat in zip(longitudes, latitudes, strict=True)
        ]
    )
    small = np.sort(rng.choice(px.cell_count(resolution), size=64, replace=False))

    for name, cover, segments in [
        (
            "cap",
            lambda **kw: px.cover_cap(caps, radii, resolution, threads=1, **kw),
            300,
        ),
        (
            "sweep",
            lambda **kw: px.cover_sweep(left, right, resolution, threads=1, **kw),
            39,
        ),
    ]:
        coverage = cover()
        values = rng.normal(size=segments)
        for reducer in (px.Count(cells=small), px.Sum(values, cells=small)):
            np.testing.assert_array_equal(
                cover(into=reducer),
                coverage.reduce(reducer),
                err_msg=f"{name} {type(reducer).__name__}",
            )
