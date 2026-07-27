"""End-to-end correctness and performance scorecard for Polypix.

Run the standard corpus from a source checkout with:

    python -m benchmarks.scorecard --output scorecard.json

Only NumPy and Polypix are required.  The healpy and cdshealpix adapters report
themselves as unavailable when their optional packages are not installed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np

import polypix as px


SCHEMA_VERSION = 1
CENTER_EPSILON = 2.0e-14
BackendName = Literal["polypix", "healpy", "cdshealpix"]
WorkloadKind = Literal["footprints", "strip", "candidates"]


class UnsupportedWorkload(RuntimeError):
    """Raised when an installed backend cannot run an equivalent workload."""


@dataclass(frozen=True)
class Workload:
    """One deterministic end-to-end scorecard workload."""

    name: str
    kind: WorkloadKind
    resolution: int
    item_count: int
    footprints: np.ndarray | tuple[np.ndarray, ...] | None = None
    left_edge: np.ndarray | None = None
    right_edge: np.ndarray | None = None
    candidate_cells: np.ndarray | None = None
    primary: bool = True


@dataclass(frozen=True)
class CoverageResult:
    """Backend-neutral, standard-RING representation used by the scorecard."""

    cells: np.ndarray
    offsets: np.ndarray


@dataclass(frozen=True)
class Backend:
    name: BackendName
    version: str | None
    available: bool
    detail: str
    run: Callable[[Workload, int | None], CoverageResult] | None


def lonlat_to_xyz(lon_deg: float, lat_deg: float) -> np.ndarray:
    """Convert unambiguous scorecard fixture coordinates to a unit vector."""

    lon = math.radians(lon_deg)
    lat = math.radians(lat_deg)
    cos_lat = math.cos(lat)
    return np.asarray(
        [cos_lat * math.cos(lon), cos_lat * math.sin(lon), math.sin(lat)],
        dtype=np.float64,
    )


def _tangent_basis(center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = (
        np.asarray([0.0, 0.0, 1.0])
        if abs(float(center[2])) < 0.9
        else np.asarray([1.0, 0.0, 0.0])
    )
    first = np.cross(reference, center)
    first /= np.linalg.norm(first)
    second = np.cross(center, first)
    second /= np.linalg.norm(second)
    return first, second


def regular_footprint(
    center: np.ndarray,
    radius_deg: float,
    vertex_count: int,
    rotation_rad: float = 0.0,
) -> np.ndarray:
    """Construct a small, strictly convex great-circle polygon."""

    center = np.asarray(center, dtype=np.float64)
    center = center / np.linalg.norm(center)
    first, second = _tangent_basis(center)
    radius = math.radians(radius_deg)
    angles = rotation_rad + np.arange(vertex_count, dtype=np.float64) * (
        2.0 * math.pi / vertex_count
    )
    tangent = (
        np.cos(angles)[:, np.newaxis] * first + np.sin(angles)[:, np.newaxis] * second
    )
    return np.ascontiguousarray(
        math.cos(radius) * center + math.sin(radius) * tangent,
        dtype=np.float64,
    )


def _fixture_centers(count: int) -> np.ndarray:
    """Return deterministic, well-distributed centers away from exact poles."""

    index = np.arange(count, dtype=np.float64)
    longitude = np.remainder(index * 137.50776405003785, 360.0) - 180.0
    latitude = np.degrees(np.arcsin(-0.85 + 1.7 * (index + 0.5) / count))
    return np.asarray(
        [
            lonlat_to_xyz(float(lon), float(lat))
            for lon, lat in zip(longitude, latitude, strict=True)
        ],
        dtype=np.float64,
    )


def dense_footprints(count: int, radius_deg: float = 0.7) -> np.ndarray:
    """Generate deterministic quadrilateral footprints."""

    centers = _fixture_centers(count)
    return np.asarray(
        [
            regular_footprint(
                center,
                radius_deg * (0.8 + 0.4 * ((index % 11) / 10.0)),
                4,
                rotation_rad=(index % 13) * 0.07,
            )
            for index, center in enumerate(centers)
        ],
        dtype=np.float64,
    )


def ragged_footprints(
    count: int,
    radius_deg: float = 0.8,
) -> tuple[np.ndarray, ...]:
    """Generate deterministic convex footprints with 3--6 vertices."""

    centers = _fixture_centers(count)
    return tuple(
        regular_footprint(
            center,
            radius_deg * (0.85 + 0.3 * ((index % 7) / 6.0)),
            3 + index % 4,
            rotation_rad=(index % 17) * 0.05,
        )
        for index, center in enumerate(centers)
    )


def strip_edges(
    sample_count: int, half_width_deg: float = 0.6
) -> tuple[np.ndarray, np.ndarray]:
    """Generate paired edges along a generic inclined spherical track."""

    phase = np.linspace(-1.25 * math.pi, 1.25 * math.pi, sample_count)
    longitude = np.degrees(phase)
    latitude = 42.0 * np.sin(phase * 0.55)
    centerline = np.asarray(
        [
            lonlat_to_xyz(float(lon), float(lat))
            for lon, lat in zip(longitude, latitude, strict=True)
        ],
        dtype=np.float64,
    )
    tangent = np.empty_like(centerline)
    tangent[1:-1] = centerline[2:] - centerline[:-2]
    tangent[0] = centerline[1] - centerline[0]
    tangent[-1] = centerline[-1] - centerline[-2]
    tangent -= np.sum(tangent * centerline, axis=1)[:, np.newaxis] * centerline
    tangent /= np.linalg.norm(tangent, axis=1)[:, np.newaxis]
    cross_track = np.cross(centerline, tangent)
    cross_track /= np.linalg.norm(cross_track, axis=1)[:, np.newaxis]
    width = math.radians(half_width_deg)
    left = math.cos(width) * centerline + math.sin(width) * cross_track
    right = math.cos(width) * centerline - math.sin(width) * cross_track
    return np.ascontiguousarray(left), np.ascontiguousarray(right)


def sparse_candidate_cells(resolution: int, count: int) -> np.ndarray:
    """Generate deterministic, globally distributed standard RING indices."""

    pixel_count = 12 * (4**resolution)
    if count > pixel_count:
        raise ValueError("candidate count exceeds the number of cells")
    # 104729 is coprime with 3 * 2**N, so this sequence cannot repeat before
    # visiting the complete fixed-resolution HEALPix index space.
    values = (
        np.arange(count, dtype=np.uint64) * np.uint64(104729) + np.uint64(8191)
    ) % np.uint64(pixel_count)
    return np.ascontiguousarray(values)


def build_workloads(
    profile: Literal["smoke", "standard"] = "standard",
) -> list[Workload]:
    """Build the fixed corpus without timing fixture construction."""

    if profile == "smoke":
        dense_count = 24
        ragged_count = 16
        strip_samples = 25
        candidate_footprints = 16
        candidate_count = 1024
        resolutions = (4, 6)
        candidate_resolution = 9
    else:
        dense_count = 4096
        ragged_count = 2048
        strip_samples = 4097
        candidate_footprints = 512
        candidate_count = 8192
        resolutions = (6, 9)
        candidate_resolution = 12

    dense = dense_footprints(dense_count)
    ragged = ragged_footprints(ragged_count)
    left, right = strip_edges(strip_samples)
    candidates = sparse_candidate_cells(candidate_resolution, candidate_count)
    candidate_shapes = dense_footprints(candidate_footprints, radius_deg=2.5)

    return [
        Workload(
            name=f"dense_quads_r{resolution}",
            kind="footprints",
            resolution=resolution,
            item_count=dense_count,
            footprints=dense,
        )
        for resolution in resolutions
    ] + [
        Workload(
            name=f"ragged_3_to_6_r{resolutions[-1]}",
            kind="footprints",
            resolution=resolutions[-1],
            item_count=ragged_count,
            footprints=ragged,
        ),
        Workload(
            name=f"paired_edge_strip_r{resolutions[-1]}",
            kind="strip",
            resolution=resolutions[-1],
            item_count=strip_samples - 1,
            left_edge=left,
            right_edge=right,
        ),
        Workload(
            name=f"sparse_candidates_r{candidate_resolution}",
            kind="candidates",
            resolution=candidate_resolution,
            item_count=candidate_footprints,
            footprints=candidate_shapes,
            candidate_cells=candidates,
        ),
        Workload(
            name=f"single_footprint_r{resolutions[-1]}",
            kind="footprints",
            resolution=resolutions[-1],
            item_count=1,
            footprints=dense_footprints(1, radius_deg=2.0),
            primary=False,
        ),
    ]


def strip_footprints(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Expand paired edges into the equivalent quadrilateral batch."""

    footprints = np.empty((left.shape[0] - 1, 4, 3), dtype=np.float64)
    footprints[:, 0] = left[:-1]
    footprints[:, 1] = right[:-1]
    footprints[:, 2] = right[1:]
    footprints[:, 3] = left[1:]
    return footprints


def _segments(footprints: np.ndarray | tuple[np.ndarray, ...]) -> Sequence[np.ndarray]:
    if isinstance(footprints, tuple):
        return footprints
    if footprints.ndim == 2:
        return (footprints,)
    return footprints


def _polypix_result(result: px.Coverage) -> CoverageResult:
    return CoverageResult(
        cells=np.ascontiguousarray(result.cells, dtype=np.uint64),
        offsets=np.ascontiguousarray(result.offsets, dtype=np.uint64),
    )


def _run_polypix(workload: Workload, threads: int | None) -> CoverageResult:
    if workload.kind == "strip":
        assert workload.left_edge is not None and workload.right_edge is not None
        return _polypix_result(
            px.cover_strip(
                workload.left_edge,
                workload.right_edge,
                workload.resolution,
                candidate_cells=workload.candidate_cells,
                threads=threads,
            )
        )

    assert workload.footprints is not None
    return _polypix_result(
        px.cover_footprint(
            workload.footprints,
            workload.resolution,
            candidate_cells=workload.candidate_cells,
            threads=threads,
        )
    )


def _xyz_to_lonlat_rad(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    longitude = np.arctan2(vertices[:, 1], vertices[:, 0])
    latitude = np.arctan2(
        vertices[:, 2],
        np.hypot(vertices[:, 0], vertices[:, 1]),
    )
    return longitude, latitude


def _oriented_edge_normals(vertices: np.ndarray) -> np.ndarray:
    normalized = vertices / np.linalg.norm(vertices, axis=1)[:, np.newaxis]
    normals = np.cross(normalized, np.roll(normalized, -1, axis=0))
    interior = np.sum(normalized, axis=0)
    interior /= np.linalg.norm(interior)
    if float(np.sum(normals @ interior)) < 0.0:
        normals = -normals
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]
    return normals


def _candidate_coverage(
    footprints: Sequence[np.ndarray],
    candidates: np.ndarray,
    candidate_centers: np.ndarray,
) -> CoverageResult:
    chunks: list[np.ndarray] = []
    offsets = np.empty(len(footprints) + 1, dtype=np.uint64)
    offsets[0] = 0
    for index, footprint in enumerate(footprints):
        normals = _oriented_edge_normals(np.asarray(footprint))
        mask = np.all(candidate_centers @ normals.T >= -CENTER_EPSILON, axis=1)
        chunks.append(candidates[mask])
        offsets[index + 1] = offsets[index] + np.uint64(np.count_nonzero(mask))
    cells = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint64)
    return CoverageResult(np.ascontiguousarray(cells), offsets)


def _concatenate_segments(chunks: Sequence[np.ndarray]) -> CoverageResult:
    offsets = np.empty(len(chunks) + 1, dtype=np.uint64)
    offsets[0] = 0
    for index, chunk in enumerate(chunks):
        offsets[index + 1] = offsets[index] + np.uint64(chunk.size)
    cells = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint64)
    return CoverageResult(
        np.ascontiguousarray(cells, dtype=np.uint64),
        offsets,
    )


def _healpy_backend() -> Backend:
    try:
        import healpy as hp
    except ImportError as exc:
        return Backend("healpy", None, False, str(exc), None)

    def run(workload: Workload, threads: int | None) -> CoverageResult:
        if threads not in (None, 1):
            raise UnsupportedWorkload(
                "healpy query_polygon has no thread-count control"
            )
        nside = 1 << workload.resolution
        if workload.kind == "strip":
            assert workload.left_edge is not None and workload.right_edge is not None
            footprints: Sequence[np.ndarray] = strip_footprints(
                workload.left_edge, workload.right_edge
            )
        else:
            assert workload.footprints is not None
            footprints = _segments(workload.footprints)

        if workload.kind == "candidates":
            assert workload.candidate_cells is not None
            # healpy's ufunc accepts signed pixel indices.  Every valid
            # fixed-resolution HEALPix index fits safely in int64.
            healpy_cells = workload.candidate_cells.astype(np.int64, copy=False)
            x, y, z = hp.pix2vec(nside, healpy_cells, nest=False)
            centers = np.column_stack((x, y, z))
            return _candidate_coverage(footprints, workload.candidate_cells, centers)

        chunks = [
            np.asarray(
                hp.query_polygon(
                    nside,
                    np.asarray(footprint),
                    inclusive=False,
                    nest=False,
                ),
                dtype=np.uint64,
            )
            for footprint in footprints
        ]
        return _concatenate_segments(chunks)

    return Backend(
        "healpy",
        getattr(hp, "__version__", None),
        True,
        "query_polygon(inclusive=False, nest=False); batch loop is timed",
        run,
    )


def _cdshealpix_backend() -> Backend:
    try:
        import astropy.units as units
        from astropy.coordinates import Latitude, Longitude
        from cdshealpix import nested, ring, to_ring
    except ImportError as exc:
        return Backend("cdshealpix", None, False, str(exc), None)

    try:
        version = importlib.metadata.version("cdshealpix")
    except importlib.metadata.PackageNotFoundError:
        version = None

    def vectors(lon: Any, lat: Any) -> np.ndarray:
        lon_values = np.asarray(lon.to_value(units.rad))
        lat_values = np.asarray(lat.to_value(units.rad))
        cos_lat = np.cos(lat_values)
        return np.column_stack(
            (
                cos_lat * np.cos(lon_values),
                cos_lat * np.sin(lon_values),
                np.sin(lat_values),
            )
        )

    def nested_centers(cells: np.ndarray, resolution: int) -> np.ndarray:
        lon, lat = nested.healpix_to_lonlat(cells, resolution)
        return vectors(lon, lat)

    def ring_centers(cells: np.ndarray, resolution: int) -> np.ndarray:
        lon, lat = ring.healpix_to_lonlat(cells, 1 << resolution)
        return vectors(lon, lat)

    def run(workload: Workload, threads: int | None) -> CoverageResult:
        if threads not in (None, 1):
            raise UnsupportedWorkload(
                "cdshealpix polygon_search has no thread-count control"
            )
        if workload.kind == "strip":
            assert workload.left_edge is not None and workload.right_edge is not None
            footprints: Sequence[np.ndarray] = strip_footprints(
                workload.left_edge, workload.right_edge
            )
        else:
            assert workload.footprints is not None
            footprints = _segments(workload.footprints)

        if workload.kind == "candidates":
            assert workload.candidate_cells is not None
            return _candidate_coverage(
                footprints,
                workload.candidate_cells,
                ring_centers(workload.candidate_cells, workload.resolution),
            )

        chunks: list[np.ndarray] = []
        for footprint in footprints:
            lon, lat = _xyz_to_lonlat_rad(np.asarray(footprint))
            ipix, depths, _ = nested.polygon_search(
                Longitude(lon, unit=units.rad),
                Latitude(lat, unit=units.rad),
                workload.resolution,
                flat=True,
            )
            ipix = np.asarray(ipix, dtype=np.uint64)
            depths = np.asarray(depths)
            at_resolution = ipix[depths == workload.resolution]
            # polygon_search is a cell coverage/MOC operation, not a documented
            # center-only query.  Filter its flat cover by cell center so this
            # explicitly named adapter has semantics equivalent to Polypix.
            if at_resolution.size:
                normals = _oriented_edge_normals(np.asarray(footprint))
                inside = np.all(
                    nested_centers(at_resolution, workload.resolution) @ normals.T
                    >= -CENTER_EPSILON,
                    axis=1,
                )
                at_resolution = at_resolution[inside]
            chunks.append(
                np.asarray(
                    to_ring(at_resolution, workload.resolution),
                    dtype=np.uint64,
                )
            )
        return _concatenate_segments(chunks)

    return Backend(
        "cdshealpix",
        version,
        True,
        "flat NESTED polygon_search, center filter, and timed RING conversion",
        run,
    )


def discover_backends(names: Sequence[BackendName]) -> list[Backend]:
    """Discover requested adapters without making optional packages mandatory."""

    backends: list[Backend] = []
    for name in names:
        if name == "polypix":
            backends.append(
                Backend(
                    "polypix",
                    getattr(px, "__version__", None),
                    True,
                    "complete public Polypix call",
                    _run_polypix,
                )
            )
        elif name == "healpy":
            backends.append(_healpy_backend())
        elif name == "cdshealpix":
            backends.append(_cdshealpix_backend())
        else:  # pragma: no cover - argparse and the type checker prevent this
            raise ValueError(f"unknown backend: {name}")
    return backends


def _canonical_digest(result: CoverageResult) -> str:
    digest = hashlib.sha256()
    canonical_offsets = np.empty_like(result.offsets)
    canonical_offsets[0] = 0
    for index, (start, stop) in enumerate(
        zip(result.offsets[:-1], result.offsets[1:], strict=True)
    ):
        cells = np.sort(result.cells[int(start) : int(stop)])
        digest.update(cells.astype("<u8", copy=False).tobytes())
        canonical_offsets[index + 1] = canonical_offsets[index] + cells.size
    digest.update(canonical_offsets.astype("<u8", copy=False).tobytes())
    return digest.hexdigest()


def _has_segment_duplicates(result: CoverageResult) -> bool:
    return any(
        np.unique(result.cells[int(start) : int(stop)]).size != int(stop - start)
        for start, stop in zip(result.offsets[:-1], result.offsets[1:], strict=True)
    )


def _time_backend(
    backend: Backend,
    workload: Workload,
    threads: int | None,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, Any], CoverageResult | None]:
    mode = "auto" if threads is None else str(threads)
    if not backend.available or backend.run is None:
        return {
            "backend": backend.name,
            "workload": workload.name,
            "threads": mode,
            "status": "unavailable",
            "detail": backend.detail,
        }, None

    try:
        for _ in range(warmup):
            backend.run(workload, threads)
        timings_ns: list[int] = []
        result: CoverageResult | None = None
        for _ in range(repeats):
            started = time.perf_counter_ns()
            result = backend.run(workload, threads)
            timings_ns.append(time.perf_counter_ns() - started)
        assert result is not None
        median_ns = statistics.median(timings_ns)
        mean_ns = statistics.fmean(timings_ns)
        return {
            "backend": backend.name,
            "workload": workload.name,
            "threads": mode,
            "status": "ok",
            "timing_scope": "complete adapter/public workflow",
            "samples_ns": timings_ns,
            "median_ns": median_ns,
            "minimum_ns": min(timings_ns),
            "mean_ns": mean_ns,
            "stdev_ns": statistics.pstdev(timings_ns),
            "items_per_second": workload.item_count / (median_ns / 1.0e9),
            "item_count": workload.item_count,
            "cell_count": int(result.cells.size),
            "materialized_bytes": int(result.cells.nbytes + result.offsets.nbytes),
            "membership_sha256": _canonical_digest(result),
            "duplicate_cells_within_segment": _has_segment_duplicates(result),
        }, result
    except UnsupportedWorkload as exc:
        return {
            "backend": backend.name,
            "workload": workload.name,
            "threads": mode,
            "status": "unsupported",
            "detail": str(exc),
        }, None
    except Exception as exc:  # keep a long benchmark run report-friendly
        return {
            "backend": backend.name,
            "workload": workload.name,
            "threads": mode,
            "status": "error",
            "detail": f"{type(exc).__name__}: {exc}",
        }, None


def _polypix_centers(cells: np.ndarray, resolution: int) -> np.ndarray:
    return np.asarray(px.centers(cells, resolution), dtype=np.float64)


def _oracle_centers(
    cells: np.ndarray,
    resolution: int,
) -> tuple[np.ndarray, str]:
    try:
        import healpy as hp
    except ImportError:
        return (
            _polypix_centers(cells, resolution),
            "Polypix centers plus independent NumPy containment",
        )

    x, y, z = hp.pix2vec(
        1 << resolution,
        cells.astype(np.int64, copy=False),
        nest=False,
    )
    return (
        np.column_stack((x, y, z)),
        "healpy.pix2vec plus independent NumPy containment",
    )


def _brute_force_membership(vertices: np.ndarray, resolution: int) -> np.ndarray:
    cells = np.arange(12 * (4**resolution), dtype=np.uint64)
    centers, _ = _oracle_centers(cells, resolution)
    normals = _oriented_edge_normals(vertices)
    return cells[np.all(centers @ normals.T >= -CENTER_EPSILON, axis=1)]


def _boundary_footprint(resolution: int, cell: int) -> np.ndarray:
    center = _polypix_centers(np.asarray([cell], dtype=np.uint64), resolution)[0]
    # Use a stable tangent for this particular polar-cap cell.  Constructing
    # the edge from the returned center is intentional: it exercises the
    # public center helper and the closed-region coverage rule together.
    reference = (
        np.asarray([0.0, 0.0, 1.0])
        if abs(float(center[2])) < 0.9
        else np.asarray([0.0, 1.0, 0.0])
    )
    first = np.cross(reference, center)
    first /= np.linalg.norm(first)
    second = np.cross(center, first)
    second /= np.linalg.norm(second)

    def point(x: float, y: float) -> np.ndarray:
        value = center + x * first + y * second
        return value / np.linalg.norm(value)

    return np.asarray(
        [
            point(-0.04, 0.0),
            point(0.04, 0.0),
            point(0.04, 0.04),
            point(-0.04, 0.04),
        ]
    )


def adversarial_footprints() -> list[tuple[str, np.ndarray, int]]:
    """Small correctness cases evaluated against a brute-force center oracle."""

    return [
        (
            "antimeridian",
            regular_footprint(lonlat_to_xyz(180.0, 5.0), 13.0, 4, 0.3),
            3,
        ),
        (
            "north_pole",
            regular_footprint(np.asarray([0.0, 0.0, 1.0]), 20.0, 5, 0.1),
            3,
        ),
        (
            "south_pole",
            regular_footprint(np.asarray([0.0, 0.0, -1.0]), 18.0, 4, 0.2),
            3,
        ),
        (
            "center_on_boundary",
            _boundary_footprint(3, 123),
            3,
        ),
        (
            "near_hemisphere_limit",
            regular_footprint(lonlat_to_xyz(35.0, -10.0), 88.0, 4, 0.25),
            2,
        ),
    ]


def randomized_correctness_footprints(
    count: int = 256,
    seed: int = 0x50_4F_4C_59,
) -> tuple[np.ndarray, ...]:
    """Generate a reproducible randomized correctness corpus."""

    generator = np.random.default_rng(seed)
    centers = generator.normal(size=(count, 3))
    centers /= np.linalg.norm(centers, axis=1)[:, np.newaxis]
    return tuple(
        regular_footprint(
            center,
            radius_deg=float(generator.uniform(1.0, 55.0)),
            vertex_count=int(generator.integers(3, 7)),
            rotation_rad=float(generator.uniform(0.0, 2.0 * math.pi)),
        )
        for center in centers
    )


def run_adversarial_correctness() -> list[dict[str, Any]]:
    """Run Polypix against an exhaustive low-resolution center oracle."""

    checks: list[dict[str, Any]] = []
    _, oracle = _oracle_centers(np.asarray([0], dtype=np.uint64), 0)
    for name, footprint, resolution in adversarial_footprints():
        workload = Workload(
            name=name,
            kind="footprints",
            resolution=resolution,
            item_count=1,
            footprints=footprint,
            primary=False,
        )
        try:
            result = _run_polypix(workload, None)
            expected = _brute_force_membership(footprint, resolution)
            actual = np.sort(result.cells)
            expected = np.sort(expected)
            checks.append(
                {
                    "name": name,
                    "resolution": resolution,
                    "status": "pass" if np.array_equal(actual, expected) else "fail",
                    "oracle": oracle,
                    "expected_cells": int(expected.size),
                    "actual_cells": int(actual.size),
                    "missing_cells": int(np.setdiff1d(expected, actual).size),
                    "extra_cells": int(np.setdiff1d(actual, expected).size),
                }
            )
        except Exception as exc:
            checks.append(
                {
                    "name": name,
                    "resolution": resolution,
                    "status": "error",
                    "oracle": oracle,
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )

    random_resolution = 3
    random_missing = 0
    random_extra = 0
    random_errors: list[str] = []
    randomized = randomized_correctness_footprints()
    for index, footprint in enumerate(randomized):
        workload = Workload(
            name=f"fixed_seed_randomized_{index}",
            kind="footprints",
            resolution=random_resolution,
            item_count=1,
            footprints=footprint,
            primary=False,
        )
        try:
            actual = np.sort(_run_polypix(workload, None).cells)
            expected = np.sort(_brute_force_membership(footprint, random_resolution))
            random_missing += int(np.setdiff1d(expected, actual).size)
            random_extra += int(np.setdiff1d(actual, expected).size)
        except Exception as exc:
            random_errors.append(f"case {index}: {type(exc).__name__}: {exc}")
    checks.append(
        {
            "name": "fixed_seed_randomized",
            "resolution": random_resolution,
            "oracle": oracle,
            "status": (
                "pass"
                if not random_errors and random_missing == 0 and random_extra == 0
                else "fail"
            ),
            "case_count": len(randomized),
            "missing_cells": random_missing,
            "extra_cells": random_extra,
            "errors": random_errors,
        }
    )
    return checks


def build_report(
    *,
    profile: Literal["smoke", "standard"] = "standard",
    backend_names: Sequence[BackendName] = ("polypix", "healpy", "cdshealpix"),
    warmup: int = 1,
    repeats: int = 5,
    thread_modes: Sequence[int | None] = (None, 1),
) -> dict[str, Any]:
    """Execute the scorecard and return its JSON-serializable report."""

    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeats < 1:
        raise ValueError("repeats must be positive")

    workloads = build_workloads(profile)
    backends = discover_backends(backend_names)
    results: list[dict[str, Any]] = []
    references: dict[str, CoverageResult] = {}

    # Polypix runs first so optional adapters can be compared by membership,
    # independent of their native output order.
    ordered_backends = sorted(backends, key=lambda backend: backend.name != "polypix")
    for backend in ordered_backends:
        modes = thread_modes if backend.name == "polypix" else (None,)
        for workload in workloads:
            for threads in modes:
                record, result = _time_backend(
                    backend, workload, threads, warmup, repeats
                )
                reference = references.get(workload.name)
                if backend.name == "polypix" and threads is None and result is not None:
                    references[workload.name] = result
                    record["matches_polypix_auto"] = True
                elif reference is not None and result is not None:
                    record["matches_polypix_auto"] = _canonical_digest(
                        result
                    ) == _canonical_digest(reference)
                results.append(record)

    try:
        numpy_version = importlib.metadata.version("numpy")
    except importlib.metadata.PackageNotFoundError:
        numpy_version = np.__version__

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "profile": profile,
        "timing": {
            "clock": "time.perf_counter_ns",
            "warmup_calls": warmup,
            "measured_calls": repeats,
            "fixtures_built_outside_timing": True,
            "scope": "complete public call or explicitly described equivalent workflow",
        },
        "environment": {
            "python": sys.version,
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "numpy": numpy_version,
        },
        "backends": [
            {
                "name": backend.name,
                "version": backend.version,
                "available": backend.available,
                "detail": backend.detail,
            }
            for backend in backends
        ],
        "workloads": [
            {
                "name": workload.name,
                "kind": workload.kind,
                "resolution": workload.resolution,
                "item_count": workload.item_count,
                "candidate_count": (
                    int(workload.candidate_cells.size)
                    if workload.candidate_cells is not None
                    else None
                ),
                "primary": workload.primary,
            }
            for workload in workloads
        ],
        "correctness": run_adversarial_correctness(),
        "results": results,
    }


def _parse_threads(value: str) -> tuple[int | None, ...]:
    modes: list[int | None] = []
    for part in value.split(","):
        normalized = part.strip().lower()
        if normalized == "auto":
            modes.append(None)
            continue
        threads = int(normalized)
        if threads < 1:
            raise argparse.ArgumentTypeError("thread counts must be positive")
        modes.append(threads)
    if not modes:
        raise argparse.ArgumentTypeError("at least one thread mode is required")
    return tuple(modes)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Polypix correctness and performance scorecard."
    )
    parser.add_argument("--profile", choices=("smoke", "standard"), default="standard")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("polypix", "healpy", "cdshealpix"),
        default=("polypix", "healpy", "cdshealpix"),
    )
    parser.add_argument(
        "--threads",
        type=_parse_threads,
        default=(None, 1),
        help="comma-separated Polypix modes, for example auto,1",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="return a non-zero status for correctness failures or competitor mismatches",
    )
    args = parser.parse_args(argv)

    report = build_report(
        profile=args.profile,
        backend_names=args.backends,
        warmup=args.warmup,
        repeats=args.repeats,
        thread_modes=args.threads,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(payload)
    else:
        args.output.write_text(payload, encoding="utf-8")

    if args.fail_on_mismatch:
        correctness_failed = any(
            check["status"] != "pass" for check in report["correctness"]
        )
        result_failed = any(
            result.get("status") == "error"
            or result.get("matches_polypix_auto") is False
            for result in report["results"]
        )
        if correctness_failed or result_failed:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
