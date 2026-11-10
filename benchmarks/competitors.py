"""Compare Polypix with other sphere-gridding libraries.

Two workloads, both built from the same random shapes:

* visibility: count, for every grid cell, how many circles cover it;
* observation: find the cells covered by any of a set of small footprints.

Each library covers the shapes on its own grid, and we time that call plus the
step that turns its output into per-cell counts. We then look the answer up at
random points and compare it with exact spherical geometry.

    python benchmarks/competitors.py                 # the published runs
    python benchmarks/competitors.py --quick         # a ten-second smoke test

See competitors.md for the setup and what each comparison means.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from typing import NamedTuple

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"


class Scene(NamedTuple):
    centers: np.ndarray  # (tracks, 3) unit vectors
    radii: np.ndarray  # (tracks,) radians
    footprints: np.ndarray  # (4 * tracks, 4, 3) counter-clockwise quadrilaterals


def make_scene(tracks: int, seed: int) -> Scene:
    """Random circles and footprints between 63 degrees south and north.

    Each track has one circle of 3 to 12 degrees radius, and four overlapping
    1.4 by 0.9 degree footprints stepping east from the same center.
    """
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-np.pi, np.pi, tracks)
    lat = rng.uniform(-1.1, 1.1, tracks)
    radii = np.radians(rng.uniform(3, 12, tracks))
    centers = np.column_stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )
    east = np.column_stack([-np.sin(lon), np.cos(lon), np.zeros(tracks)])
    north = np.cross(centers, east)

    corners = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)]) * (0.012, 0.008)
    steps = np.array([(0.015 * i, 0) for i in range(4)])
    offsets = steps[:, None, :] + corners  # (step, corner, east/north)
    footprints = (
        centers[:, None, None]
        + offsets[None, ..., :1] * east[:, None, None]
        + offsets[None, ..., 1:] * north[:, None, None]
    ).reshape(-1, 4, 3)
    footprints /= np.linalg.norm(footprints, axis=-1, keepdims=True)
    return Scene(centers, radii, footprints)


def exact_counts(points: np.ndarray, scene: Scene, workload: str) -> np.ndarray:
    """Circles covering each point, or 1 where any footprint covers it."""
    if workload == "visibility":
        return np.sum(points @ scene.centers.T >= np.cos(scene.radii), axis=1)
    covered = np.zeros(len(points), dtype=bool)
    for corners in scene.footprints:
        edge_normals = np.cross(corners, np.roll(corners, -1, axis=0))
        covered |= np.all(points @ edge_normals.T >= 0, axis=1)
    return covered.astype(np.int64)


def latlng(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Latitude and longitude in degrees."""
    return np.degrees(np.arcsin(xyz[..., 2])), np.degrees(
        np.arctan2(xyz[..., 1], xyz[..., 0])
    )


def circle_outline(
    center: np.ndarray, radius: float, vertices: int = 128
) -> np.ndarray:
    """Points on a circle, for libraries that only accept polygons."""
    east = np.cross([0.0, 0.0, 1.0], center)
    east /= np.linalg.norm(east)
    north = np.cross(center, east)
    angle = np.linspace(0, 2 * np.pi, vertices, endpoint=False)[:, None]
    ring = np.cos(angle) * east + np.sin(angle) * north
    return np.cos(radius) * center + np.sin(radius) * ring


def count_cells(cell_lists: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Sorted unique cell IDs and how many lists contain each."""
    return np.unique(np.concatenate(cell_lists), return_counts=True)


# Each library turns a scene into either a dense array indexed by cell ID or a
# sorted (cell IDs, counts) pair. Imports happen inside the functions so the
# script still runs when some libraries are missing.


def polypix_cover(scene, level, mode, workload, threads=1):
    import polypix

    options = dict(resolution=level, mode=mode, threads=threads, reduce=polypix.Count())
    if workload == "visibility":
        return polypix.cover_cap(scene.centers, scene.radii, **options)
    return polypix.cover_polygon(scene.footprints, **options)


def healpy_cover(scene, level, mode, workload):
    import healpy

    nside = 2**level
    counts = np.zeros(12 * nside**2, dtype=np.int64)
    if workload == "visibility":
        for center, radius in zip(scene.centers, scene.radii, strict=True):
            counts[healpy.query_disc(nside, center, radius)] += 1
    else:
        for corners in scene.footprints:
            counts[healpy.query_polygon(nside, corners)] += 1
    return counts


def hpgeom_cover(scene, level, mode, workload):
    import hpgeom

    nside = 2**level
    counts = np.zeros(12 * nside**2, dtype=np.int64)
    if workload == "visibility":
        for center, radius in zip(scene.centers, scene.radii, strict=True):
            counts[hpgeom.query_circle_vec(nside, center, radius, nest=False)] += 1
    else:
        for corners in scene.footprints:
            counts[hpgeom.query_polygon_vec(nside, corners, nest=False)] += 1
    return counts


def mocpy_cover(scene, level, mode, workload):
    import astropy.units as u
    from mocpy import MOC

    if workload == "visibility":
        lat, lon = latlng(scene.centers)
        mocs = MOC.from_cones(
            lon=lon * u.deg,
            lat=lat * u.deg,
            radius=scene.radii * u.rad,
            max_depth=level,
            n_threads=1,
        )
    else:
        lat, lon = latlng(scene.footprints)
        # MOCPy wants alternating rows of longitudes and latitudes, in [0, 360).
        rows = np.stack([lon % 360, lat], axis=1).reshape(-1, lon.shape[1])
        mocs = MOC.from_polygons(rows, max_depth=level, n_threads=1)
    return count_cells([moc.flatten() for moc in mocs])


def h3_cover(scene, level, mode, workload):
    from h3.api import numpy_int as h3

    if workload == "visibility":
        shapes = [
            circle_outline(*cap) for cap in zip(scene.centers, scene.radii, strict=True)
        ]
    else:
        shapes = scene.footprints
    cells = []
    for shape in shapes:
        polygon = h3.LatLngPoly(np.column_stack(latlng(shape)).tolist())
        if mode == "center":
            cells.append(h3.polygon_to_cells(polygon, level))
        else:
            cells.append(
                h3.h3shape_to_cells_experimental(polygon, level, contain="overlap")
            )
    return count_cells(cells)


def s2_cover(scene, level, mode, workload):
    import s2geometry as s2

    def point(xyz):
        return s2.S2Point_FromRaw(*map(float, xyz))

    if workload == "visibility":
        regions = [
            s2.S2Cap(point(center), s2.S1Angle.Radians(float(radius)))
            for center, radius in zip(scene.centers, scene.radii, strict=True)
        ]
    else:
        regions = []
        for corners in scene.footprints:
            loop = s2.S2Loop([point(corner) for corner in corners])
            loop.Normalize()
            regions.append(s2.S2Polygon(loop))
    coverer = s2.S2RegionCoverer()
    coverer.set_min_level(level)
    coverer.set_max_level(level)
    cells = [
        np.array([cell.id() for cell in coverer.GetCovering(region)], np.uint64)
        for region in regions
    ]
    return count_cells(cells)


def healpix_ring_ids(points, level):
    import polypix

    return polypix.cell_at(points, resolution=level)


def healpix_nested_ids(points, level):
    import hpgeom

    return hpgeom.vector_to_pixel(2**level, *points.T, nest=True)


def h3_ids(points, level):
    from h3.api import numpy_int as h3

    lat, lon = latlng(points)
    return np.array(
        [h3.latlng_to_cell(*p, level) for p in zip(lat, lon, strict=True)], np.uint64
    )


def s2_ids(points, level):
    import s2geometry as s2

    return np.array(
        [s2.S2CellId(s2.S2Point_FromRaw(*p)).parent(level).id() for p in points],
        np.uint64,
    )


@dataclass(frozen=True)
class Library:
    modules: tuple[str, ...]  # imported before timing
    cover: Callable
    cell_ids: Callable  # untimed: grid cell of each evaluation point
    cell_count: Callable[[int], int]
    modes: tuple[str, ...]


def h3_cells(level):
    from h3.api import numpy_int as h3

    return h3.get_num_cells(level)


def s2_cells(level):
    return 6 * 4**level


def healpix_cells(level):
    import polypix

    return polypix.cell_count(level)


def healpix(cover, modes, cell_ids=healpix_ring_ids, modules=("polypix",)):
    """A library on the HEALPix grid; Polypix is needed for cell counts."""
    return Library(modules, cover, cell_ids, healpix_cells, modes)


LIBRARIES = {
    "polypix": healpix(polypix_cover, ("center", "overlap")),
    "polypix-threaded": healpix(
        partial(polypix_cover, threads=None), ("center", "overlap")
    ),
    "healpy": healpix(healpy_cover, ("center",), modules=("polypix", "healpy")),
    "hpgeom": healpix(hpgeom_cover, ("center",), modules=("polypix", "hpgeom")),
    "mocpy": healpix(
        mocpy_cover,
        ("overlap",),
        cell_ids=healpix_nested_ids,
        modules=("polypix", "mocpy", "astropy.units", "hpgeom"),
    ),
    "h3": Library(
        ("h3.api.numpy_int",), h3_cover, h3_ids, h3_cells, ("center", "overlap")
    ),
    "s2": Library(("s2geometry",), s2_cover, s2_ids, s2_cells, ("overlap",)),
}


@dataclass(frozen=True)
class Case:
    library: str
    workload: str
    mode: str
    level: int
    tracks: int
    samples: int  # healpix-large passes 0 to skip the accuracy check
    seed: int
    iterations: int


def lookup(result, ids: np.ndarray) -> np.ndarray:
    """Read counts at the given cells from a dense array or sparse pair."""
    if isinstance(result, np.ndarray):
        return result[ids]
    cells, counts = result
    if not len(cells):
        return np.zeros(len(ids), dtype=counts.dtype)
    index = np.minimum(np.searchsorted(cells, ids), len(cells) - 1)
    return np.where(cells[index] == ids, counts[index], 0)


def measure(case: Case) -> dict:
    """Time one case; runs in a fresh process."""
    library = LIBRARIES[case.library]
    for module in library.modules:
        importlib.import_module(module)
    scene = make_scene(case.tracks, case.seed)

    times = []
    for _ in range(case.iterations + 1):
        start = time.perf_counter()
        result = library.cover(scene, case.level, case.mode, case.workload)
        times.append(time.perf_counter() - start)

    record = asdict(case) | {
        "cells": library.cell_count(case.level),
        "shapes": len(
            scene.centers if case.workload == "visibility" else scene.footprints
        ),
        "first_seconds": times[0],
        "warm_seconds": times[1:],
    }
    if case.library in ("polypix-threaded", "healpy", "hpgeom"):
        reference = polypix_cover(scene, case.level, case.mode, case.workload)
        if not np.array_equal(result, reference):
            raise AssertionError(f"{case.library} and Polypix select different cells")
    if case.samples:
        rng = np.random.default_rng(case.seed + 1)
        points = rng.normal(size=(case.samples, 3))
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        found = lookup(result, library.cell_ids(points, case.level))
        if case.workload == "observation":
            found = (found > 0).astype(np.int64)
        error = found - exact_counts(points, scene, case.workload)
        record["mean_abs_error"] = float(np.mean(np.abs(error)))
        record["mean_error"] = float(np.mean(error))
    return record


def cross_grid_cases(args) -> list[Case]:
    """Every library on its own grid, from about 3,000 to 800,000 cells."""
    levels = {
        "polypix": range(4, 9),
        "healpy": range(4, 9),
        "hpgeom": range(4, 9),
        "mocpy": range(4, 9),
        "h3": range(2, 5),
        "s2": range(5, 9),
    }
    if args.quick:
        levels = {name: [min(lv) + 1] for name, lv in levels.items()}
    shared = (args.tracks, args.samples, args.seed, args.iterations)
    return [
        Case(name, workload, mode, level, *shared)
        for name, name_levels in levels.items()
        for mode in LIBRARIES[name].modes
        for workload in ("visibility", "observation")
        for level in name_levels
    ]


def healpix_large_cases(args) -> list[Case]:
    """Many more shapes, HEALPix libraries only, exact cell-for-cell agreement."""
    tracks = {"visibility": 10_000, "observation": 1_000}
    if args.quick:
        tracks = {"visibility": 100, "observation": 10}
    return [
        Case(name, workload, "center", level, count, 0, args.seed, args.iterations)
        for name in ("polypix", "polypix-threaded", "healpy", "hpgeom")
        for workload, count in tracks.items()
        for level in (5, 6, 7)
    ]


SUITES = {"cross-grid": cross_grid_cases, "healpix-large": healpix_large_cases}


def environment() -> dict:
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        names = [
            line.split(":", 1)[1].strip()
            for line in cpuinfo.read_text().splitlines()
            if line.startswith("model name")
        ]
        cpu = names[0] if names else cpu
    versions = {}
    for package in "numpy healpy hpgeom mocpy astropy h3 s2geometry".split():
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    # Read from the module itself: it may be a local build with no metadata.
    versions["polypix"] = importlib.import_module("polypix").__version__
    git = ["git", "-C", str(RESULTS.parent)]
    return {
        "commit": subprocess.run(
            [*git, "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "uncommitted_changes": bool(
            subprocess.run(
                [*git, "status", "--porcelain"], capture_output=True, text=True
            ).stdout.strip()
        ),
        "platform": platform.platform(),
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "versions": versions,
    }


def run_suite(name: str, args, machine: dict) -> bool:
    cases = SUITES[name](args)
    records = []
    for case in cases:
        for repeat in range(args.repeats):
            # A fresh process per repeat, so every first call is a real first call.
            with ProcessPoolExecutor(1, mp_context=get_context("spawn")) as pool:
                try:
                    record = pool.submit(measure, case).result()
                except Exception:
                    record = asdict(case) | {"error": traceback.format_exc()}
            records.append(record | {"repeat": repeat})
            seconds = statistics.median(record.get("warm_seconds") or [float("nan")])
            print(
                f"{name}: {case.library} {case.workload} {case.mode} level {case.level}"
                f" #{repeat + 1}: {seconds * 1e3:.2f} ms"
                + (" FAILED" if "error" in record else ""),
                file=sys.stderr,
            )
    report = {"suite": name, "environment": machine, "records": records}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{name}.json").write_text(json.dumps(report, indent=1) + "\n")
    return all("error" not in record for record in records)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--suite", choices=[*SUITES, "all"], default="all")
    parser.add_argument("--tracks", type=int, help="cross-grid shapes")
    parser.add_argument("--samples", type=int, help="accuracy points")
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--repeats", type=int, help="processes per case")
    parser.add_argument("--iterations", type=int, help="warm calls per process")
    parser.add_argument("--out", type=Path, help="default: results/, or results/quick")
    parser.add_argument("--quick", action="store_true", help="tiny smoke test")
    args = parser.parse_args()
    defaults = {"tracks": 128, "samples": 100_000, "repeats": 5, "iterations": 5}
    if args.quick:
        defaults = {"tracks": 8, "samples": 2_000, "repeats": 1, "iterations": 1}
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    args.out = args.out or (RESULTS / "quick" if args.quick else RESULTS)
    for name in ("tracks", "samples", "repeats", "iterations"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be at least 1")

    # Keep NumPy's BLAS on one thread; children inherit this before importing it.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    machine = environment()  # before any results are written to the checkout
    suites = list(SUITES) if args.suite == "all" else [args.suite]
    ok = [run_suite(name, args, machine) for name in suites]
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
