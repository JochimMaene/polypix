"""Reproducible application benchmarks; see competitors.md for semantics."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BACKENDS = (
    "polypix-serial",
    "polypix-auto",
    "healpy",
    "hpgeom",
    "mocpy",
    "h3",
    "s2geometry",
)


def lonlat(xyz):
    return np.degrees(np.arctan2(xyz[..., 1], xyz[..., 0])), np.degrees(
        np.arcsin(xyz[..., 2])
    )


def scene(count, seed):
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-np.pi, np.pi, count)
    lat = rng.uniform(-1.1, 1.1, count)
    centers = np.column_stack(
        (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat))
    )
    east = np.column_stack((-np.sin(lon), np.cos(lon), np.zeros(count)))
    north = np.cross(centers, east)
    radii = rng.uniform(np.radians(3), np.radians(12), count)
    # Consecutive overlapping footprints along each synthetic track.
    polygons = []
    for c, e, n in zip(centers, east, north, strict=True):
        for step in range(4):
            p = np.array(
                [
                    c + (step * 0.015 + x) * e + y * n
                    for x, y in [
                        (-0.012, -0.008),
                        (0.012, -0.008),
                        (0.012, 0.008),
                        (-0.012, 0.008),
                    ]
                ]
            )
            polygons.append(p / np.linalg.norm(p, axis=1)[:, None])
    return centers, radii, np.array(polygons)


def truth(points, centers, radii, polygons, workflow):
    result = np.zeros(len(points), dtype=np.int64)
    if workflow == "visibility":
        for c, r in zip(centers, radii, strict=True):
            result += points @ c >= np.cos(r)
    else:
        for p in polygons:
            normals = np.cross(p, np.roll(p, -1, axis=0))
            result |= np.all(points @ normals.T >= -1e-14, axis=1)
    return result


def prepare(backend, order, points):
    """Import and prepare evaluation cells outside the timed region."""
    if backend.startswith("polypix"):
        import polypix as lib

        return lib, order, lib.cell_at(points, resolution=order)
    if backend == "healpy":
        import healpy as lib

        return lib, order, lib.vec2pix(2**order, *points.T)
    if backend == "hpgeom":
        import hpgeom as lib

        return lib, order, lib.vector_to_pixel(2**order, *points.T, nest=False)
    if backend == "mocpy":
        import mocpy as lib

        return lib, order, None
    if backend == "h3":
        import h3 as lib

        level = min(
            range(16),
            key=lambda x: abs(math.log(lib.get_num_cells(x) / (12 * 4**order))),
        )
        lon, lat = lonlat(points)
        return (
            lib,
            level,
            [lib.latlng_to_cell(y, x, level) for x, y in zip(lon, lat, strict=True)],
        )
    import s2geometry as lib

    level = min(range(31), key=lambda x: abs(math.log((6 * 4**x) / (12 * 4**order))))
    ids = [
        lib.S2CellId(lib.S2LatLng.FromDegrees(y, x)).parent(level).id()
        for x, y in zip(*lonlat(points), strict=True)
    ]
    return lib, level, ids


def execute(backend, lib, level, centers, radii, polygons, workflow):
    """Return counts or a union on the backend's own grid."""
    if backend.startswith("polypix"):
        kw = dict(
            resolution=level,
            threads=1 if backend.endswith("serial") else None,
            reduce=lib.Count(),
        )
        counts = (
            lib.cover_cap(centers, radii, **kw)
            if workflow == "visibility"
            else lib.cover_polygon(polygons, **kw)
        )
        return counts if workflow == "visibility" else counts > 0
    if backend in ("healpy", "hpgeom"):
        counts = np.zeros(12 * 4**level, dtype=np.int64)
        if workflow == "visibility":
            if backend == "healpy":
                regions = (
                    lib.query_disc(2**level, c, r)
                    for c, r in zip(centers, radii, strict=True)
                )
            else:
                regions = (
                    lib.query_circle_vec(2**level, c, r, nest=False)
                    for c, r in zip(centers, radii, strict=True)
                )
        else:
            regions = (
                lib.query_polygon(2**level, p)
                if backend == "healpy"
                else lib.query_polygon_vec(2**level, p, nest=False)
                for p in polygons
            )
        for cells in regions:
            counts[cells] += 1
        return counts if workflow == "visibility" else counts > 0
    if backend == "mocpy":
        import astropy.units as u

        if workflow == "visibility":
            lon, lat = lonlat(centers)
            return lib.MOC.from_cones(
                lon=lon * u.deg, lat=lat * u.deg, radius=radii * u.rad, max_depth=level
            )
        lon, lat = lonlat(polygons)
        vertices = np.stack((lon % 360, lat), axis=1).reshape(-1, polygons.shape[1])
        regions = lib.MOC.from_polygons(vertices, max_depth=level)
        return regions[0].union(*regions[1:])
    counts = {}
    if backend == "s2geometry":
        coverer = lib.S2RegionCoverer()
        coverer.set_min_level(level)
        coverer.set_max_level(level)
    shapes = zip(centers, radii, strict=True) if workflow == "visibility" else polygons
    for shape in shapes:
        if backend == "h3":
            if workflow == "visibility":
                c, r = shape
                e = np.cross(c, [0.0, 0.0, 1.0])
                e /= np.linalg.norm(e)
                n = np.cross(c, e)
                angles = np.arange(128) * 2 * np.pi / 128
                p = np.cos(r) * c + np.sin(r) * (
                    np.cos(angles)[:, None] * e + np.sin(angles)[:, None] * n
                )
            else:
                p = shape
            lon, lat = lonlat(p)
            cells = lib.polygon_to_cells(
                lib.LatLngPoly(list(zip(lat, lon, strict=True))), level
            )
        else:
            if workflow == "visibility":
                c, r = shape
                x, y = lonlat(c)
                region = lib.S2Cap(
                    lib.S2LatLng.FromDegrees(y, x).ToPoint(),
                    lib.S1Angle.Radians(float(r)),
                )
            else:
                loop = lib.S2Loop(
                    [
                        lib.S2LatLng.FromDegrees(y, x).ToPoint()
                        for x, y in zip(*lonlat(shape), strict=True)
                    ]
                )
                loop.Normalize()
                region = lib.S2Polygon(loop)
            cells = [c.id() for c in coverer.GetCovering(region)]
        for cell in cells:
            counts[cell] = counts.get(cell, 0) + 1 if workflow == "visibility" else 1
    return counts


def evaluate(backend, result, ids, points, workflow):
    if backend == "mocpy":
        import astropy.units as u

        lon, lat = lonlat(points)
        if workflow == "visibility":
            return sum(
                (
                    m.contains_lonlat(lon * u.deg, lat * u.deg).astype(np.int64)
                    for m in result
                ),
                np.zeros(len(points), dtype=np.int64),
            )
        return result.contains_lonlat(lon * u.deg, lat * u.deg).astype(np.int64)
    if isinstance(result, dict):
        return np.array([result.get(cell, 0) for cell in ids])
    return result[ids].astype(np.int64)


def worker(args):
    import resource

    centers, radii, polygons = scene(args.regions, args.seed)
    # Equal-area global samples: the target region is the whole sphere.
    rng = np.random.default_rng(args.seed + 1)
    points = rng.normal(size=(args.samples, 3))
    points /= np.linalg.norm(points, axis=1)[:, None]
    lib, level, ids = prepare(args.backend, args.order, points)
    timings = []
    for iteration in range(args.iterations + 1):
        started = time.perf_counter()
        result = execute(
            args.backend, lib, level, centers, radii, polygons, args.workflow
        )
        actual = evaluate(args.backend, result, ids, points, args.workflow)
        timings.append(time.perf_counter() - started)
        # Do not retain the preceding output during the next allocation.
        if iteration < args.iterations:
            del result, actual
    elapsed = statistics.median(timings[1:])
    # Capture process high-water RSS before validation allocates its own arrays.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss *= 1 if sys.platform == "darwin" else 1024
    expected = truth(points, centers, radii, polygons, args.workflow)
    exact = None
    if args.backend in ("healpy", "hpgeom"):
        import polypix as px

        reference = execute(
            "polypix-serial", px, args.order, centers, radii, polygons, args.workflow
        )
        exact = bool(np.array_equal(result, reference))
        if not exact:
            raise AssertionError("HEALPix outputs differ from Polypix")
    size = (
        12 * 4**level
        if args.backend not in ("h3", "s2geometry")
        else lib.get_num_cells(level)
        if args.backend == "h3"
        else 6 * 4**level
    )
    return dict(
        backend=args.backend,
        workflow=args.workflow,
        level=level,
        global_cell_count=size,
        mean_cell_area_steradians=4 * np.pi / size,
        seconds=elapsed,
        cold_seconds=timings[0],
        warm_seconds=timings[1:],
        peak_process_rss_bytes=rss,
        output_cells=int(np.count_nonzero(result))
        if isinstance(result, np.ndarray)
        else len(result)
        if isinstance(result, dict)
        else None,
        mean_absolute_sample_error=float(np.mean(abs(actual - expected))),
        sample_disagreement_fraction=float(np.mean(actual != expected)),
        estimated_covered_fraction=float(np.mean(actual > 0)),
        reference_covered_fraction=float(np.mean(expected > 0)),
        exact_healpix_match=exact,
        input_sha256=hashlib.sha256(
            centers.tobytes() + radii.tobytes() + polygons.tobytes()
        ).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, default=6)
    parser.add_argument("--regions", type=int, default=128)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--workflows",
        nargs="+",
        choices=("visibility", "observation"),
        default=["visibility", "observation"],
    )
    parser.add_argument(
        "--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS)
    )
    parser.add_argument("--output", type=Path, default=Path("competitor-results.json"))
    parser.add_argument("--backend", choices=BACKENDS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--workflow", choices=("visibility", "observation"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if (
        not 0 <= args.order <= 10
        or min(args.regions, args.samples, args.repeats, args.iterations) < 1
    ):
        parser.error(
            "order must be 0..10; regions, samples and repeats must be positive"
        )
    if args.backend:
        print(json.dumps(worker(args)))
        return
    records = []
    for backend in args.backends:
        for workflow in args.workflows:
            for repeat in range(args.repeats):
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--iterations",
                    str(args.iterations),
                    "--backend",
                    backend,
                    "--workflow",
                    workflow,
                    "--order",
                    str(args.order),
                    "--regions",
                    str(args.regions),
                    "--samples",
                    str(args.samples),
                    "--seed",
                    str(args.seed),
                ]
                env = dict(
                    os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1])
                )
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                ):
                    env[name] = "1"
                run = subprocess.run(cmd, env=env, capture_output=True, text=True)
                record = (
                    json.loads(run.stdout)
                    if run.returncode == 0
                    else dict(backend=backend, workflow=workflow, error=run.stderr)
                )
                record["repeat"] = repeat
                records.append(record)
                print(
                    f"{backend} {workflow} {repeat + 1}: {record.get('seconds', 'ERROR')}",
                    file=sys.stderr,
                )
    versions = {}
    for name in (
        "polypix",
        "numpy",
        "healpy",
        "hpgeom",
        "mocpy",
        "h3",
        "s2geometry",
        "astropy",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    report = dict(
        revision=revision,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        platform=platform.platform(),
        cpu=cpu,
        cpu_count=os.cpu_count(),
        python=sys.version,
        versions=versions,
        configuration=vars(args) | {"output": str(args.output)},
        records=records,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if any("error" in r for r in records):
        sys.exit(1)


if __name__ == "__main__":
    main()
