"""Coarse native wall-time guard for the parallel candidate path."""

from __future__ import annotations

import os
import statistics
import time

import numpy as np

import polypix as px


def _available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _footprints(count: int) -> np.ndarray:
    side = int(round(count**0.5))
    if side * side != count:
        raise ValueError("footprint count must be a square")
    longitudes, latitudes = np.meshgrid(
        np.linspace(-170.0, 170.0, side),
        np.linspace(-60.0, 60.0, side),
    )
    longitudes = longitudes.ravel()
    latitudes = latitudes.ravel()
    half_lon = np.deg2rad(0.05)
    half_lat = np.deg2rad(0.04)
    longitude = np.deg2rad(longitudes)[:, None] + np.array(
        [-half_lon, half_lon, half_lon, -half_lon]
    )
    latitude = np.deg2rad(latitudes)[:, None] + np.array(
        [-half_lat, -half_lat, half_lat, half_lat]
    )
    cos_latitude = np.cos(latitude)
    return np.stack(
        (
            cos_latitude * np.cos(longitude),
            cos_latitude * np.sin(longitude),
            np.sin(latitude),
        ),
        axis=-1,
    )


def _seconds(
    footprints: np.ndarray, candidates: np.ndarray, threads: int | None
) -> float:
    started = time.perf_counter()
    coverage = px.cover_footprint(
        footprints,
        12,
        candidate_cells=candidates,
        threads=threads,
    )
    elapsed = time.perf_counter() - started
    if coverage.offsets.shape != (footprints.shape[0] + 1,):
        raise AssertionError("parallel smoke workload returned malformed offsets")
    return elapsed


def main() -> None:
    cpu_count = _available_cpus()
    if cpu_count < 2:
        print("Skipping parallel scaling check: fewer than two CPUs are available.")
        return

    footprints = _footprints(4_096)
    pixel_count = 12 * 4**12
    candidates = np.arange(2_000_000, dtype=np.uint64) * np.uint64(
        pixel_count // 2_000_000
    )

    # Warm the reusable pool and native code before comparing steady state.
    serial_result = px.cover_footprint(
        footprints, 12, candidate_cells=candidates, threads=1
    )
    automatic_result = px.cover_footprint(footprints, 12, candidate_cells=candidates)
    np.testing.assert_array_equal(automatic_result.cells, serial_result.cells)
    np.testing.assert_array_equal(automatic_result.offsets, serial_result.offsets)

    serial_samples = []
    automatic_samples = []
    for iteration in range(6):
        order = (1, None) if iteration % 2 == 0 else (None, 1)
        for threads in order:
            elapsed = _seconds(footprints, candidates, threads)
            (automatic_samples if threads is None else serial_samples).append(elapsed)
    serial = statistics.median(serial_samples)
    automatic = statistics.median(automatic_samples)
    speedup = serial / automatic
    print(
        f"Native candidate scaling on {cpu_count} CPUs: "
        f"{serial * 1e3:.1f} ms serial, {automatic * 1e3:.1f} ms automatic, "
        f"{speedup:.2f}x"
    )
    if speedup < 1.25:
        raise AssertionError(
            "automatic candidate coverage must be at least 1.25x faster "
            f"than serial on a multi-CPU runner; measured {speedup:.2f}x"
        )


if __name__ == "__main__":
    main()
