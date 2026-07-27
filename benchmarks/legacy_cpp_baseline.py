"""Compare the current public API with Polypix's released v0.2.1 C++ API.

This driver intentionally runs unchanged from either checkout. It reuses the
fixed scorecard fixtures, adapts the renamed API and v0.2.1's packed NESTED
tokens to the current RING contract, and times complete public calls. See
docs/development.md for the two-checkout procedure.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

import polypix as px

try:
    from benchmarks.scorecard import CoverageResult, Workload, build_workloads
except ModuleNotFoundError:
    # An absolute invocation from the detached v0.2.1 checkout does not put the
    # current repository root on sys.path. Load the adjacent fixture module
    # without making the current polypix package importable.
    scorecard_path = Path(__file__).with_name("scorecard.py")
    spec = importlib.util.spec_from_file_location("_polypix_scorecard", scorecard_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scorecard fixtures from {scorecard_path}")
    scorecard = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = scorecard
    spec.loader.exec_module(scorecard)
    CoverageResult = scorecard.CoverageResult
    Workload = scorecard.Workload
    build_workloads = scorecard.build_workloads


BASELINE_COMMIT = "20d2df6"


def _compact_bits(value: int) -> int:
    result = 0
    bit = 0
    while value:
        result |= (value & 1) << bit
        value >>= 2
        bit += 1
    return result


def _nested_to_ring(cells: np.ndarray, resolution: int) -> np.ndarray:
    """Convert ordinary fixed-resolution NESTED IDs to RING IDs."""

    nside = 1 << resolution
    face_size = nside * nside
    pixel_count = 12 * face_size
    cap_cells = 2 * nside * (nside - 1)
    ring_latitudes = (2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4)
    ring_longitudes = (1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7)
    result = np.empty(cells.size, dtype=np.uint64)

    for output_index, raw_cell in enumerate(cells):
        face, nested = divmod(int(raw_cell), face_size)
        x = _compact_bits(nested)
        y = _compact_bits(nested >> 1)
        ring = ring_latitudes[face] * nside - x - y - 1
        if ring < nside:
            cells_on_face = ring
            preceding = 2 * ring * (ring - 1)
            shift = 0
        elif ring > 3 * nside:
            cells_on_face = 4 * nside - ring
            preceding = pixel_count - 2 * cells_on_face * (cells_on_face + 1)
            shift = 0
        else:
            cells_on_face = nside
            preceding = cap_cells + (ring - nside) * 4 * nside
            shift = (ring - nside) & 1
        longitude = (
            ring_longitudes[face] * cells_on_face + x - y + 1 + shift
        ) // 2
        result[output_index] = preceding + (longitude - 1) % (4 * cells_on_face)
    return result


def _standard_result(result: object, resolution: int) -> CoverageResult:
    if hasattr(result, "cells"):
        cells = np.asarray(result.cells, dtype=np.uint64)
    else:
        prefix = np.uint64(1 << (4 + 2 * resolution))
        nested = np.asarray(result.cell_ids, dtype=np.uint64) ^ prefix
        cells = _nested_to_ring(nested, resolution)
    return CoverageResult(
        np.ascontiguousarray(cells),
        np.ascontiguousarray(result.offsets, dtype=np.uint64),
    )


def _runner(workload: Workload, threads: int | None) -> Callable[[], CoverageResult]:
    is_current = hasattr(px, "cover_strip")
    candidates = workload.candidate_cells
    if candidates is not None and not is_current:
        raise RuntimeError(
            "v0.2.1 accepted packed NESTED candidates, while the current "
            "scorecard fixture is RING"
        )

    if workload.kind == "strip":
        assert workload.left_edge is not None and workload.right_edge is not None

        def run_strip() -> CoverageResult:
            if is_current:
                result = px.cover_strip(
                    workload.left_edge,
                    workload.right_edge,
                    workload.resolution,
                    candidate_cells=candidates,
                    threads=threads,
                )
            else:
                result = px.cover_swath(
                    workload.left_edge,
                    workload.right_edge,
                    workload.resolution,
                    allowed_cell_ids=candidates,
                )
            return _standard_result(result, workload.resolution)

        return run_strip

    assert workload.footprints is not None
    if isinstance(workload.footprints, tuple) and not is_current:
        raise RuntimeError("v0.2.1 did not support ragged footprint batches")

    def run_footprints() -> CoverageResult:
        if is_current:
            result = px.cover_footprint(
                workload.footprints,
                workload.resolution,
                candidate_cells=candidates,
                threads=threads,
            )
        else:
            result = px.cover_footprint(
                workload.footprints,
                workload.resolution,
                allowed_cell_ids=candidates,
            )
        return _standard_result(result, workload.resolution)

    return run_footprints


def _digest(result: CoverageResult) -> str:
    digest = hashlib.sha256()
    for start, end in zip(result.offsets[:-1], result.offsets[1:], strict=True):
        digest.update(np.sort(result.cells[int(start) : int(end)]).tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def benchmark(
    *,
    profile: str,
    workload_names: set[str] | None,
    threads: int | None,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    old_thread_setting = os.environ.get("POLYPIX_NUM_THREADS")
    is_current = hasattr(px, "cover_strip")
    implementation = "rust-owned-ring" if is_current else "legacy-cpp"
    if not is_current:
        os.environ["POLYPIX_NUM_THREADS"] = str(threads) if threads is not None else ""

    records: list[dict[str, Any]] = []
    try:
        for workload in build_workloads(profile):
            if workload_names is not None and workload.name not in workload_names:
                continue
            try:
                run = _runner(workload, threads)
                for _ in range(warmup):
                    run()
                samples: list[int] = []
                result: CoverageResult | None = None
                for _ in range(repeats):
                    start = time.perf_counter_ns()
                    result = run()
                    samples.append(time.perf_counter_ns() - start)
                assert result is not None
                median = statistics.median(samples)
                records.append(
                    {
                        "workload": workload.name,
                        "status": "ok",
                        "samples_ns": samples,
                        "median_ns": median,
                        "items_per_second": workload.item_count / (median / 1e9),
                        "cell_count": int(result.cells.size),
                        "membership_sha256": _digest(result),
                    }
                )
            except RuntimeError as exc:
                records.append(
                    {
                        "workload": workload.name,
                        "status": "unsupported",
                        "detail": str(exc),
                    }
                )
    finally:
        if not is_current:
            if old_thread_setting is None:
                os.environ.pop("POLYPIX_NUM_THREADS", None)
            else:
                os.environ["POLYPIX_NUM_THREADS"] = old_thread_setting

    return {
        "baseline_commit": BASELINE_COMMIT,
        "implementation": implementation,
        "polypix_version": px.__version__,
        "profile": profile,
        "threads": "auto" if threads is None else threads,
        "warmup": warmup,
        "repeats": repeats,
        "results": records,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "standard"), default="standard")
    parser.add_argument("--workloads", nargs="*")
    parser.add_argument("--threads", default="auto")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    threads = None if args.threads == "auto" else int(args.threads)
    report = benchmark(
        profile=args.profile,
        workload_names=set(args.workloads) if args.workloads else None,
        threads=threads,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
