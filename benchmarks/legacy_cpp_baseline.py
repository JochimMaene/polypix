"""Compare the current public API with Polypix's released v0.2.1 C++ API.

This driver intentionally runs unchanged from either checkout. It reuses the
fixed scorecard fixtures, adapts only the renamed API and v0.2.1's packed cell
tokens, and times complete public calls. See docs/development.md for the
two-checkout procedure.
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


def _packed_cells(cells: np.ndarray, resolution: int) -> np.ndarray:
    prefix = np.uint64(1 << (4 + 2 * resolution))
    return np.ascontiguousarray(cells, dtype=np.uint64) | prefix


def _standard_result(result: object, resolution: int) -> CoverageResult:
    if hasattr(result, "cells"):
        cells = np.asarray(result.cells, dtype=np.uint64)
    else:
        prefix = np.uint64(1 << (4 + 2 * resolution))
        cells = np.asarray(result.cell_ids, dtype=np.uint64) ^ prefix
    return CoverageResult(
        np.ascontiguousarray(cells),
        np.ascontiguousarray(result.offsets, dtype=np.uint64),
    )


def _runner(workload: Workload, threads: int | None) -> Callable[[], CoverageResult]:
    is_current = hasattr(px, "cover_strip")
    candidates = workload.candidate_cells
    if candidates is not None and not is_current:
        candidates = _packed_cells(candidates, workload.resolution)

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
    implementation = "rust-cds" if is_current else "legacy-cpp"
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
