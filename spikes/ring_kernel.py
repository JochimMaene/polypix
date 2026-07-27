"""PROTOTYPE: compare direct HEALPix ring scan-conversion with CDS/BMOC.

Question: can the owned native algorithm make a 10x product speedup credible
while returning exactly the same center memberships? Run with:

    pixi run ring-prototype

This is throwaway benchmark scaffolding, not a public Polypix API.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import numpy as np

from benchmarks.scorecard import Workload, build_workloads, strip_footprints
from polypix._core import _cover, _cover_ring_prototype


def thread_count(value: str) -> int | None:
    if value == "auto":
        return None
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("threads must be 'auto' or a positive integer")
    return count


def dense_input(workload: Workload) -> tuple[np.ndarray, np.ndarray]:
    if workload.kind == "strip":
        assert workload.left_edge is not None and workload.right_edge is not None
        footprints = strip_footprints(workload.left_edge, workload.right_edge)
    else:
        assert workload.footprints is not None
        footprints = workload.footprints
    if isinstance(footprints, tuple):
        counts = np.asarray([len(footprint) for footprint in footprints], dtype=np.uint64)
        return (
            np.ascontiguousarray(np.concatenate(footprints, axis=0)),
            np.concatenate(
                (np.zeros(1, dtype=np.uint64), np.cumsum(counts, dtype=np.uint64))
            ),
        )
    if footprints.ndim == 2:
        footprints = footprints[np.newaxis, :, :]
    count, vertices, _ = footprints.shape
    return (
        np.ascontiguousarray(footprints.reshape(count * vertices, 3)),
        np.arange(0, count * vertices + 1, vertices, dtype=np.uint64),
    )


def timed(function: Callable[[], dict], repeats: int) -> tuple[float, dict]:
    for _ in range(2):
        function()
    samples = []
    result = {}
    for _ in range(repeats):
        started = time.perf_counter_ns()
        result = function()
        samples.append(time.perf_counter_ns() - started)
    return float(statistics.median(samples)), result


def same_segments(reference: dict, candidate: dict) -> bool:
    if not np.array_equal(reference["offsets"], candidate["offsets"]):
        return False
    for start, end in zip(
        reference["offsets"][:-1], reference["offsets"][1:], strict=True
    ):
        segment = slice(int(start), int(end))
        if not np.array_equal(
            np.sort(reference["cells"][segment]),
            np.sort(candidate["cells"][segment]),
        ):
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "standard"), default="standard")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--threads", type=thread_count, default=1)
    args = parser.parse_args()

    print("PROTOTYPE — direct RING center scan-conversion")
    print("Target: >=15x native margin and exact NESTED membership")
    print(f"profile={args.profile} threads={args.threads} repeats={args.repeats}")
    print()
    successful = True
    for workload in build_workloads(args.profile):
        if workload.candidate_cells is not None:
            continue
        vertices, offsets = dense_input(workload)
        resolution = workload.resolution
        reference_time, reference = timed(
            lambda: _cover(vertices, offsets, resolution, None, args.threads),
            args.repeats,
        )
        nested_time, nested = timed(
            lambda: _cover_ring_prototype(
                vertices, offsets, resolution, args.threads, True
            ),
            args.repeats,
        )
        ring_time, ring = timed(
            lambda: _cover_ring_prototype(
                vertices, offsets, resolution, args.threads, False
            ),
            args.repeats,
        )
        correct = same_segments(reference, nested)
        speedup = reference_time / ring_time
        conversion_tax = nested_time / ring_time - 1.0
        successful &= correct
        print(
            f"{workload.name:30} "
            f"cds={reference_time / 1e6:9.3f} ms  "
            f"ring={ring_time / 1e6:9.3f} ms  "
            f"speedup={speedup:6.2f}x  "
            f"nested_tax={conversion_tax:6.1%}  "
            f"cells={len(ring['cells']):8}  "
            f"correct={correct}"
        )
    return 0 if successful else 2


if __name__ == "__main__":
    raise SystemExit(main())
