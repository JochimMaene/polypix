from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np

from . import cover_footprint


def _load_footprints(path: Path) -> np.ndarray:
    payload = np.load(path)
    if isinstance(payload, np.lib.npyio.NpzFile):
        if "footprints" in payload:
            payload = np.asarray(payload["footprints"], dtype=np.float64)
        else:
            raise ValueError("NPZ footprint files must contain a 'footprints' array.")
    if isinstance(payload, np.ndarray):
        footprints = np.ascontiguousarray(np.asarray(payload, dtype=np.float64))
        if footprints.ndim != 3 or footprints.shape[-1] != 3:
            raise ValueError("Dense footprint files must have shape (footprints, vertices, 3).")
        return footprints
    raise ValueError(f"Unsupported footprint file: {path}")


def _benchmark(
    footprints: np.ndarray,
    resolution: int,
    warmup: int,
    repeats: int,
) -> tuple[float, float, int]:
    def runner():
        return cover_footprint(footprints, resolution)

    for _ in range(warmup):
        runner()

    timings: list[float] = []
    result = None
    footprint_count = footprints.shape[0]
    for _ in range(repeats):
        started = time.perf_counter()
        result = runner()
        timings.append(time.perf_counter() - started)

    mean_seconds = statistics.fmean(timings)
    return footprint_count / mean_seconds, mean_seconds, result.cell_ids.size


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Polypix on a footprint batch.")
    parser.add_argument("--footprints-file", required=True)
    parser.add_argument("--resolution", type=int, nargs="+", required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    footprints = _load_footprints(Path(args.footprints_file))
    print("resolution\tfootprint_s\tmean_s\ttotal_cells")
    for resolution in args.resolution:
        footprint_s, mean_seconds, total_cells = _benchmark(
            footprints, resolution, args.warmup, args.repeats
        )
        print(f"{resolution}\t{footprint_s:.6f}\t{mean_seconds:.6f}\t{total_cells}")


if __name__ == "__main__":
    main()
