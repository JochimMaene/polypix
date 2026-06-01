from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np

from . import _cover_flat


def _load_polygons(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path)
    if isinstance(payload, np.lib.npyio.NpzFile):
        if "vertices" in payload and "polygon_offsets" in payload:
            vertices = np.ascontiguousarray(np.asarray(payload["vertices"], dtype=np.float64))
            offsets = np.ascontiguousarray(np.asarray(payload["polygon_offsets"], dtype=np.uint64))
            if vertices.ndim != 2 or vertices.shape[1] != 3:
                raise ValueError("NPZ vertices must have shape (vertices, 3).")
            return vertices, offsets
        if "polygons" in payload:
            payload = np.asarray(payload["polygons"], dtype=np.float64)
        else:
            raise ValueError("NPZ polygon files must contain either 'vertices' + 'polygon_offsets' or 'polygons'.")
    if isinstance(payload, np.ndarray):
        polygons = np.ascontiguousarray(np.asarray(payload, dtype=np.float64))
        if polygons.ndim != 3 or polygons.shape[-1] != 3:
            raise ValueError("Dense polygon files must have shape (polygons, vertices, 3).")
        polygon_count, vertex_count, _dims = polygons.shape
        offsets = np.arange(0, (polygon_count + 1) * vertex_count, vertex_count, dtype=np.uint64)
        return polygons.reshape(polygon_count * vertex_count, 3), offsets
    raise ValueError(f"Unsupported polygon file: {path}")


def _benchmark(
    vertices: np.ndarray,
    offsets: np.ndarray,
    resolution: int,
    warmup: int,
    repeats: int,
) -> tuple[float, float, int]:
    runner = lambda: _cover_flat(vertices, resolution, offsets)

    for _ in range(warmup):
        runner()

    timings: list[float] = []
    result = None
    polygon_count = offsets.size - 1
    for _ in range(repeats):
        started = time.perf_counter()
        result = runner()
        timings.append(time.perf_counter() - started)

    mean_seconds = statistics.fmean(timings)
    return polygon_count / mean_seconds, mean_seconds, int(result[0].size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Polypix on a polygon batch.")
    parser.add_argument("--polygons-file", required=True)
    parser.add_argument("--resolution", type=int, nargs="+", required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    vertices, offsets = _load_polygons(Path(args.polygons_file))
    print("resolution\tpoly_s\tmean_s\ttotal_cells")
    for resolution in args.resolution:
        poly_s, mean_seconds, total_cells = _benchmark(
            vertices, offsets, resolution, args.warmup, args.repeats
        )
        print(f"{resolution}\t{poly_s:.6f}\t{mean_seconds:.6f}\t{total_cells}")


if __name__ == "__main__":
    main()
