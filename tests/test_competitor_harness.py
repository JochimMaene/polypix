"""Checks for the competitor benchmark that only need Polypix installed."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

import polypix as px

spec = importlib.util.spec_from_file_location(
    "competitors", Path(__file__).parents[1] / "benchmarks" / "competitors.py"
)
assert spec is not None and spec.loader is not None
bench = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bench  # dataclasses look their module up here
spec.loader.exec_module(bench)


def test_exact_counts_agree_with_polypix_at_cell_centers():
    scene = bench.make_scene(6, seed=1)
    centers = px.cell_centers(np.arange(px.cell_count(5)), resolution=5)
    for workload in ("visibility", "observation"):
        expected = bench.exact_counts(centers, scene, workload)
        found = bench.polypix_cover(scene, 5, "center", workload)
        if workload == "observation":
            found = found > 0
        np.testing.assert_array_equal(found, expected)


def test_lookup_reads_dense_and_sparse_results():
    ids = np.array([3, 5, 9], dtype=np.uint64)
    sparse = (np.array([1, 5, 7], dtype=np.uint64), np.array([2, 4, 6]))
    np.testing.assert_array_equal(bench.lookup(sparse, ids), [0, 4, 0])
    np.testing.assert_array_equal(bench.lookup(np.arange(10) * 2, ids), [6, 10, 18])


def test_measure_reports_timings_and_accuracy():
    case = bench.Case("polypix", "visibility", "center", 4, 4, 2000, 1, 2)
    record = bench.measure(case)
    assert len(record["warm_seconds"]) == 2
    assert record["cells"] == px.cell_count(4)
    assert 0 <= record["mean_abs_error"] < 0.1
