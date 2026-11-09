"""Small independent checks for the optional benchmark harness."""

import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "competitors", Path(__file__).parents[1] / "benchmarks" / "competitors.py"
)
assert spec is not None and spec.loader is not None
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_scene_and_analytic_reference():
    centers, radii, polygons = bench.scene(3, 42)
    again = bench.scene(3, 42)
    for a, b in zip((centers, radii, polygons), again, strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(np.linalg.norm(polygons, axis=2), 1)
    points = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    np.testing.assert_array_equal(
        bench.truth(points, points[:1], np.array([0.1]), polygons, "visibility"), [1, 0]
    )
    center = polygons[0].sum(axis=0)
    center /= np.linalg.norm(center)
    np.testing.assert_array_equal(
        bench.truth(
            np.array([center, -center]), centers, radii, polygons[:1], "observation"
        ),
        [1, 0],
    )


def test_polypix_serial_auto_equivalent():
    import polypix as px

    centers, radii, polygons = bench.scene(4, 42)
    for workflow in ("visibility", "observation"):
        serial = bench.execute(
            "polypix-serial", px, 4, centers, radii, polygons, workflow
        )
        auto = bench.execute("polypix-auto", px, 4, centers, radii, polygons, workflow)
        np.testing.assert_array_equal(serial, auto)
        points = px.cell_centers(np.arange(px.cell_count(4)), resolution=4)
        np.testing.assert_array_equal(
            serial, bench.truth(points, centers, radii, polygons, workflow)
        )


def test_worker_records_cold_and_warm_measurements():
    import argparse
    import sys

    import pytest

    if sys.platform == "win32":
        pytest.skip("RSS measurement requires Unix resource module")
    args = argparse.Namespace(
        regions=3,
        seed=42,
        samples=100,
        backend="polypix-serial",
        order=3,
        workflow="visibility",
        iterations=2,
    )
    result = bench.worker(args)
    assert result["cold_seconds"] > 0
    assert len(result["warm_seconds"]) == 2
    assert result["seconds"] == np.median(result["warm_seconds"])
    assert result["peak_process_rss_bytes"] > 0
    assert 0 <= result["sample_disagreement_fraction"] <= 1


def test_published_matrix_covers_sizes_and_resolutions(monkeypatch, tmp_path):
    import runpy
    import subprocess

    calls = []
    (tmp_path / "benchmarks").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        subprocess, "run", lambda command, **kwargs: calls.append(command)
    )
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "benchmarks/run_comparisons.py")
    )
    assert len(calls) == 9
    assert {command[command.index("--order") + 1] for command in calls} == {
        "5",
        "6",
        "7",
    }
    large = [command for command in calls if "--regions" in command]
    assert [command[command.index("--regions") + 1] for command in large] == [
        "10000"
    ] * 3 + ["1000"] * 3


def test_chart_summary_uses_process_medians_and_rejects_missing_data():
    import pytest

    pytest.importorskip("matplotlib")
    spec = importlib.util.spec_from_file_location(
        "plots", Path(__file__).parents[1] / "benchmarks/plot_comparisons.py"
    )
    assert spec is not None and spec.loader is not None
    plots = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plots)
    report = {
        "records": [
            {"backend": "a", "workflow": "b", "seconds": x, "cold_seconds": x * 2}
            for x in (3, 1, 2)
        ]
    }
    assert plots.summarize(report, "a", "b") == (2, 1, 3)
    assert plots.summarize(report, "a", "b", True) == (4, 2, 6)
    with pytest.raises(ValueError, match="Missing"):
        plots.summarize(report, "missing", "b")
