from __future__ import annotations

import json

import numpy as np

from benchmarks import scorecard


def test_scorecard_fixtures_are_deterministic_and_well_shaped() -> None:
    first = scorecard.build_workloads("smoke")
    second = scorecard.build_workloads("smoke")

    assert [workload.name for workload in first] == [
        workload.name for workload in second
    ]
    for left, right in zip(first, second, strict=True):
        assert left.kind == right.kind
        if isinstance(left.footprints, np.ndarray):
            np.testing.assert_array_equal(left.footprints, right.footprints)
        elif isinstance(left.footprints, tuple):
            assert isinstance(right.footprints, tuple)
            assert {footprint.shape[0] for footprint in left.footprints} == {
                3,
                4,
                5,
                6,
            }
            for left_footprint, right_footprint in zip(
                left.footprints, right.footprints, strict=True
            ):
                np.testing.assert_array_equal(left_footprint, right_footprint)
        if left.left_edge is not None:
            np.testing.assert_array_equal(left.left_edge, right.left_edge)
            np.testing.assert_array_equal(left.right_edge, right.right_edge)
        if left.candidate_cells is not None:
            np.testing.assert_array_equal(left.candidate_cells, right.candidate_cells)
            assert np.unique(left.candidate_cells).size == left.candidate_cells.size


def test_all_generated_vectors_are_normalized_and_convex() -> None:
    for workload in scorecard.build_workloads("smoke"):
        if workload.kind == "strip":
            assert workload.left_edge is not None
            assert workload.right_edge is not None
            footprints = scorecard.strip_footprints(
                workload.left_edge, workload.right_edge
            )
        elif workload.footprints is not None:
            footprints = scorecard._segments(workload.footprints)
        else:
            continue

        for footprint in footprints:
            np.testing.assert_allclose(
                np.linalg.norm(footprint, axis=1),
                1.0,
                rtol=0.0,
                atol=2.0e-15,
            )
            normals = scorecard._oriented_edge_normals(footprint)
            interior = np.sum(footprint, axis=0)
            interior /= np.linalg.norm(interior)
            assert np.all(normals @ interior > 0.0)


def test_adversarial_cases_match_exhaustive_center_oracle() -> None:
    checks = scorecard.run_adversarial_correctness()

    assert {check["name"] for check in checks} == {
        "antimeridian",
        "north_pole",
        "south_pole",
        "center_on_boundary",
        "near_hemisphere_limit",
        "fixed_seed_randomized",
    }
    assert all(check["status"] == "pass" for check in checks)


def test_cds_control_point_regression_uses_the_intended_polygon_side() -> None:
    lon = -45.006968888513825
    lat = -35.65525862937339
    half_lon = 6.474936777280457
    half_lat = 0.5725726026463771
    footprint = np.asarray(
        [
            scorecard.lonlat_to_xyz(lon - half_lon, lat - half_lat),
            scorecard.lonlat_to_xyz(lon + half_lon, lat - half_lat),
            scorecard.lonlat_to_xyz(lon + half_lon, lat + half_lat),
            scorecard.lonlat_to_xyz(lon - half_lon, lat + half_lat),
        ]
    )

    actual = scorecard._run_polypix(
        scorecard.Workload(
            name="cds_control_point_regression",
            kind="footprints",
            resolution=5,
            item_count=1,
            footprints=footprint,
        ),
        threads=1,
    )
    expected = scorecard._brute_force_membership(footprint, resolution=5)

    assert expected.size > 0
    np.testing.assert_array_equal(np.sort(actual.cells), np.sort(expected))


def test_strip_workload_matches_explicit_footprints() -> None:
    strip = next(
        workload
        for workload in scorecard.build_workloads("smoke")
        if workload.kind == "strip"
    )
    assert strip.left_edge is not None and strip.right_edge is not None
    expanded = scorecard.Workload(
        name="expanded",
        kind="footprints",
        resolution=strip.resolution,
        item_count=strip.item_count,
        footprints=scorecard.strip_footprints(strip.left_edge, strip.right_edge),
    )

    strip_result = scorecard._run_polypix(strip, None)
    expanded_result = scorecard._run_polypix(expanded, None)

    assert scorecard._canonical_digest(strip_result) == scorecard._canonical_digest(
        expanded_result
    )


def test_sparse_candidate_workload_matches_explicit_post_filter() -> None:
    workload = next(
        workload
        for workload in scorecard.build_workloads("smoke")
        if workload.kind == "candidates"
    )
    assert workload.candidate_cells is not None
    assert workload.footprints is not None
    direct = scorecard._run_polypix(workload, None)
    candidate_centers = scorecard._polypix_centers(
        workload.candidate_cells, workload.resolution
    )
    expected = scorecard._candidate_coverage(
        scorecard._segments(workload.footprints),
        workload.candidate_cells,
        candidate_centers,
    )

    assert scorecard._canonical_digest(direct) == scorecard._canonical_digest(expected)


def test_smoke_report_is_machine_readable_and_optional_backends_skip() -> None:
    report = scorecard.build_report(
        profile="smoke",
        backend_names=("polypix", "healpy", "cdshealpix"),
        warmup=0,
        repeats=1,
        thread_modes=(None,),
    )
    round_tripped = json.loads(json.dumps(report))

    assert round_tripped["schema_version"] == 1
    assert round_tripped["timing"]["scope"].startswith("complete public call")
    assert all(
        record["status"] in {"ok", "unsupported", "unavailable"}
        for record in round_tripped["results"]
    )
    assert all(check["status"] == "pass" for check in round_tripped["correctness"])
    assert any(
        record["status"] == "ok" and record["backend"] == "polypix"
        for record in round_tripped["results"]
    )
