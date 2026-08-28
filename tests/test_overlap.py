"""Whole-cell intersection coverage."""

from __future__ import annotations

import numpy as np
import pytest

import polypix as px


def xyz(longitude_deg: object, latitude_deg: object) -> np.ndarray:
    longitude = np.radians(longitude_deg)
    latitude = np.radians(latitude_deg)
    return np.stack(
        (
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ),
        axis=-1,
    )


# Generated independently by densely sampling each region boundary through
# cell_at(), then unioning those cells with center coverage. The fixtures stay
# checked in; no HEALPix package is a test or runtime dependency.
POLYGON_OVERLAP_RESOLUTION_3 = np.asarray(
    [
        176,
        205,
        206,
        207,
        208,
        209,
        238,
        239,
        240,
        241,
        269,
        270,
        271,
        272,
        273,
        301,
        302,
        303,
        304,
        305,
        333,
        334,
        335,
        336,
        337,
        338,
        365,
        366,
        367,
        368,
        369,
        397,
        398,
        399,
        400,
        401,
        402,
        430,
        431,
        432,
        433,
        462,
        463,
        464,
        465,
        495,
        496,
        527,
    ],
    dtype=np.int64,
)
CAP_OVERLAP_RESOLUTION_3 = np.asarray(
    [
        112,
        113,
        114,
        144,
        145,
        146,
        147,
        176,
        177,
        178,
        179,
        207,
        208,
        209,
        210,
        211,
        240,
        241,
        242,
        243,
        271,
        272,
        273,
        274,
        275,
        304,
        305,
        306,
        337,
        338,
        369,
    ],
    dtype=np.int64,
)


def test_overlap_matches_independent_boundary_fixtures() -> None:
    polygon = xyz(
        [-31.0, -5.0, 22.0, 13.0, -27.0],
        [7.0, -18.0, -9.0, 24.0, 29.0],
    )
    axis = xyz(17.0, 23.0)

    np.testing.assert_array_equal(
        px.cover_polygon(polygon, 3, mode="overlap").cells,
        POLYGON_OVERLAP_RESOLUTION_3,
    )
    np.testing.assert_array_equal(
        px.cover_cap(axis, np.radians(19.0), 3, mode="overlap").cells,
        CAP_OVERLAP_RESOLUTION_3,
    )


def test_subcell_regions_and_point_caps_do_not_disappear() -> None:
    polygon = xyz(
        [10.0, 10.001, 10.001, 10.0],
        [10.0, 10.0, 10.001, 10.001],
    )
    expected = px.cell_at(polygon[0], 8)

    assert px.cover_polygon(polygon, 8).cells.size == 0
    np.testing.assert_array_equal(
        px.cover_polygon(polygon, 8, mode="overlap").cells, expected
    )
    assert px.cover_cap(polygon[0], 0.0, 8).cells.size == 0
    np.testing.assert_array_equal(
        px.cover_cap(polygon[0], 0.0, 8, mode="overlap").cells, expected
    )


def test_emitted_transition_corner_does_not_abort_overlap() -> None:
    corner = np.asarray([-0.4140976024357428, 0.6197408581112958, 0.6666666666666667])
    np.testing.assert_array_equal(
        px.cover_cap(corner, 0.0, 3, mode="overlap").cells,
        [93, 122, 123, 155],
    )


def test_overlap_respects_holes_candidates_and_binary_reducers() -> None:
    outer = xyz([-30.0, 30.0, 30.0, -30.0], [-30.0, -30.0, 30.0, 30.0])
    hole = xyz([-10.0, 10.0, 10.0, -10.0], [-10.0, -10.0, 10.0, 10.0])
    polygon = px.Polygon(outer, hole)
    resolution = 5
    full = px.cover_polygon(polygon, resolution, mode="overlap").cells
    candidates = np.arange(0, px.cell_count(resolution), 11)
    expected = np.intersect1d(full, candidates)

    restricted = px.cover_polygon(
        polygon,
        resolution,
        mode="overlap",
        candidate_cells=candidates,
    )
    np.testing.assert_array_equal(restricted.cells, expected)
    np.testing.assert_array_equal(
        px.cover_polygon(
            polygon,
            resolution,
            mode="overlap",
            candidate_cells=candidates,
            reduce=px.Count(),
        ),
        np.isin(candidates, full).astype(np.int64),
    )


def test_overlap_sweep_segments_match_their_polygons() -> None:
    left = xyz([-10.0, 0.0, 10.0], [4.0, 5.0, 4.0])
    right = xyz([-10.0, 0.0, 10.0], [-4.0, -5.0, -4.0])
    actual = px.cover_sweep(left, right, 5, mode="overlap")

    for index in range(2):
        polygon = np.asarray(
            [left[index], right[index], right[index + 1], left[index + 1]]
        )
        np.testing.assert_array_equal(
            actual[index], px.cover_polygon(polygon, 5, mode="overlap").cells
        )


def test_center_coverage_is_a_subset_of_overlap() -> None:
    polygon = xyz(
        [-31.0, -5.0, 22.0, 13.0, -27.0],
        [7.0, -18.0, -9.0, 24.0, 29.0],
    )
    axis = xyz(17.0, 23.0)
    left = xyz([-10.0, 0.0, 10.0], [4.0, 5.0, 4.0])
    right = xyz([-10.0, 0.0, 10.0], [-4.0, -5.0, -4.0])

    polygon_center = px.cover_polygon(polygon, 5)
    polygon_overlap = px.cover_polygon(polygon, 5, mode="overlap")
    assert np.setdiff1d(polygon_center.cells, polygon_overlap.cells).size == 0

    cap_center = px.cover_cap(axis, np.radians(19.0), 5)
    cap_overlap = px.cover_cap(axis, np.radians(19.0), 5, mode="overlap")
    assert np.setdiff1d(cap_center.cells, cap_overlap.cells).size == 0

    sweep_center = px.cover_sweep(left, right, 5)
    sweep_overlap = px.cover_sweep(left, right, 5, mode="overlap")
    for index in range(len(sweep_center)):
        assert np.setdiff1d(sweep_center[index], sweep_overlap[index]).size == 0


def test_candidate_cap_overlap_is_thread_invariant() -> None:
    resolution = 8
    centers = np.repeat(xyz(10.0, 10.0)[None, :], 2400, axis=0)
    radii = np.zeros(len(centers))
    candidates = np.arange(0, px.cell_count(resolution), 64)

    sequential = px.cover_cap(
        centers,
        radii,
        resolution,
        candidate_cells=candidates,
        mode="overlap",
        threads=1,
    )
    parallel = px.cover_cap(
        centers,
        radii,
        resolution,
        candidate_cells=candidates,
        mode="overlap",
        threads=8,
    )
    np.testing.assert_array_equal(parallel.cells, sequential.cells)
    np.testing.assert_array_equal(parallel.offsets, sequential.offsets)


def test_overlap_handles_poles_full_sphere_and_sparse_resolution_29() -> None:
    np.testing.assert_array_equal(
        px.cover_cap([0.0, 0.0, 1.0], 0.0, 1, mode="overlap").cells,
        [0, 1, 2, 3],
    )
    np.testing.assert_array_equal(
        px.cover_cap([1.0, 0.0, 0.0], np.pi, 1, mode="overlap").cells,
        np.arange(px.cell_count(1)),
    )
    cells = np.asarray([0, 1, px.cell_count(29) - 1], dtype=np.int64)
    counts = px.cover_cap(
        px.cell_centers(cells[:1], 29),
        0.0,
        29,
        mode="overlap",
        candidate_cells=cells,
        reduce=px.Count(),
    )
    np.testing.assert_array_equal(counts, [1, 0, 0])


def test_mode_validation_and_center_default() -> None:
    polygon = xyz([-5.0, 5.0, 5.0, -5.0], [-5.0, -5.0, 5.0, 5.0])
    np.testing.assert_array_equal(
        px.cover_polygon(polygon, 4).cells,
        px.cover_polygon(polygon, 4, mode="center").cells,
    )
    with pytest.raises(ValueError, match="mode must be 'center' or 'overlap'"):
        px.cover_polygon(polygon, 4, mode="full")  # type: ignore[arg-type]
