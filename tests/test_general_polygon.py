from __future__ import annotations

import numpy as np
import pytest

import polypix as px


def vectors(points: list[tuple[float, float]]) -> np.ndarray:
    lon_lat = np.radians(points)
    longitude = lon_lat[:, 0]
    latitude = lon_lat[:, 1]
    radial = np.cos(latitude)
    return np.column_stack(
        (radial * np.cos(longitude), radial * np.sin(longitude), np.sin(latitude))
    )


def planar_contains(points: np.ndarray, queries: np.ndarray) -> np.ndarray:
    inside = np.zeros(len(queries), dtype=np.bool_)
    x = queries[:, 0]
    y = queries[:, 1]
    for (x1, y1), (x2, y2) in zip(points, np.roll(points, -1, axis=0), strict=True):
        if y1 == y2:
            continue
        crosses = (y1 > y) != (y2 > y)
        intersection = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        inside ^= crosses & (x < intersection)
    return inside


def all_center_longitudes_latitudes(resolution: int) -> tuple[np.ndarray, np.ndarray]:
    cells = np.arange(px.cell_count(resolution), dtype=np.int64)
    centers = px.cell_centers(cells, resolution)
    coordinates = np.column_stack(
        (
            np.degrees(np.arctan2(centers[:, 1], centers[:, 0])),
            np.degrees(np.arcsin(centers[:, 2])),
        )
    )
    return cells, coordinates


def test_concave_polygon_matches_an_independent_center_check() -> None:
    boundary = np.asarray([(0, 0), (10, 0), (10, 10), (5, 5), (0, 10)], float)
    cells, coordinates = all_center_longitudes_latitudes(5)

    expected = cells[planar_contains(boundary, coordinates)]
    actual = px.cover_polygon(vectors(boundary.tolist()), 5).cells

    np.testing.assert_array_equal(actual, expected)


def test_polygon_holes_and_multipolygon_form_one_deduplicated_region() -> None:
    outer = vectors([(-10, -10), (10, -10), (10, 10), (-10, 10)])
    hole = vectors([(-3, -3), (3, -3), (3, 3), (-3, 3)])
    overlap = vectors([(5, -5), (15, -5), (15, 5), (5, 5)])
    polygon = px.Polygon(outer, hole)
    region = px.MultiPolygon(polygon, px.Polygon(overlap))

    coverage = px.cover_polygon(region, 6)
    without_hole = np.setdiff1d(
        px.cover_polygon(outer, 6).cells,
        px.cover_polygon(hole, 6).cells,
    )
    expected = np.union1d(without_hole, px.cover_polygon(overlap, 6).cells)

    assert len(coverage) == 1
    np.testing.assert_array_equal(coverage.cells, expected)
    assert len(coverage.cells) == len(np.unique(coverage.cells))
    selected = coverage.cells[::3]
    np.testing.assert_array_equal(
        px.cover_polygon(region, 6, candidate_cells=selected).cells,
        selected,
    )


def test_region_batch_reducers_count_each_region_once() -> None:
    first = px.MultiPolygon(
        px.Polygon(vectors([(-5, -5), (5, -5), (5, 5), (-5, 5)])),
        px.Polygon(vectors([(0, -5), (10, -5), (10, 5), (0, 5)])),
    )
    second = px.Polygon(vectors([(20, 0), (25, 0), (25, 5), (20, 5)]))
    coverage = px.cover_polygon([first, second], 5)

    np.testing.assert_array_equal(
        px.cover_polygon([first, second], 5, reduce=px.Count()),
        coverage.reduce(px.Count()),
    )
    np.testing.assert_array_equal(
        px.cover_polygon([first, second], 5, reduce=px.Sum([2.0, 3.0])),
        coverage.reduce(px.Sum([2.0, 3.0])),
    )


def test_polygon_containers_validate_and_own_their_coordinates() -> None:
    outer = vectors([(-10, -10), (10, -10), (10, 10), (-10, 10)])
    polygon = px.Polygon(outer)
    outer[:] = 0.0

    assert not polygon.outer.flags.writeable
    assert np.all(np.linalg.norm(polygon.outer, axis=1) > 0.0)
    with pytest.raises(ValueError, match="cross or touch itself"):
        px.Polygon(vectors([(-10, -10), (10, 10), (10, -10), (-10, 10)]))
    with pytest.raises(ValueError, match="strictly inside"):
        px.Polygon(
            polygon.outer,
            vectors([(20, 20), (21, 20), (21, 21), (20, 21)]),
        )
    with pytest.raises(TypeError, match="Polygon objects"):
        px.MultiPolygon(polygon, polygon.outer)  # type: ignore[arg-type]


def test_empty_multipolygon_is_one_empty_region() -> None:
    coverage = px.cover_polygon(px.MultiPolygon(), 3)
    assert len(coverage) == 1
    assert coverage.cells.size == 0


@pytest.mark.parametrize(
    "polygon",
    [
        px.Polygon(
            vectors(
                [(170, -10), (-170, -10), (-175, 0), (-170, 10), (170, 10), (175, 0)]
            )
        ),
        px.Polygon(
            vectors([(0, 60), (90, 60), (180, 60), (-90, 60)]),
            vectors([(0, 80), (-90, 80), (180, 80), (90, 80)]),
        ),
    ],
)
def test_general_scan_bounds_match_testing_every_cell(polygon: px.Polygon) -> None:
    resolution = 5
    every_cell = np.arange(px.cell_count(resolution), dtype=np.int64)
    scanned = px.cover_polygon(polygon, resolution)
    tested = px.cover_polygon(polygon, resolution, candidate_cells=every_cell)
    np.testing.assert_array_equal(scanned.cells, tested.cells)
