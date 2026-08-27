from __future__ import annotations

import pickle

import numpy as np
import pytest

import polypix as px


class GeoObject:
    def __init__(self, mapping: object) -> None:
        self.__geo_interface__ = mapping


def vectors(points: list[tuple[float, float]]) -> np.ndarray:
    lon_lat = np.radians(points)
    longitude = lon_lat[:, 0]
    latitude = lon_lat[:, 1]
    radial = np.cos(latitude)
    return np.column_stack(
        (radial * np.cos(longitude), radial * np.sin(longitude), np.sin(latitude))
    )


def cap_ring(angle: float, radius: float, vertex_count: int = 8) -> np.ndarray:
    angle_rad = np.radians(angle)
    radius_rad = np.radians(radius)
    axis = np.asarray([np.cos(angle_rad), np.sin(angle_rad), 0.0])
    horizontal = np.asarray([-np.sin(angle_rad), np.cos(angle_rad), 0.0])
    phase = np.linspace(0.0, 2.0 * np.pi, vertex_count, endpoint=False)
    return np.cos(radius_rad) * axis + np.sin(radius_rad) * (
        np.cos(phase)[:, None] * horizontal
        + np.sin(phase)[:, None] * np.asarray([0.0, 0.0, 1.0])
    )


def gnomonic_ring(points: list[tuple[float, float]], scale: float = 1.0) -> np.ndarray:
    ring = np.asarray([(1.0, scale * x, scale * y) for x, y in points])
    return ring / np.linalg.norm(ring, axis=1, keepdims=True)


def uneven_cap_ring(angle: float, radii: list[float]) -> np.ndarray:
    angle_rad = np.radians(angle)
    radius_rad = np.radians(radii)
    axis = np.asarray([np.cos(angle_rad), np.sin(angle_rad), 0.0])
    horizontal = np.asarray([-np.sin(angle_rad), np.cos(angle_rad), 0.0])
    phase = np.linspace(0.0, 2.0 * np.pi, len(radii), endpoint=False)
    return np.cos(radius_rad)[:, None] * axis + np.sin(radius_rad)[:, None] * (
        np.cos(phase)[:, None] * horizontal
        + np.sin(phase)[:, None] * np.asarray([0.0, 0.0, 1.0])
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


def all_cell_centers(resolution: int) -> tuple[np.ndarray, np.ndarray]:
    cells = np.arange(px.cell_count(resolution), dtype=np.int64)
    return cells, px.cell_centers(cells, resolution)


def test_concave_polygon_matches_an_independent_center_check() -> None:
    boundary = vectors([(0, 0), (10, 0), (10, 10), (5, 5), (0, 10)])
    cells, centers = all_cell_centers(5)
    visible = centers[:, 0] > 0.0
    inside = np.zeros(len(cells), dtype=np.bool_)
    inside[visible] = planar_contains(
        boundary[:, 1:] / boundary[:, :1],
        centers[visible, 1:] / centers[visible, :1],
    )

    expected = cells[inside]
    actual = px.cover_polygon(boundary, 5).cells

    np.testing.assert_array_equal(actual, expected)


def test_detailed_concave_polygon_matches_an_independent_center_check() -> None:
    vertex_count = 64
    angles = np.arange(vertex_count) * (2.0 * np.pi / vertex_count)
    radii = np.where(np.arange(vertex_count) % 2 == 0, 0.30, 0.24)
    projected = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    boundary = np.column_stack((np.ones(vertex_count), projected))
    boundary /= np.linalg.norm(boundary, axis=1, keepdims=True)

    resolution = 5
    cells = np.arange(px.cell_count(resolution), dtype=np.int64)
    centers = px.cell_centers(cells, resolution)
    visible = centers[:, 0] > 0.0
    inside = np.zeros(len(cells), dtype=np.bool_)
    inside[visible] = planar_contains(
        projected,
        np.column_stack(
            (
                centers[visible, 1] / centers[visible, 0],
                centers[visible, 2] / centers[visible, 0],
            )
        ),
    )
    expected = cells[inside]

    np.testing.assert_array_equal(
        px.cover_polygon(boundary, resolution).cells, expected
    )


def test_raw_batch_can_mix_convex_and_concave_polygons() -> None:
    convex = vectors(
        [
            (-10, -5),
            (-5, -10),
            (5, -10),
            (10, -5),
            (10, 5),
            (5, 10),
            (-5, 10),
            (-10, 5),
        ]
    )
    concave = vectors(
        [(-10, -10), (10, -10), (10, 10), (2, 10), (2, 0), (-2, 0), (-2, 10), (-10, 10)]
    )

    batch = px.cover_polygon(np.stack((convex, concave)), 6, threads=1)

    np.testing.assert_array_equal(
        batch[0], px.cover_polygon(convex, 6, threads=1).cells
    )
    np.testing.assert_array_equal(
        batch[1], px.cover_polygon(concave, 6, threads=1).cells
    )


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


def test_geo_interface_preserves_holes_multipart_union_and_altitude() -> None:
    outer = [(-10, -10), (10, -10), (10, 10), (-10, 10), (-10, -10)]
    hole = [(-3, -3), (3, -3), (3, 3), (-3, 3), (-3, -3)]
    overlap = [(5, -5), (15, -5), (15, 5), (5, 5), (5, -5)]
    mapping = {
        "type": "MultiPolygon",
        "coordinates": [
            [
                [(*position, 100.0) for position in outer],
                [(*position, 0.0) for position in hole],
            ],
            [[(*position, -50.0) for position in overlap]],
        ],
    }
    expected = px.MultiPolygon(
        px.Polygon(vectors(outer), vectors(hole)),
        px.Polygon(vectors(overlap)),
    )

    coverage = px.cover_polygon(GeoObject(mapping), 6)

    np.testing.assert_array_equal(coverage.cells, px.cover_polygon(expected, 6).cells)
    np.testing.assert_array_equal(
        px.cover_polygon(geometry=mapping, resolution=6).cells,
        coverage.cells,
    )


def test_geo_interface_feature_and_batch_keep_one_segment_per_input() -> None:
    first = {
        "type": "Polygon",
        "coordinates": [[(-5, -5), (5, -5), (5, 5), (-5, 5), (-5, -5)]],
    }
    second_geometry = {
        "type": "Polygon",
        "coordinates": [[(20, 0), (25, 0), (25, 5), (20, 5), (20, 0)]],
    }
    second = {
        "type": "Feature",
        "id": "ignored",
        "properties": {"also": "ignored"},
        "geometry": second_geometry,
    }

    coverage = px.cover_polygon([GeoObject(first), second], 5)

    assert len(coverage) == 2
    np.testing.assert_array_equal(
        coverage[0],
        px.cover_polygon(px.Polygon(vectors(first["coordinates"][0])), 5).cells,
    )
    np.testing.assert_array_equal(
        coverage[1],
        px.cover_polygon(
            px.Polygon(vectors(second_geometry["coordinates"][0])), 5
        ).cells,
    )


@pytest.mark.parametrize(
    "mapping",
    [
        {"type": "Polygon", "coordinates": []},
        {"type": "MultiPolygon", "coordinates": []},
        {"type": "Feature", "properties": {}, "geometry": None},
    ],
)
def test_empty_geo_interface_is_one_empty_region(mapping: dict[str, object]) -> None:
    coverage = px.cover_polygon(mapping, 3)
    assert len(coverage) == 1
    assert coverage.cells.size == 0


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ({"type": "Point", "coordinates": (0, 0)}, "unsupported geometry type"),
        ({"type": "FeatureCollection", "features": []}, "sequence of geometries"),
        ({"type": "GeometryCollection", "geometries": []}, "unsupported geometry type"),
        (
            {
                "type": "Polygon",
                "coordinates": [[(181, 0), (1, 0), (0, 1), (181, 0)]],
            },
            "longitude",
        ),
        (
            {
                "type": "Polygon",
                "coordinates": [[(0, 0), (1, 91), (1, 0), (0, 0)]],
            },
            "latitude",
        ),
        (
            {
                "type": "Polygon",
                "coordinates": [[(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0)]],
            },
            "shape",
        ),
        (
            {
                "type": "Polygon",
                "coordinates": [[(0, 0, 1), (1, 0), (1, 1), (0, 0, 1)]],
            },
            "shape",
        ),
        (
            {
                "type": "Polygon",
                "coordinates": [[(0, 0), (1, 0), (0, np.inf), (0, 0)]],
            },
            "finite coordinates",
        ),
    ],
)
def test_invalid_geo_interface_has_a_clear_error(
    mapping: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        px.cover_polygon(mapping, 3)


def test_geo_interface_property_must_be_a_mapping() -> None:
    with pytest.raises(TypeError, match="__geo_interface__ must be a mapping"):
        px.cover_polygon(GeoObject(None), 3)


def test_geo_interface_batch_names_native_validation_failure() -> None:
    good = {
        "type": "Polygon",
        "coordinates": [[(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]],
    }
    crossing = {
        "type": "Polygon",
        "coordinates": [[(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)]],
    }

    with pytest.raises(
        ValueError, match=r"geometry\[1\]\['coordinates'\]: Ring must not cross"
    ):
        px.cover_polygon([good, crossing, good], 3)


def test_distant_holes_are_compared_in_the_outer_ring_hemisphere() -> None:
    polygon = px.Polygon(
        cap_ring(0.0, 89.0, 48),
        cap_ring(-50.0, 4.0),
        cap_ring(40.0, 4.0),
    )

    assert len(polygon.holes) == 2


def test_hemisphere_validation_does_not_depend_on_vertex_density() -> None:
    boundary = [(-10.0, -1.0), (10.0, -1.0)]
    boundary.extend((10.0, float(y)) for y in np.linspace(-0.8, 1.0, 20))
    boundary.extend([(-10.0, 1.0), (-5.0, 0.0)])

    polygon = px.Polygon(gnomonic_ring(boundary))
    simplified = px.Polygon(
        gnomonic_ring(
            [(-10.0, -1.0), (10.0, -1.0), (10.0, 1.0), (-10.0, 1.0), (-5.0, 0.0)]
        )
    )

    assert len(polygon.outer) == len(boundary)
    np.testing.assert_array_equal(
        px.cover_polygon(polygon, 5).cells,
        px.cover_polygon(simplified, 5).cells,
    )


def test_hole_must_stay_inside_the_outer_ring_hemisphere() -> None:
    hole = uneven_cap_ring(85.0, [1.0, 29.0, 3.0, 9.0, 22.0])

    with pytest.raises(ValueError, match="strictly inside"):
        px.Polygon(cap_ring(0.0, 89.0, 48), hole)


def test_small_simple_ring_is_not_mistaken_for_a_crossing_ring() -> None:
    boundary = [
        (0.3154391160499824, 0.1903668753093077),
        (0.2601874926834980, 0.24717897398330543),
        (-0.22379507620652334, 0.11621164013174248),
        (-0.49887531843793076, 0.12282523093622569),
        (0.8138773258877517, -0.24293848886683733),
    ]

    px.Polygon(gnomonic_ring(boundary, 3.0e-7))
    with pytest.raises(ValueError, match="cross or touch itself"):
        px.Polygon(
            gnomonic_ring(
                [(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)],
                3.0e-7,
            )
        )


def test_raw_crossing_ring_reports_the_general_validation_error() -> None:
    crossing = gnomonic_ring([(0.0, 0.0), (2.0, 2.0), (2.0, 0.0), (0.0, 2.0)])

    with pytest.raises(ValueError, match="cross or touch itself"):
        px.cover_polygon(crossing, 6)


@pytest.mark.parametrize("structured_first", [False, True])
def test_mixed_polygon_representations_have_one_clear_error(
    structured_first: bool,
) -> None:
    array = gnomonic_ring([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    polygon = px.Polygon(array)
    batch = [polygon, array] if structured_first else [array, polygon]

    with pytest.raises(
        TypeError, match="cannot be used in a structured geometry batch"
    ):
        px.cover_polygon(batch, 3)


def test_geo_interface_cannot_mix_with_other_representations() -> None:
    array = gnomonic_ring([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    mapping = {
        "type": "Polygon",
        "coordinates": [[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]],
    }

    with pytest.raises(TypeError, match="geo-interface objects with Polygon objects"):
        px.cover_polygon([GeoObject(mapping), px.Polygon(array)], 3)
    with pytest.raises(
        TypeError, match="cannot be used in a structured geometry batch"
    ):
        px.cover_polygon([mapping, array], 3)
    with pytest.raises(TypeError, match=r"geometry\[1\] cannot be used"):
        px.cover_polygon([mapping, None], 3)


def test_object_array_points_to_the_documented_batch_form() -> None:
    mapping = {
        "type": "Polygon",
        "coordinates": [[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]],
    }
    geometries = np.asarray([GeoObject(mapping), GeoObject(mapping)], dtype=object)

    with pytest.raises(TypeError, match=r"object-dtype arrays.*list\(geometry\)"):
        px.cover_polygon(geometries, 3)


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


def test_polygon_coverage_uses_the_geometry_validated_at_construction() -> None:
    polygon = px.Polygon(vectors([(-10, -10), (10, -10), (10, 10), (-10, 10)]))
    expected = px.cover_polygon(polygon, 5).cells

    polygon.outer.setflags(write=True)
    polygon.outer[:] = vectors([(80, -10), (100, -10), (100, 10), (80, 10)])
    polygon.outer.setflags(write=False)

    np.testing.assert_array_equal(px.cover_polygon(polygon, 5).cells, expected)


def test_polygon_pickle_rebuilds_its_prepared_geometry() -> None:
    polygon = px.Polygon(
        vectors([(-10, -10), (10, -10), (10, 10), (-10, 10)]),
        vectors([(-3, -3), (3, -3), (3, 3), (-3, 3)]),
    )

    restored = pickle.loads(pickle.dumps(polygon))

    np.testing.assert_array_equal(restored.outer, polygon.outer)
    np.testing.assert_array_equal(restored.holes[0], polygon.holes[0])
    np.testing.assert_array_equal(
        px.cover_polygon(restored, 5).cells,
        px.cover_polygon(polygon, 5).cells,
    )


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
