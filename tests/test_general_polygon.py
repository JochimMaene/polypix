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
    return cells, px.cell_centers(cells, resolution=resolution)


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
    actual = px.cover_polygon(boundary, resolution=5).cells

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
    centers = px.cell_centers(cells, resolution=resolution)
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
        px.cover_polygon(boundary, resolution=resolution).cells, expected
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

    batch = px.cover_polygon(np.stack((convex, concave)), resolution=6, threads=1)

    np.testing.assert_array_equal(
        batch[0], px.cover_polygon(convex, resolution=6, threads=1).cells
    )
    np.testing.assert_array_equal(
        batch[1], px.cover_polygon(concave, resolution=6, threads=1).cells
    )


def test_polygon_holes_and_multipolygon_form_one_deduplicated_region() -> None:
    outer = vectors([(-10, -10), (10, -10), (10, 10), (-10, 10)])
    hole = vectors([(-3, -3), (3, -3), (3, 3), (-3, 3)])
    overlap = vectors([(5, -5), (15, -5), (15, 5), (5, 5)])
    polygon = px.Polygon(outer, hole)
    region = px.MultiPolygon(polygon, px.Polygon(overlap))

    coverage = px.cover_polygon(region, resolution=6)
    without_hole = np.setdiff1d(
        px.cover_polygon(outer, resolution=6).cells,
        px.cover_polygon(hole, resolution=6).cells,
    )
    expected = np.union1d(without_hole, px.cover_polygon(overlap, resolution=6).cells)

    assert len(coverage) == 1
    np.testing.assert_array_equal(coverage.cells, expected)
    assert len(coverage.cells) == len(np.unique(coverage.cells))
    selected = coverage.cells[::3]
    np.testing.assert_array_equal(
        px.cover_polygon(region, resolution=6, candidate_cells=selected).cells,
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

    coverage = px.cover_polygon(GeoObject(mapping), resolution=6)

    np.testing.assert_array_equal(
        coverage.cells, px.cover_polygon(expected, resolution=6).cells
    )
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

    coverage = px.cover_polygon([GeoObject(first), second], resolution=5)

    assert len(coverage) == 2
    np.testing.assert_array_equal(
        coverage[0],
        px.cover_polygon(
            px.Polygon(vectors(first["coordinates"][0])), resolution=5
        ).cells,
    )
    np.testing.assert_array_equal(
        coverage[1],
        px.cover_polygon(
            px.Polygon(vectors(second_geometry["coordinates"][0])), resolution=5
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
    coverage = px.cover_polygon(mapping, resolution=3)
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
        px.cover_polygon(mapping, resolution=3)


def test_geo_interface_property_must_be_a_mapping() -> None:
    with pytest.raises(TypeError, match="__geo_interface__ must be a mapping"):
        px.cover_polygon(GeoObject(None), resolution=3)


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
        px.cover_polygon([good, crossing, good], resolution=3)


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
        px.cover_polygon(polygon, resolution=5).cells,
        px.cover_polygon(simplified, resolution=5).cells,
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
        px.cover_polygon(crossing, resolution=6)


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
        px.cover_polygon(batch, resolution=3)


def test_geo_interface_cannot_mix_with_other_representations() -> None:
    array = gnomonic_ring([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    mapping = {
        "type": "Polygon",
        "coordinates": [[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]],
    }

    with pytest.raises(TypeError, match="geo-interface objects with Polygon objects"):
        px.cover_polygon([GeoObject(mapping), px.Polygon(array)], resolution=3)
    with pytest.raises(
        TypeError, match="cannot be used in a structured geometry batch"
    ):
        px.cover_polygon([mapping, array], resolution=3)
    with pytest.raises(TypeError, match=r"geometry\[1\] cannot be used"):
        px.cover_polygon([mapping, None], resolution=3)


def test_object_array_points_to_the_documented_batch_form() -> None:
    mapping = {
        "type": "Polygon",
        "coordinates": [[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]],
    }
    geometries = np.asarray([GeoObject(mapping), GeoObject(mapping)], dtype=object)

    with pytest.raises(TypeError, match=r"object-dtype arrays.*list\(geometry\)"):
        px.cover_polygon(geometries, resolution=3)


def test_empty_object_array_remains_an_empty_batch() -> None:
    coverage = px.cover_polygon(np.empty(0, dtype=object), resolution=3)

    assert len(coverage) == 0
    np.testing.assert_array_equal(coverage.offsets, [0])


@pytest.mark.parametrize("geometry", [None, object()])
def test_non_array_objects_keep_the_numeric_input_error(geometry: object) -> None:
    with pytest.raises(TypeError, match="geometry must contain real numbers"):
        px.cover_polygon(geometry, resolution=3)


def test_non_sequence_iterable_points_to_the_sequence_batch_form() -> None:
    with pytest.raises(TypeError, match=r"non-sequence iterables.*list\(geometry\)"):
        px.cover_polygon(iter(()), resolution=3)


def test_sequence_subclass_can_expose_geo_interface() -> None:
    mapping = {
        "type": "Polygon",
        "coordinates": [[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]],
    }

    class TupleGeo(tuple[()]):
        @property
        def __geo_interface__(self) -> dict[str, object]:
            return mapping

    coverage = px.cover_polygon([TupleGeo(), TupleGeo()], resolution=3)

    assert len(coverage) == 2


def test_geo_interface_takes_precedence_over_mapping_subclass() -> None:
    mapping = {
        "type": "Polygon",
        "coordinates": [[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]],
    }

    class MappingGeo(dict[str, object]):
        @property
        def __geo_interface__(self) -> dict[str, object]:
            return mapping

    coverage = px.cover_polygon(MappingGeo(not_geojson=True), resolution=3)

    assert len(coverage) == 1


def test_region_batch_reducers_count_each_region_once() -> None:
    first = px.MultiPolygon(
        px.Polygon(vectors([(-5, -5), (5, -5), (5, 5), (-5, 5)])),
        px.Polygon(vectors([(0, -5), (10, -5), (10, 5), (0, 5)])),
    )
    second = px.Polygon(vectors([(20, 0), (25, 0), (25, 5), (20, 5)]))
    coverage = px.cover_polygon([first, second], resolution=5)

    np.testing.assert_array_equal(
        px.cover_polygon([first, second], resolution=5, reduce=px.Count()),
        coverage.reduce(px.Count()),
    )
    np.testing.assert_array_equal(
        px.cover_polygon([first, second], resolution=5, reduce=px.Sum([2.0, 3.0])),
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
    expected = px.cover_polygon(polygon, resolution=5).cells

    polygon.outer.setflags(write=True)
    polygon.outer[:] = vectors([(80, -10), (100, -10), (100, 10), (80, 10)])
    polygon.outer.setflags(write=False)

    np.testing.assert_array_equal(
        px.cover_polygon(polygon, resolution=5).cells, expected
    )


def test_polygon_pickle_rebuilds_its_prepared_geometry() -> None:
    polygon = px.Polygon(
        vectors([(-10, -10), (10, -10), (10, 10), (-10, 10)]),
        vectors([(-3, -3), (3, -3), (3, 3), (-3, 3)]),
    )

    restored = pickle.loads(pickle.dumps(polygon))

    np.testing.assert_array_equal(restored.outer, polygon.outer)
    np.testing.assert_array_equal(restored.holes[0], polygon.holes[0])
    np.testing.assert_array_equal(
        px.cover_polygon(restored, resolution=5).cells,
        px.cover_polygon(polygon, resolution=5).cells,
    )


def test_empty_multipolygon_is_one_empty_region() -> None:
    coverage = px.cover_polygon(px.MultiPolygon(), resolution=3)
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
    scanned = px.cover_polygon(polygon, resolution=resolution)
    tested = px.cover_polygon(
        polygon, resolution=resolution, candidate_cells=every_cell
    )
    np.testing.assert_array_equal(scanned.cells, tested.cells)


def _equatorial_spike(
    resolution: int, gap_cells: int, edge_length: float
) -> tuple[np.ndarray, int, np.ndarray]:
    """A concave ring whose easternmost feature is one short equatorial edge.

    Cell centres on the ``z == 0`` ring share the edge's plane exactly, so
    they reach the boundary branch of the containment test with nothing left
    to break the tie but the endpoint check. The returned cell is the first
    centre east of the tip: outside the ring by ``gap_cells`` cell widths,
    and outside the ring's longitude bounds by the same amount.
    """
    nside = 1 << resolution
    outside_cell = int(6 * nside * nside - 2 * nside + 123456)
    center = px.cell_centers([outside_cell], resolution=resolution)[0]
    longitude = float(np.arctan2(center[1], center[0]))
    step = 2.0 * np.pi / (4 * nside)
    gap = gap_cells * step
    block = edge_length

    def on_equator(offset: float) -> list[float]:
        angle = longitude + offset
        return [np.cos(angle), np.sin(angle), 0.0]

    def vertex(offset: float, latitude: float) -> list[float]:
        angle = longitude + offset
        return [
            np.cos(latitude) * np.cos(angle),
            np.cos(latitude) * np.sin(angle),
            np.sin(latitude),
        ]

    # The spike's own corner sits on the equator, so the tip edge stays in
    # that plane exactly, and every other edge spans a whole `block`. No
    # feature here is anywhere near the scale at which a ring's concavity
    # stops being distinguishable from a straight edge.
    ring = np.array(
        [
            vertex(-3.0 * block, -block),
            vertex(-2.0 * block, -block),
            on_equator(-gap - edge_length),
            on_equator(-gap),
            vertex(-2.0 * block, block),
            vertex(-3.0 * block, block),
        ]
    )
    window = np.arange(outside_cell - 40, outside_cell + 40, dtype=np.int64)
    return ring, outside_cell, window


def test_short_edge_boundary_agrees_between_scanning_and_candidates() -> None:
    """Endpoint slack must not depend on how long the edge is.

    Measuring the endpoint check in cosines lets a 1e-6-radian edge claim a
    point 1e-14/sin(length) radians past its end, which at resolution 29 is
    several cells. The ring scan bounds do not stretch that far, so the same
    polygon then answered one way as a scan and another as a candidate test.
    """
    resolution = 29
    ring, outside_cell, window = _equatorial_spike(resolution, 3, 1e-6)

    scanned = px.cover_polygon(ring, resolution=resolution)
    tested = px.cover_polygon(ring, resolution=resolution, candidate_cells=window)

    assert outside_cell not in scanned.cells
    assert outside_cell not in tested.cells
    np.testing.assert_array_equal(
        np.sort(tested.cells), np.sort(np.intersect1d(scanned.cells, window))
    )
    # The ring's cells step uniformly in longitude, so the tip sits exactly
    # on the centre of the cell three west. Everything from there westward is
    # inside, and that centre stays inside: the endpoint check still has to
    # admit a point genuinely touching the edge's end.
    tip_cell = outside_cell - 3
    np.testing.assert_array_equal(np.sort(tested.cells), window[window <= tip_cell])
    assert tip_cell in scanned.cells


def test_short_edge_boundary_agrees_on_both_candidate_planner_choices() -> None:
    """A reduction takes candidates as a hint, so both branches must agree.

    Below the planner's threshold it tests the candidates; above it, it scans
    and restricts afterwards. Those are different predicates on the same
    cell, and a boundary case has to come out the same either way.
    """
    resolution = 29
    ring, outside_cell, window = _equatorial_spike(resolution, 3, 1e-6)
    # The estimate for this footprint puts the crossover near 166K cells.
    scanning = np.arange(outside_cell - 200_000, outside_cell + 1, dtype=np.int64)

    for candidates in (window, scanning):
        counts = px.cover_polygon(
            ring, resolution=resolution, candidate_cells=candidates, reduce=px.Count()
        )
        covered = px.cover_polygon(
            ring, resolution=resolution, candidate_cells=candidates
        )
        assert int(np.asarray(counts)[candidates == outside_cell][0]) == 0
        np.testing.assert_array_equal(
            np.sort(candidates[np.asarray(counts) > 0]), np.sort(covered.cells)
        )


def _tangent_plane(center: list[float]):
    def unit(vector: np.ndarray) -> np.ndarray:
        return vector / np.linalg.norm(vector)

    axis = unit(np.asarray(center, dtype=float))
    first = unit(np.cross([0.0, 0.0, 1.0], axis))
    second = np.cross(axis, first)
    return lambda x, y: unit(axis + x * first + y * second)


def test_barely_concave_polygon_keeps_the_cells_in_its_dent() -> None:
    # The reflex vertex sits 1e-11 rad from its neighbour, as near-duplicate
    # GIS vertices do, so the convex validator alone would accept the ring.
    point = _tangent_plane([0.6, 0.3, 0.5])
    quad = np.array(
        [point(0, 0), point(1e-11, 0), point(1e-3, -5e-6), point(5e-4, 1e-3)]
    )
    hole = np.array([point(4e-4, 5e-4), point(6e-4, 5e-4), point(5e-4, 6e-4)])
    plain = px.cover_polygon(quad, resolution=18).cells
    holed = px.cover_polygon(px.Polygon(quad, hole), resolution=18).cells
    assert np.isin(holed, plain).all()


def _cover_as(kind: str, vertices: np.ndarray, **options: object) -> px.Coverage:
    """Cover a triangle or quad as a polygon, or as the one-segment sweep
    whose quad (left[0], right[0], right[1], left[1]) has the same vertices.
    A triangle becomes a sweep pinched at its first vertex."""
    if kind == "polygon":
        return px.cover_polygon(vertices, **options)  # type: ignore[call-overload]
    left = vertices[[0, 3]] if len(vertices) == 4 else vertices[[0, 0]]
    return px.cover_sweep(left, vertices[[1, 2]], **options)  # type: ignore[call-overload]


@pytest.mark.parametrize("kind", ["polygon", "sweep"])
def test_tiny_concave_footprint_keeps_interior_cells(kind: str) -> None:
    vertices = np.array(
        [
            [-0.2528946054863888, -0.2284496612206952, 0.940135666169534],
            [-0.2528945914533086, -0.22844966922082713, 0.9401356680004003],
            [-0.2528945866684893, -0.22844967763114138, 0.94013566724383],
            [-0.2528945795380308, -0.2284496503820889, 0.9401356757833329],
        ]
    )
    inside = 25882078244167306
    for mode in ("center", "overlap"):
        coverage = _cover_as(kind, vertices, resolution=28, mode=mode)
        assert inside in coverage.cells
        candidates = _cover_as(
            kind, vertices, resolution=28, mode=mode, candidate_cells=[inside]
        )
        assert candidates.cells.tolist() == [inside]


@pytest.mark.parametrize("kind", ["polygon", "sweep"])
def test_sharp_vertex_does_not_reach_past_its_tip(kind: str) -> None:
    resolution = 20
    longitude = 1000.8 * 2 * np.pi / (4 << resolution)
    half_angle, length = 1e-9, 0.01
    tip = np.array([np.cos(longitude), np.sin(longitude), 0.0])
    base = np.array(
        [
            np.cos(longitude - length) * np.cos(length * half_angle),
            np.sin(longitude - length) * np.cos(length * half_angle),
            np.sin(length * half_angle),
        ]
    )
    triangle = np.array([tip, base, base * [1, 1, -1]])

    centers = px.cell_centers(
        _cover_as(kind, triangle, resolution=resolution).cells, resolution=resolution
    )
    assert (np.arctan2(centers[:, 1], centers[:, 0]) <= longitude).all()

    near = px.cell_at(tip, resolution=resolution)[0]
    candidates = np.arange(near - 40, near + 40)
    full = _cover_as(kind, triangle, resolution=resolution, mode="overlap").cells
    selected = _cover_as(
        kind,
        triangle,
        resolution=resolution,
        mode="overlap",
        candidate_cells=candidates,
    ).cells
    np.testing.assert_array_equal(np.intersect1d(full, candidates), selected)


def test_polar_boundary_tie_agrees_across_scan_and_candidates() -> None:
    # Cell 19800's center lies 1.5e-15 rad outside an edge, inside the
    # containment tolerance. Next to the pole that tolerance spans more
    # longitude than the scan bounds did, so the full scan skipped the cell
    # while a candidate test accepted it.
    resolution, cell = 29, 19800
    point = px.cell_centers([cell], resolution=resolution)[0]
    longitude = np.arctan2(point[1], point[0])
    angles = longitude + np.array([1e-8, 1.0, 1.0, 1e-8])
    radial = np.array([1e-8, 1e-8, 1e-6, 1e-6])
    quad = np.column_stack(
        (radial * np.cos(angles), radial * np.sin(angles), np.sqrt(1 - radial**2))
    )
    steps = np.linspace(0, 1, 5, endpoint=False)[:, None]
    subdivided = np.concatenate(
        [quad[i] + steps * (quad[(i + 1) % 4] - quad[i]) for i in range(4)]
    )
    for mode in ("center", "overlap"):
        for vertices in (quad, subdivided):
            full = px.cover_polygon(vertices, resolution=resolution, mode=mode)
            selected = px.cover_polygon(
                vertices, resolution=resolution, mode=mode, candidate_cells=[cell]
            )
            assert cell in full.cells, mode
            assert selected.cells.tolist() == [cell], mode


def test_acute_tip_tie_agrees_across_scan_candidates_and_subdivision() -> None:
    # The convex test accepts points within its tolerance of every edge
    # plane, which past an acute tip reaches tolerance / sin(angle / 2). A cell
    # center placed just inside that reach must be treated the same by a full
    # scan, a candidate test, and the same triangle with subdivided edges.
    def unit(vectors: np.ndarray) -> np.ndarray:
        return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)

    resolution, cell, half = 19, 1_234_567_890, 1e-3
    center = px.cell_centers([cell], resolution=resolution)[0]
    east = unit(np.cross([0.0, 0.0, 1.0], center))
    normal = unit(np.cross(center, east))
    tip = unit(center - 0.5e-14 / np.sin(half) * east)
    length = 20 * 2.0**-resolution
    spread = length * np.tan(half) * normal
    triangle = np.array(
        [tip, unit(tip - length * east + spread), unit(tip - length * east - spread)]
    )
    steps = np.linspace(0, 1, 4, endpoint=False)[:, None]
    subdivided = unit(
        np.concatenate(
            [
                triangle[i] + steps * (triangle[(i + 1) % 3] - triangle[i])
                for i in range(3)
            ]
        )
    )
    answers = {
        cell in px.cover_polygon(triangle, resolution=resolution).cells,
        cell in px.cover_polygon(subdivided, resolution=resolution).cells,
        cell
        in px.cover_polygon(
            triangle, resolution=resolution, candidate_cells=[cell]
        ).cells,
    }
    assert len(answers) == 1
