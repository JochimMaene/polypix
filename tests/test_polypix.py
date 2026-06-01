from __future__ import annotations

import math
import os
import unittest

import numpy as np

import polypix as px


def lonlat_to_vec(lon_deg: float, lat_deg: float) -> np.ndarray:
    lon = math.radians(lon_deg)
    lat = math.radians(lat_deg)
    cos_lat = math.cos(lat)
    return np.array(
        [cos_lat * math.cos(lon), cos_lat * math.sin(lon), math.sin(lat)],
        dtype=np.float64,
    )


def cell_id_prefix(resolution: int) -> np.uint64:
    return np.uint64(1 << (4 + 2 * resolution))


def encode_nested(resolution: int, nested_indices: np.ndarray | list[int]) -> np.ndarray:
    return np.asarray(nested_indices, dtype=np.uint64) | cell_id_prefix(resolution)


def orient_convex(vertices_lonlat: list[tuple[float, float]]) -> list[np.ndarray]:
    vectors = [lonlat_to_vec(lon, lat) for lon, lat in vertices_lonlat]
    if np.allclose(vectors[0], vectors[-1]):
        vectors = vectors[:-1]
    interior = np.sum(vectors, axis=0)
    interior /= np.linalg.norm(interior)
    orientation = 0.0
    for index, current in enumerate(vectors):
        nxt = vectors[(index + 1) % len(vectors)]
        orientation += float(np.dot(np.cross(current, nxt), interior))
    if orientation < 0:
        vectors.reverse()
    return vectors


def contains_convex(polygon: list[np.ndarray], point: np.ndarray) -> bool:
    epsilon = 1e-14
    for index, current in enumerate(polygon):
        nxt = polygon[(index + 1) % len(polygon)]
        if float(np.dot(np.cross(current, nxt), point)) < -epsilon:
            return False
    return True


def brute_force_cover(vertices_lonlat: list[tuple[float, float]], resolution: int) -> np.ndarray:
    polygon = orient_convex(vertices_lonlat)
    nested_indices = np.arange(12 * (4**resolution), dtype=np.uint64)
    cell_ids = encode_nested(resolution, nested_indices)
    centers = px.center(cell_ids)
    covered = [
        cell_id
        for cell_id, (lon, lat) in zip(cell_ids, centers, strict=True)
        if contains_convex(polygon, lonlat_to_vec(float(lon), float(lat)))
    ]
    return np.asarray(covered, dtype=np.uint64)


def split_cells(cell_ids: np.ndarray, counts: np.ndarray) -> list[np.ndarray]:
    return np.split(cell_ids, np.cumsum(counts[:-1]))


class PolypixTests(unittest.TestCase):
    def test_cover_single_polygon_matches_bruteforce_oracle(self) -> None:
        polygon = [(-5.0, -5.0), (12.0, -4.0), (10.0, 9.0), (-6.0, 7.0)]
        vectors = np.asarray([lonlat_to_vec(lon, lat) for lon, lat in polygon], dtype=np.float64)

        expected = brute_force_cover(polygon, resolution=2)
        cell_ids = px.cover(px.Polygon.from_xyz(vectors), resolution=2)

        np.testing.assert_array_equal(cell_ids, expected)

    def test_cover_accepts_lonlat_polygon(self) -> None:
        polygon = np.asarray([(-5.0, -5.0), (12.0, -4.0), (10.0, 9.0), (-6.0, 7.0)], dtype=np.float64)

        expected = brute_force_cover(polygon.tolist(), resolution=2)
        cell_ids = px.cover(px.Polygon.from_lonlat(polygon), resolution=2)

        np.testing.assert_array_equal(cell_ids, expected)

    def test_cover_single_polygon_matches_bruteforce_across_antimeridian(self) -> None:
        polygon = [(170.0, -8.0), (-170.0, -8.0), (-170.0, 8.0), (170.0, 8.0)]
        vectors = np.asarray([lonlat_to_vec(lon, lat) for lon, lat in polygon], dtype=np.float64)

        cell_ids = px.cover(px.Polygon.from_xyz(vectors), resolution=2)
        np.testing.assert_array_equal(cell_ids, brute_force_cover(polygon, resolution=2))

    def test_cover_matches_bruteforce_for_spherical_edge_cases(self) -> None:
        cases = {
            "closed_ring": (
                [(-5.0, -5.0), (12.0, -4.0), (10.0, 9.0), (-6.0, 7.0), (-5.0, -5.0)],
                2,
            ),
            "reversed_orientation": (
                [(-6.0, 7.0), (10.0, 9.0), (12.0, -4.0), (-5.0, -5.0)],
                2,
            ),
            "antimeridian": (
                [(170.0, -8.0), (-170.0, -8.0), (-170.0, 8.0), (170.0, 8.0)],
                3,
            ),
            "north_pole": (
                [(-45.0, 70.0), (45.0, 70.0), (135.0, 70.0), (-135.0, 70.0)],
                3,
            ),
        }

        for name, (polygon, resolution) in cases.items():
            with self.subTest(name=name):
                vertices = np.asarray(polygon, dtype=np.float64)
                cell_ids = px.cover(px.Polygon.from_lonlat(vertices), resolution=resolution)

                np.testing.assert_array_equal(cell_ids, brute_force_cover(polygon, resolution=resolution))

    def test_cover_batched_polygons_returns_cells_and_counts(self) -> None:
        polygons = [
            (-5.0, -5.0),
            (12.0, -4.0),
            (10.0, 9.0),
            (-6.0, 7.0),
            (20.0, -10.0),
            (33.0, -10.0),
            (33.0, 0.0),
            (20.0, 0.0),
        ]
        vertices = np.asarray([lonlat_to_vec(lon, lat) for lon, lat in polygons], dtype=np.float64).reshape(2, 4, 3)

        cell_ids, counts = px.cover(px.MultiPolygon.from_xyz(vertices), resolution=2)
        first = px.cover(px.Polygon.from_xyz(vertices[0]), resolution=2)
        second = px.cover(px.Polygon.from_xyz(vertices[1]), resolution=2)
        cells_by_polygon = split_cells(cell_ids, counts)

        self.assertEqual(counts.shape, (2,))
        np.testing.assert_array_equal(counts, np.asarray([first.size, second.size], dtype=np.intp))
        np.testing.assert_array_equal(cells_by_polygon[0], first)
        np.testing.assert_array_equal(cells_by_polygon[-1], second)

    def test_cover_batched_polygons_matches_bruteforce(self) -> None:
        polygons = [
            (-5.0, -5.0),
            (12.0, -4.0),
            (10.0, 9.0),
            (-6.0, 7.0),
            (20.0, -10.0),
            (33.0, -10.0),
            (33.0, 0.0),
            (20.0, 0.0),
        ]
        vertices = np.asarray([lonlat_to_vec(lon, lat) for lon, lat in polygons], dtype=np.float64).reshape(2, 4, 3)
        cell_ids, counts = px.cover(px.MultiPolygon.from_xyz(vertices), resolution=2)
        covered = split_cells(cell_ids, counts)

        np.testing.assert_array_equal(covered[0], brute_force_cover(polygons[:4], resolution=2))
        np.testing.assert_array_equal(covered[1], brute_force_cover(polygons[4:], resolution=2))

    def test_parallel_cover_matches_single_thread(self) -> None:
        polygon = np.asarray(
            [lonlat_to_vec(lon, lat) for lon, lat in [(-5.0, -5.0), (12.0, -4.0), (10.0, 9.0), (-6.0, 7.0)]],
            dtype=np.float64,
        )
        vertices = np.repeat(polygon[np.newaxis, :, :], 300, axis=0)

        previous_threads = os.environ.get("POLYPIX_NUM_THREADS")
        try:
            os.environ["POLYPIX_NUM_THREADS"] = "1"
            expected_cell_ids, expected_counts = px.cover(px.MultiPolygon.from_xyz(vertices), resolution=3)
            os.environ["POLYPIX_NUM_THREADS"] = "4"
            actual_cell_ids, actual_counts = px.cover(px.MultiPolygon.from_xyz(vertices), resolution=3)
        finally:
            if previous_threads is None:
                os.environ.pop("POLYPIX_NUM_THREADS", None)
            else:
                os.environ["POLYPIX_NUM_THREADS"] = previous_threads

        np.testing.assert_array_equal(actual_cell_ids, expected_cell_ids)
        np.testing.assert_array_equal(actual_counts, expected_counts)

    def test_cover_accepts_ragged_polygon_list(self) -> None:
        polygon_a = np.asarray(
            [lonlat_to_vec(lon, lat) for lon, lat in [(-5.0, -5.0), (12.0, -4.0), (10.0, 9.0), (-6.0, 7.0)]],
            dtype=np.float64,
        )
        polygon_b = np.asarray(
            [lonlat_to_vec(lon, lat) for lon, lat in [(20.0, -10.0), (33.0, -10.0), (33.0, 0.0), (26.0, 5.0), (20.0, 0.0)]],
            dtype=np.float64,
        )

        cell_ids, counts = px.cover(px.MultiPolygon.from_xyz([polygon_a, polygon_b]), resolution=2)
        result = split_cells(cell_ids, counts)

        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result[0], px.cover(px.Polygon.from_xyz(polygon_a), resolution=2))
        np.testing.assert_array_equal(result[1], px.cover(px.Polygon.from_xyz(polygon_b), resolution=2))

    def test_cover_accepts_explicit_flat_lonlat_offsets(self) -> None:
        vertices = np.asarray(
            [
                (-5.0, -5.0),
                (12.0, -4.0),
                (10.0, 9.0),
                (-6.0, 7.0),
                (-45.0, 70.0),
                (45.0, 70.0),
                (135.0, 70.0),
                (-135.0, 70.0),
            ],
            dtype=np.float64,
        )

        cell_ids, counts = px.cover(px.MultiPolygon(vertices, [0, 4, 8], "lonlat"), resolution=2)
        result = split_cells(cell_ids, counts)

        self.assertEqual(counts.shape, (2,))
        np.testing.assert_array_equal(result[0], brute_force_cover(vertices[:4].tolist(), resolution=2))
        np.testing.assert_array_equal(result[1], brute_force_cover(vertices[4:].tolist(), resolution=2))

    def test_center_accepts_scalar_and_array(self) -> None:
        cell_ids = encode_nested(3, [0, 17, 123])
        first = px.center(int(cell_ids[0]))
        centers = px.center(cell_ids)

        self.assertIsInstance(first, tuple)
        np.testing.assert_allclose(centers[0], np.asarray(first))
        self.assertEqual(centers.shape, (3, 2))

    def test_center_and_boundary_accept_mixed_resolution_cells(self) -> None:
        cell_ids = np.asarray(
            [
                encode_nested(0, [0])[0],
                encode_nested(2, [17])[0],
                encode_nested(4, [123])[0],
            ],
            dtype=np.uint64,
        )

        centers = px.center(cell_ids)
        boundaries = px.boundary(cell_ids)

        self.assertEqual(centers.shape, (3, 2))
        self.assertEqual(boundaries.shape, (3, 4, 2))
        np.testing.assert_allclose(centers[1], np.asarray(px.center(int(cell_ids[1]))))
        np.testing.assert_allclose(boundaries[2], px.boundary(int(cell_ids[2])))

    def test_boundary_accepts_scalar_and_array(self) -> None:
        cell_ids = encode_nested(3, [17, 123])

        boundary = px.boundary(int(cell_ids[0]))
        boundaries = px.boundary(cell_ids)

        self.assertEqual(boundary.shape, (4, 2))
        self.assertEqual(boundaries.shape, (2, 4, 2))
        np.testing.assert_allclose(boundaries[0], boundary)

    def test_center_and_boundary_accept_empty_arrays(self) -> None:
        cell_ids = np.empty(0, dtype=np.uint64)

        self.assertEqual(px.center(cell_ids).shape, (0, 2))
        self.assertEqual(px.boundary(cell_ids).shape, (0, 4, 2))

    def test_resolution_requires_integer_without_coercion(self) -> None:
        polygon = [(-5.0, -5.0), (12.0, -4.0), (10.0, 9.0), (-6.0, 7.0)]
        vertices = np.asarray([lonlat_to_vec(lon, lat) for lon, lat in polygon], dtype=np.float64)
        geometry = px.Polygon.from_xyz(vertices)

        for resolution in (2.0, "2", True):
            with self.subTest(resolution=resolution):
                with self.assertRaises(TypeError):
                    px.cover(geometry, resolution=resolution)

        self.assertIsInstance(px.cover(geometry, resolution=np.int64(2)), np.ndarray)

    def test_cell_ids_require_integers_without_coercion(self) -> None:
        for cell_ids in (256.0, [256.9], np.asarray([256.9]), True, [True]):
            with self.subTest(cell_ids=cell_ids):
                with self.assertRaises(TypeError):
                    px.center(cell_ids)

        for cell_ids in (-1, [-1], np.asarray([-1], dtype=np.int64)):
            with self.subTest(cell_ids=cell_ids):
                with self.assertRaises(ValueError):
                    px.boundary(cell_ids)

        self.assertIsInstance(px.center(np.uint64(encode_nested(2, [0])[0])), tuple)

    def test_polygon_offsets_require_integers_without_coercion(self) -> None:
        vertices = np.asarray(
            [
                lonlat_to_vec(-5.0, -5.0),
                lonlat_to_vec(12.0, -4.0),
                lonlat_to_vec(10.0, 9.0),
                lonlat_to_vec(-6.0, 7.0),
            ],
            dtype=np.float64,
        )

        for offsets in ([0.0, 4.0], [0.2, 4.9], np.asarray([0.0, 4.0])):
            with self.subTest(offsets=offsets):
                with self.assertRaises(TypeError):
                    px.MultiPolygon(vertices, offsets, "xyz")

        with self.assertRaises(ValueError):
            px.MultiPolygon(vertices, [0, -1], "xyz")

    def test_cover_requires_polygon_objects(self) -> None:
        polygon = np.asarray(
            [
                lonlat_to_vec(-5.0, -5.0),
                lonlat_to_vec(12.0, -4.0),
                lonlat_to_vec(10.0, 9.0),
                lonlat_to_vec(-6.0, 7.0),
            ],
            dtype=np.float64,
        )

        with self.assertRaisesRegex(TypeError, "requires a Polygon or MultiPolygon"):
            px.cover(polygon, resolution=2)

    def test_cover_rejects_invalid_resolution_bounds(self) -> None:
        vertices = np.asarray(
            [
                lonlat_to_vec(lon, lat)
                for lon, lat in [(-5.0, -5.0), (12.0, -4.0), (10.0, 9.0)]
            ],
            dtype=np.float64,
        )

        for resolution in (-1, 30):
            with self.subTest(resolution=resolution):
                with self.assertRaisesRegex(ValueError, "resolution must be between 0 and 29"):
                    px.cover(px.Polygon.from_xyz(vertices), resolution=resolution)

    def test_cover_rejects_invalid_xyz_vertices(self) -> None:
        polygons = {
            "non_unit": np.asarray(
                [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            "non_finite": np.asarray(
                [[np.nan, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            "zero_length": np.asarray(
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
        }

        for name, polygon in polygons.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    px.cover(px.Polygon.from_xyz(polygon), resolution=1)

    def test_cover_rejects_invalid_lonlat_vertices(self) -> None:
        polygons = {
            "latitude_out_of_range": np.asarray(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 91.0]],
                dtype=np.float64,
            ),
            "non_finite": np.asarray(
                [[np.nan, 0.0], [1.0, 0.0], [0.0, 1.0]],
                dtype=np.float64,
            ),
        }

        for name, polygon in polygons.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    px.cover(px.Polygon.from_lonlat(polygon), resolution=1)

    def test_cover_rejects_invalid_polygon_geometry(self) -> None:
        polygons = {
            "too_few_vertices": np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
            "duplicate_consecutive_vertices": np.asarray(
                [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                dtype=np.float64,
            ),
            "non_convex": np.asarray(
                [[0.0, 0.0], [2.0, 0.0], [1.0, 1.0], [2.0, 2.0], [0.0, 2.0]],
                dtype=np.float64,
            ),
        }

        for name, polygon in polygons.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    px.cover(px.Polygon.from_lonlat(polygon), resolution=1)

    def test_center_and_boundary_reject_invalid_packed_cell_ids(self) -> None:
        invalid_cell_ids = [
            0,
            1 << 5,
            int(cell_id_prefix(0) + 12),
        ]

        for cell_id in invalid_cell_ids:
            with self.subTest(cell_id=cell_id):
                with self.assertRaisesRegex(ValueError, "valid packed HEALPix token"):
                    px.center(cell_id)
                with self.assertRaisesRegex(ValueError, "valid packed HEALPix token"):
                    px.boundary(cell_id)

    def test_cover_accepts_empty_polygon_batch(self) -> None:
        cell_ids, counts = px.cover(px.MultiPolygon.from_xyz([]), resolution=1)

        self.assertEqual(cell_ids.dtype, np.dtype("uint64"))
        self.assertEqual(counts.dtype, np.dtype("intp"))
        self.assertEqual(cell_ids.shape, (0,))
        self.assertEqual(counts.shape, (0,))

    def test_only_fast_public_endpoints_are_exposed(self) -> None:
        self.assertEqual(px.__all__, ["MultiPolygon", "Polygon", "__version__", "boundary", "center", "cover"])
        for name in [
            "cell_area_from_resolution",
            "cell_boundary",
            "cell_center",
            "cell_centers",
            "children",
            "cover_many_lonlat",
            "cover_many_unit_vectors",
            "cover_one_lonlat",
            "cover_one_unit_vectors",
            "decode_cell_id",
            "encode_cell_id",
            "parent",
        ]:
            self.assertFalse(hasattr(px, name), name)


if __name__ == "__main__":
    unittest.main()
