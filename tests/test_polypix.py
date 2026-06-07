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
    centers = px.centers(cell_ids)
    covered = [
        cell_id
        for cell_id, (lon, lat) in zip(cell_ids, centers, strict=True)
        if contains_convex(polygon, lonlat_to_vec(float(lon), float(lat)))
    ]
    return np.asarray(covered, dtype=np.uint64)


def split_coverage(coverage: px.Coverage) -> list[np.ndarray]:
    return [
        coverage.cell_ids[start:stop]
        for start, stop in zip(coverage.offsets[:-1], coverage.offsets[1:], strict=True)
    ]


class PolypixTests(unittest.TestCase):
    def test_cover_accepts_single_xyz_array(self) -> None:
        polygon = [(-5.0, -5.0), (12.0, -4.0), (10.0, 9.0), (-6.0, 7.0)]
        vertices = np.asarray([lonlat_to_vec(lon, lat) for lon, lat in polygon], dtype=np.float64)

        coverage = px.cover_footprint(vertices, resolution=2)

        self.assertIsInstance(coverage, px.Coverage)
        np.testing.assert_array_equal(coverage.offsets, np.asarray([0, coverage.cell_ids.size], dtype=np.uint64))
        np.testing.assert_array_equal(coverage.counts, np.asarray([coverage.cell_ids.size], dtype=np.intp))
        np.testing.assert_array_equal(coverage.cell_ids, brute_force_cover(polygon, resolution=2))

    def test_cover_accepts_batched_xyz_array(self) -> None:
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

        coverage = px.cover_footprint(vertices, resolution=2)
        covered = split_coverage(coverage)

        self.assertEqual(coverage.offsets.shape, (3,))
        np.testing.assert_array_equal(covered[0], brute_force_cover(polygons[:4], resolution=2))
        np.testing.assert_array_equal(covered[1], brute_force_cover(polygons[4:], resolution=2))

    def test_cover_swath_covers_consecutive_edge_intervals(self) -> None:
        left = np.asarray([lonlat_to_vec(-5.0, -5.0), lonlat_to_vec(-4.0, 0.0), lonlat_to_vec(-3.0, 5.0)])
        right = np.asarray([lonlat_to_vec(5.0, -5.0), lonlat_to_vec(4.0, 0.0), lonlat_to_vec(3.0, 5.0)])
        polygons = np.asarray(
            [
                [left[0], right[0], right[1], left[1]],
                [left[1], right[1], right[2], left[2]],
            ],
            dtype=np.float64,
        )

        expected = px.cover_footprint(polygons, resolution=3)
        actual = px.cover_swath(left, right, resolution=3)

        np.testing.assert_array_equal(actual.offsets, expected.offsets)
        np.testing.assert_array_equal(actual.cell_ids, expected.cell_ids)

    def test_cover_single_polygon_matches_bruteforce_oracle(self) -> None:
        polygon = [(-5.0, -5.0), (12.0, -4.0), (10.0, 9.0), (-6.0, 7.0)]
        vectors = np.asarray([lonlat_to_vec(lon, lat) for lon, lat in polygon], dtype=np.float64)

        expected = brute_force_cover(polygon, resolution=2)
        coverage = px.cover_footprint(vectors, resolution=2)

        np.testing.assert_array_equal(coverage.cell_ids, expected)

    def test_cover_single_polygon_matches_bruteforce_across_antimeridian(self) -> None:
        polygon = [(170.0, -8.0), (-170.0, -8.0), (-170.0, 8.0), (170.0, 8.0)]
        vectors = np.asarray([lonlat_to_vec(lon, lat) for lon, lat in polygon], dtype=np.float64)

        coverage = px.cover_footprint(vectors, resolution=2)
        np.testing.assert_array_equal(coverage.cell_ids, brute_force_cover(polygon, resolution=2))

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
                vertices = np.asarray([lonlat_to_vec(lon, lat) for lon, lat in polygon], dtype=np.float64)
                coverage = px.cover_footprint(vertices, resolution=resolution)

                np.testing.assert_array_equal(coverage.cell_ids, brute_force_cover(polygon, resolution=resolution))

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

        coverage = px.cover_footprint(vertices, resolution=2)
        first = px.cover_footprint(vertices[0], resolution=2).cell_ids
        second = px.cover_footprint(vertices[1], resolution=2).cell_ids
        cells_by_polygon = split_coverage(coverage)

        self.assertEqual(coverage.counts.shape, (2,))
        np.testing.assert_array_equal(coverage.counts, np.asarray([first.size, second.size], dtype=np.intp))
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
        coverage = px.cover_footprint(vertices, resolution=2)
        covered = split_coverage(coverage)

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
            expected = px.cover_footprint(vertices, resolution=3)
            os.environ["POLYPIX_NUM_THREADS"] = "4"
            actual = px.cover_footprint(vertices, resolution=3)
        finally:
            if previous_threads is None:
                os.environ.pop("POLYPIX_NUM_THREADS", None)
            else:
                os.environ["POLYPIX_NUM_THREADS"] = previous_threads

        np.testing.assert_array_equal(actual.cell_ids, expected.cell_ids)
        np.testing.assert_array_equal(actual.offsets, expected.offsets)

    def test_center_accepts_scalar_and_array(self) -> None:
        cell_ids = encode_nested(3, [0, 17, 123])
        first = px.centers(int(cell_ids[0]))
        centers = px.centers(cell_ids)

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

        centers = px.centers(cell_ids)
        boundaries = px.boundaries(cell_ids)

        self.assertEqual(centers.shape, (3, 2))
        self.assertEqual(boundaries.shape, (3, 4, 2))
        np.testing.assert_allclose(centers[1], np.asarray(px.centers(int(cell_ids[1]))))
        np.testing.assert_allclose(boundaries[2], px.boundaries(int(cell_ids[2])))

    def test_boundary_accepts_scalar_and_array(self) -> None:
        cell_ids = encode_nested(3, [17, 123])

        boundary = px.boundaries(int(cell_ids[0]))
        boundaries = px.boundaries(cell_ids)

        self.assertEqual(boundary.shape, (4, 2))
        self.assertEqual(boundaries.shape, (2, 4, 2))
        np.testing.assert_allclose(boundaries[0], boundary)

    def test_center_and_boundary_accept_empty_arrays(self) -> None:
        cell_ids = np.empty(0, dtype=np.uint64)

        self.assertEqual(px.centers(cell_ids).shape, (0, 2))
        self.assertEqual(px.boundaries(cell_ids).shape, (0, 4, 2))

    def test_resolution_requires_integer_without_coercion(self) -> None:
        polygon = [(-5.0, -5.0), (12.0, -4.0), (10.0, 9.0), (-6.0, 7.0)]
        vertices = np.asarray([lonlat_to_vec(lon, lat) for lon, lat in polygon], dtype=np.float64)

        for resolution in (2.0, "2", True):
            with self.subTest(resolution=resolution):
                with self.assertRaises(TypeError):
                    px.cover_footprint(vertices, resolution=resolution)

        self.assertIsInstance(px.cover_footprint(vertices, resolution=np.int64(2)), px.Coverage)

    def test_cell_ids_require_integers_without_coercion(self) -> None:
        for cell_ids in (256.0, [256.9], np.asarray([256.9]), True, [True]):
            with self.subTest(cell_ids=cell_ids):
                with self.assertRaises(TypeError):
                    px.centers(cell_ids)

        for cell_ids in (-1, [-1], np.asarray([-1], dtype=np.int64)):
            with self.subTest(cell_ids=cell_ids):
                with self.assertRaises(ValueError):
                    px.boundaries(cell_ids)

        self.assertIsInstance(px.centers(np.uint64(encode_nested(2, [0])[0])), tuple)

    def test_cover_rejects_invalid_array_shape(self) -> None:
        polygon = np.asarray(
            [
                lonlat_to_vec(-5.0, -5.0),
                lonlat_to_vec(12.0, -4.0),
                lonlat_to_vec(10.0, 9.0),
                lonlat_to_vec(-6.0, 7.0),
            ],
            dtype=np.float64,
        )

        with self.assertRaisesRegex(ValueError, "shape"):
            px.cover_footprint(polygon[:, :2], resolution=2)

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
                    px.cover_footprint(vertices, resolution=resolution)

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
                    px.cover_footprint(polygon, resolution=1)

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
                    vertices = np.asarray([lonlat_to_vec(lon, lat) for lon, lat in polygon], dtype=np.float64)
                    px.cover_footprint(vertices, resolution=1)

    def test_center_and_boundary_reject_invalid_packed_cell_ids(self) -> None:
        invalid_cell_ids = [
            0,
            1 << 5,
            int(cell_id_prefix(0) + 12),
        ]

        for cell_id in invalid_cell_ids:
            with self.subTest(cell_id=cell_id):
                with self.assertRaisesRegex(ValueError, "valid packed HEALPix token"):
                    px.centers(cell_id)
                with self.assertRaisesRegex(ValueError, "valid packed HEALPix token"):
                    px.boundaries(cell_id)

    def test_cover_accepts_empty_polygon_batch(self) -> None:
        coverage = px.cover_footprint(np.empty((0, 4, 3), dtype=np.float64), resolution=1)

        self.assertEqual(coverage.cell_ids.dtype, np.dtype("uint64"))
        self.assertEqual(coverage.counts.dtype, np.dtype("intp"))
        self.assertEqual(coverage.cell_ids.shape, (0,))
        self.assertEqual(coverage.counts.shape, (0,))

    def test_cover_accepts_empty_zero_vertex_polygon_batch(self) -> None:
        coverage = px.cover_footprint(np.empty((0, 0, 3), dtype=np.float64), resolution=1)

        self.assertEqual(coverage.cell_ids.dtype, np.dtype("uint64"))
        self.assertEqual(coverage.offsets.dtype, np.dtype("uint64"))
        self.assertEqual(coverage.cell_ids.shape, (0,))
        np.testing.assert_array_equal(coverage.offsets, np.asarray([0], dtype=np.uint64))
        self.assertEqual(coverage.counts.shape, (0,))

    def test_cover_rejects_non_empty_zero_vertex_polygon_batch(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one vertex"):
            px.cover_footprint(np.empty((1, 0, 3), dtype=np.float64), resolution=1)

    def test_only_fast_public_endpoints_are_exposed(self) -> None:
        self.assertEqual(
            px.__all__,
            [
                "Coverage",
                "__version__",
                "boundaries",
                "centers",
                "cover_footprint",
                "cover_swath",
            ],
        )
        for name in [
            "cell_area_from_resolution",
            "cell_boundary",
            "cell_center",
            "cell_centers",
            "cover",
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
