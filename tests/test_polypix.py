from __future__ import annotations

import math
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import polypix as px


def lonlat_to_vec(lon_deg: float, lat_deg: float) -> np.ndarray:
    lon = math.radians(lon_deg)
    lat = math.radians(lat_deg)
    cos_lat = math.cos(lat)
    return np.asarray(
        [cos_lat * math.cos(lon), cos_lat * math.sin(lon), math.sin(lat)],
        dtype=np.float64,
    )


def vectors(vertices_lonlat: list[tuple[float, float]]) -> np.ndarray:
    return np.asarray(
        [lonlat_to_vec(lon, lat) for lon, lat in vertices_lonlat],
        dtype=np.float64,
    )


def orient_convex(vertices_lonlat: list[tuple[float, float]]) -> list[np.ndarray]:
    polygon = [lonlat_to_vec(lon, lat) for lon, lat in vertices_lonlat]
    if np.allclose(polygon[0], polygon[-1]):
        polygon = polygon[:-1]
    interior = np.sum(polygon, axis=0)
    interior /= np.linalg.norm(interior)
    orientation = sum(
        float(
            np.dot(
                np.cross(current, polygon[(index + 1) % len(polygon)]),
                interior,
            )
        )
        for index, current in enumerate(polygon)
    )
    if orientation < 0:
        polygon.reverse()
    return polygon


def contains_convex(polygon: list[np.ndarray], point: np.ndarray) -> bool:
    epsilon = 1e-14
    return all(
        float(
            np.dot(
                np.cross(current, polygon[(index + 1) % len(polygon)]),
                point,
            )
        )
        >= -epsilon
        for index, current in enumerate(polygon)
    )


def reference_ring_centers(resolution: int) -> np.ndarray:
    """Independent scalar HEALPix RING equations used by the coverage oracle."""
    nside = 1 << resolution
    pixel_count = 12 * nside * nside
    cap_cells = 2 * nside * (nside - 1)
    centers = np.empty((pixel_count, 3), dtype=np.float64)
    for cell in range(pixel_count):
        if cell < cap_cells:
            ring = int((1.0 + math.sqrt(1.0 + 2.0 * cell)) / 2.0)
            offset = cell - 2 * ring * (ring - 1) + 1
            z = 1.0 - ring * ring / (3.0 * nside * nside)
            longitude = (offset - 0.5) * math.pi / (2.0 * ring)
        elif cell < pixel_count - cap_cells:
            index = cell - cap_cells
            ring = index // (4 * nside) + nside
            offset = index % (4 * nside)
            shift = 0.5 if (ring + nside) % 2 == 0 else 0.0
            z = (2 * nside - ring) * 2.0 / (3.0 * nside)
            longitude = (offset + shift) * math.pi / (2.0 * nside)
        else:
            reversed_cell = pixel_count - 1 - cell
            ring = int((1.0 + math.sqrt(1.0 + 2.0 * reversed_cell)) / 2.0)
            start = pixel_count - 2 * ring * (ring + 1)
            offset = cell - start
            z = -(1.0 - ring * ring / (3.0 * nside * nside))
            longitude = (offset + 0.5) * math.pi / (2.0 * ring)
        radial = math.sqrt(max(0.0, 1.0 - z * z))
        centers[cell] = (
            radial * math.cos(longitude),
            radial * math.sin(longitude),
            z,
        )
    return centers


def brute_force_cover(
    vertices_lonlat: list[tuple[float, float]],
    resolution: int,
) -> np.ndarray:
    polygon = orient_convex(vertices_lonlat)
    cells = np.arange(12 * (4**resolution), dtype=np.uint64)
    cell_centers = reference_ring_centers(resolution)
    return cells[
        np.asarray(
            [contains_convex(polygon, center) for center in cell_centers],
            dtype=np.bool_,
        )
    ]


def split_coverage(coverage: px.Coverage) -> list[np.ndarray]:
    return [
        coverage.cells[start:stop]
        for start, stop in zip(
            coverage.offsets[:-1],
            coverage.offsets[1:],
            strict=True,
        )
    ]


def post_filter_coverage(
    coverage: px.Coverage,
    candidate_cells: np.ndarray,
) -> list[np.ndarray]:
    return [
        cells[np.isin(cells, candidate_cells)] for cells in split_coverage(coverage)
    ]


class PolypixTests(unittest.TestCase):
    def assertCellsEqual(self, actual: np.ndarray, expected: np.ndarray) -> None:
        np.testing.assert_array_equal(np.sort(actual), np.sort(expected))

    def assertSegmentsEqual(
        self,
        actual: px.Coverage,
        expected: list[np.ndarray],
    ) -> None:
        actual_segments = split_coverage(actual)
        self.assertEqual(len(actual_segments), len(expected))
        for actual_segment, expected_segment in zip(
            actual_segments,
            expected,
            strict=True,
        ):
            self.assertCellsEqual(actual_segment, expected_segment)

    def test_centers_match_independent_ring_equations_through_resolution_6(
        self,
    ) -> None:
        for resolution in range(7):
            expected = reference_ring_centers(resolution)
            cells = np.arange(expected.shape[0], dtype=np.uint64)
            np.testing.assert_allclose(
                px.centers(cells, resolution),
                expected,
                rtol=0.0,
                atol=2e-15,
            )

    def test_cover_accepts_single_xyz_array(self) -> None:
        polygon = [(-5.0, -5.0), (12.0, -4.0), (10.0, 9.0), (-6.0, 7.0)]

        coverage = px.cover_footprint(vectors(polygon), resolution=2)

        self.assertIsInstance(coverage, px.Coverage)
        self.assertEqual(coverage.resolution, 2)
        self.assertEqual(coverage.cells.dtype, np.dtype("uint64"))
        np.testing.assert_array_equal(
            coverage.offsets,
            np.asarray([0, coverage.cells.size], dtype=np.uint64),
        )
        np.testing.assert_array_equal(
            coverage.counts,
            np.asarray([coverage.cells.size], dtype=np.intp),
        )
        self.assertCellsEqual(
            coverage.cells,
            brute_force_cover(polygon, resolution=2),
        )

    def test_cover_accepts_dense_and_ragged_batches(self) -> None:
        polygons = [
            [(-5.0, -5.0), (12.0, -4.0), (10.0, 9.0), (-6.0, 7.0)],
            [
                (20.0, -10.0),
                (33.0, -10.0),
                (36.0, -4.0),
                (33.0, 0.0),
                (20.0, 0.0),
            ],
        ]
        ragged = [vectors(polygon) for polygon in polygons]

        coverage = px.cover_footprint(ragged, resolution=2)
        expected = [
            px.cover_footprint(polygon, resolution=2).cells for polygon in ragged
        ]

        self.assertSegmentsEqual(coverage, expected)
        np.testing.assert_array_equal(
            coverage.counts,
            np.asarray([cells.size for cells in expected], dtype=np.intp),
        )

        dense = np.stack((ragged[0], vectors(polygons[0]) * 7.0))
        dense_coverage = px.cover_footprint(dense, resolution=2)
        self.assertSegmentsEqual(dense_coverage, [expected[0], expected[0]])

    def test_repeated_closing_vertex_is_representation_independent(self) -> None:
        triangle = vectors([(-5.0, -5.0), (5.0, -5.0), (0.0, 5.0)])
        closed = np.vstack((triangle, triangle[0]))

        open_coverage = px.cover_footprint(triangle, resolution=3, threads=1)
        dense_closed = px.cover_footprint(closed, resolution=3, threads=1)
        ragged_closed = px.cover_footprint([closed, triangle], resolution=3, threads=1)

        np.testing.assert_array_equal(dense_closed.cells, open_coverage.cells)
        self.assertSegmentsEqual(
            ragged_closed,
            [open_coverage.cells, open_coverage.cells],
        )

    def test_cover_strip_covers_consecutive_edge_intervals(self) -> None:
        left = np.asarray(
            [
                lonlat_to_vec(-5.0, -5.0),
                lonlat_to_vec(-4.0, 0.0),
                lonlat_to_vec(-3.0, 5.0),
            ]
        )
        right = np.asarray(
            [
                lonlat_to_vec(5.0, -5.0),
                lonlat_to_vec(4.0, 0.0),
                lonlat_to_vec(3.0, 5.0),
            ]
        )
        footprints = np.asarray(
            [
                [left[0], right[0], right[1], left[1]],
                [left[1], right[1], right[2], left[2]],
            ]
        )

        expected = px.cover_footprint(footprints, resolution=3)
        actual = px.cover_strip(left, right, resolution=3)

        np.testing.assert_array_equal(actual.offsets, expected.offsets)
        np.testing.assert_array_equal(actual.cells, expected.cells)

    def test_strip_errors_name_the_invalid_edge(self) -> None:
        left = np.asarray([lonlat_to_vec(-5.0, -5.0), lonlat_to_vec(-5.0, 5.0)])
        right = np.asarray([lonlat_to_vec(5.0, -5.0), lonlat_to_vec(5.0, 5.0)])
        for name in ("left_edge_xyz", "right_edge_xyz"):
            invalid_left = left.copy()
            invalid_right = right.copy()
            if name == "left_edge_xyz":
                invalid_left[0, 0] = np.inf
            else:
                invalid_right[0, 0] = np.inf
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, name):
                    px.cover_strip(invalid_left, invalid_right, resolution=3)

    def test_candidate_coverage_matches_post_filtering(self) -> None:
        footprints = np.asarray(
            [
                vectors([(-15.0, -8.0), (8.0, -8.0), (8.0, 8.0), (-15.0, 8.0)]),
                vectors([(20.0, -8.0), (43.0, -8.0), (43.0, 8.0), (20.0, 8.0)]),
            ]
        )
        unfiltered = px.cover_footprint(footprints, resolution=4)
        selected = unfiltered.cells[::2]
        candidates = np.concatenate(
            (selected[::-1], selected[:2], np.asarray([0], dtype=np.uint64))
        )

        actual = px.cover_footprint(
            footprints,
            resolution=4,
            candidate_cells=candidates,
        )

        self.assertSegmentsEqual(
            actual,
            post_filter_coverage(unfiltered, candidates),
        )

    def test_strip_candidate_coverage_matches_post_filtering(self) -> None:
        left = np.asarray(
            [
                lonlat_to_vec(-8.0, -12.0),
                lonlat_to_vec(-7.0, 0.0),
                lonlat_to_vec(-6.0, 12.0),
            ]
        )
        right = np.asarray(
            [
                lonlat_to_vec(8.0, -12.0),
                lonlat_to_vec(7.0, 0.0),
                lonlat_to_vec(6.0, 12.0),
            ]
        )
        unfiltered = px.cover_strip(left, right, resolution=4)
        candidates = unfiltered.cells[::2][::-1]

        actual = px.cover_strip(
            left,
            right,
            resolution=4,
            candidate_cells=candidates,
        )

        self.assertSegmentsEqual(
            actual,
            post_filter_coverage(unfiltered, candidates),
        )

    def test_empty_candidates_preserve_segment_offsets(self) -> None:
        polygon = vectors([(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)])
        coverage = px.cover_footprint(
            np.repeat(polygon[np.newaxis, :, :], 3, axis=0),
            resolution=4,
            candidate_cells=[],
        )

        np.testing.assert_array_equal(
            coverage.cells,
            np.empty(0, dtype=np.uint64),
        )
        np.testing.assert_array_equal(
            coverage.offsets,
            np.zeros(4, dtype=np.uint64),
        )

    def test_candidate_inputs_are_validated(self) -> None:
        polygon = vectors([(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)])
        invalid_candidates = ([1.5], [True], [-1])
        for candidates in invalid_candidates:
            with self.subTest(candidates=candidates):
                with self.assertRaises((TypeError, ValueError)):
                    px.cover_footprint(
                        polygon,
                        resolution=4,
                        candidate_cells=candidates,
                    )

        with self.assertRaisesRegex(ValueError, "scalar or one-dimensional"):
            px.cover_footprint(
                polygon,
                resolution=4,
                candidate_cells=np.empty((1, 0), dtype=np.uint64),
            )
        with self.assertRaisesRegex(ValueError, "valid RING indices"):
            px.cover_footprint(
                polygon,
                resolution=4,
                candidate_cells=[12 * 4**4],
            )

    def test_candidate_coverage_at_antimeridian_and_pole(self) -> None:
        cases = [
            [(170.0, -8.0), (-170.0, -8.0), (-170.0, 8.0), (170.0, 8.0)],
            [(-45.0, 70.0), (45.0, 70.0), (135.0, 70.0), (-135.0, 70.0)],
        ]

        for polygon in cases:
            with self.subTest(polygon=polygon):
                footprint = vectors(polygon)
                unfiltered = px.cover_footprint(footprint, resolution=4)
                candidates = unfiltered.cells[::2][::-1]
                actual = px.cover_footprint(
                    footprint,
                    resolution=4,
                    candidate_cells=candidates,
                )
                self.assertCellsEqual(actual.cells, candidates)

    def test_cell_center_on_boundary_is_covered(self) -> None:
        resolution = 3
        cell = np.uint64(123)
        center = px.centers(cell, resolution)[0]
        reference = (
            np.asarray([0.0, 0.0, 1.0])
            if abs(center[2]) < 0.9
            else np.asarray([0.0, 1.0, 0.0])
        )
        tangent = np.cross(reference, center)
        tangent /= np.linalg.norm(tangent)
        inward = np.cross(center, tangent)
        inward /= np.linalg.norm(inward)

        def offset_point(x: float, y: float) -> np.ndarray:
            point = center + x * tangent + y * inward
            return point / np.linalg.norm(point)

        footprint = np.asarray(
            [
                offset_point(-0.05, 0.0),
                offset_point(0.05, 0.0),
                offset_point(0.05, 0.05),
                offset_point(-0.05, 0.05),
            ]
        )

        unfiltered = px.cover_footprint(footprint, resolution)
        restricted = px.cover_footprint(
            footprint,
            resolution,
            candidate_cells=[cell],
        )

        self.assertIn(cell, unfiltered.cells)
        np.testing.assert_array_equal(restricted.cells, [cell])

    def test_cover_matches_bruteforce_for_spherical_cases(self) -> None:
        cases = {
            "ordinary": (
                [(-5.0, -5.0), (12.0, -4.0), (10.0, 9.0), (-6.0, 7.0)],
                2,
            ),
            "closed_ring": (
                [
                    (-5.0, -5.0),
                    (12.0, -4.0),
                    (10.0, 9.0),
                    (-6.0, 7.0),
                    (-5.0, -5.0),
                ],
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
            "south_pole": (
                [(-135.0, -70.0), (135.0, -70.0), (45.0, -70.0), (-45.0, -70.0)],
                3,
            ),
            "near_hemisphere": (
                [(-89.0, -40.0), (89.0, -40.0), (89.0, 40.0), (-89.0, 40.0)],
                3,
            ),
        }

        for name, (polygon, resolution) in cases.items():
            with self.subTest(name=name):
                coverage = px.cover_footprint(
                    vectors(polygon),
                    resolution=resolution,
                )
                self.assertCellsEqual(
                    coverage.cells,
                    brute_force_cover(polygon, resolution),
                )

    def test_fixed_seed_random_footprints_match_independent_oracle(self) -> None:
        random = np.random.default_rng(20260727)
        for _ in range(100):
            longitude = float(random.uniform(-180.0, 180.0))
            latitude = float(random.uniform(-60.0, 60.0))
            radius = float(random.uniform(0.1, 5.0))
            vertex_count = int(random.integers(3, 7))
            polygon = [
                (
                    longitude
                    + radius
                    * math.cos(2.0 * math.pi * index / vertex_count)
                    / math.cos(math.radians(latitude)),
                    latitude
                    + radius * math.sin(2.0 * math.pi * index / vertex_count),
                )
                for index in range(vertex_count)
            ]
            actual = px.cover_footprint(vectors(polygon), resolution=3, threads=1)
            self.assertCellsEqual(
                actual.cells,
                brute_force_cover(polygon, resolution=3),
            )

    def test_historical_thin_southern_polygon_uses_intended_side(self) -> None:
        # This input exposed a side-selection defect in the former CDS-backed
        # implementation. Keep the geometry even though CDS is no longer used.
        longitude = -45.006968888513825
        latitude = -35.65525862937339
        half_width = 6.474936777280457
        half_height = 0.5725726026463771
        polygon = [
            (longitude - half_width, latitude - half_height),
            (longitude + half_width, latitude - half_height),
            (longitude + half_width, latitude + half_height),
            (longitude - half_width, latitude + half_height),
        ]

        actual = px.cover_footprint(vectors(polygon), resolution=5, threads=1)
        expected = brute_force_cover(polygon, resolution=5)

        self.assertGreater(expected.size, 0)
        self.assertCellsEqual(actual.cells, expected)

    def test_thread_counts_produce_identical_ordered_results(self) -> None:
        polygon = vectors([(-5.0, -5.0), (12.0, -4.0), (10.0, 9.0), (-6.0, 7.0)])
        footprints = np.repeat(polygon[np.newaxis, :, :], 300, axis=0)
        candidates = px.cover_footprint(polygon, resolution=4).cells[::2]

        for candidate_cells in (None, candidates):
            with self.subTest(restricted=candidate_cells is not None):
                single_threaded = px.cover_footprint(
                    footprints,
                    resolution=4,
                    candidate_cells=candidate_cells,
                    threads=1,
                )
                parallel = px.cover_footprint(
                    footprints,
                    resolution=4,
                    candidate_cells=candidate_cells,
                    threads=4,
                )
                automatic = px.cover_footprint(
                    footprints,
                    resolution=4,
                    candidate_cells=candidate_cells,
                )
                np.testing.assert_array_equal(
                    parallel.cells,
                    single_threaded.cells,
                )
                np.testing.assert_array_equal(
                    parallel.offsets,
                    single_threaded.offsets,
                )
                np.testing.assert_array_equal(
                    automatic.cells,
                    single_threaded.cells,
                )
                np.testing.assert_array_equal(
                    automatic.offsets,
                    single_threaded.offsets,
                )

    def test_thread_count_must_be_a_positive_integer(self) -> None:
        polygon = vectors([(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)])
        for threads in (0, -1, True):
            with self.subTest(threads=threads):
                with self.assertRaises((TypeError, ValueError)):
                    px.cover_footprint(polygon, resolution=2, threads=threads)

    def test_concurrent_calls_are_deterministic(self) -> None:
        polygon = vectors([(-5.0, -5.0), (12.0, -4.0), (10.0, 9.0), (-6.0, 7.0)])
        footprints = np.repeat(polygon[np.newaxis, :, :], 300, axis=0)
        expected = px.cover_footprint(
            footprints,
            resolution=3,
            threads=2,
        )

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(
                executor.map(
                    lambda _: px.cover_footprint(
                        footprints,
                        resolution=3,
                        threads=2,
                    ),
                    range(4),
                )
            )

        for result in results:
            np.testing.assert_array_equal(result.cells, expected.cells)
            np.testing.assert_array_equal(result.offsets, expected.offsets)

    def test_centers_and_boundaries_return_unit_xyz(self) -> None:
        cells = np.asarray([0, 17, 123], dtype=np.uint64)

        center_vectors = px.centers(cells, resolution=3)
        boundary_vectors = px.boundaries(cells, resolution=3)
        scalar_center = px.centers(int(cells[0]), resolution=3)
        scalar_boundary = px.boundaries(int(cells[0]), resolution=3)

        self.assertEqual(center_vectors.shape, (3, 3))
        self.assertEqual(boundary_vectors.shape, (3, 4, 3))
        self.assertEqual(scalar_center.shape, (1, 3))
        self.assertEqual(scalar_boundary.shape, (1, 4, 3))
        np.testing.assert_allclose(center_vectors[0], scalar_center[0])
        np.testing.assert_allclose(boundary_vectors[0], scalar_boundary[0])
        np.testing.assert_allclose(np.linalg.norm(center_vectors, axis=1), 1.0)
        np.testing.assert_allclose(
            np.linalg.norm(boundary_vectors, axis=2),
            1.0,
        )

    def test_empty_centers_and_boundaries_have_stable_shapes(self) -> None:
        cells = np.empty(0, dtype=np.uint64)
        self.assertEqual(px.centers(cells, resolution=3).shape, (0, 3))
        self.assertEqual(px.boundaries(cells, resolution=3).shape, (0, 4, 3))

    def test_resolution_requires_an_integer_in_range(self) -> None:
        footprint = vectors([(-5.0, -5.0), (12.0, -4.0), (10.0, 9.0), (-6.0, 7.0)])
        for resolution in (2.0, "2", True):
            with self.subTest(resolution=resolution):
                with self.assertRaises(TypeError):
                    px.cover_footprint(footprint, resolution=resolution)
        for resolution in (-1, 30):
            with self.subTest(resolution=resolution):
                with self.assertRaisesRegex(ValueError, "between 0 and 29"):
                    px.cover_footprint(footprint, resolution=resolution)

        self.assertIsInstance(
            px.cover_footprint(footprint, resolution=np.int64(2)),
            px.Coverage,
        )

    def test_cell_indices_require_valid_integers(self) -> None:
        for cells in (256.0, [256.9], np.asarray([256.9]), True, [True]):
            with self.subTest(cells=cells):
                with self.assertRaises(TypeError):
                    px.centers(cells, resolution=3)
        for cells in (-1, [-1], np.asarray([-1], dtype=np.int64)):
            with self.subTest(cells=cells):
                with self.assertRaises(ValueError):
                    px.boundaries(cells, resolution=3)
        with self.assertRaisesRegex(ValueError, "valid RING indices"):
            px.centers([12 * 4**3], resolution=3)

    def test_cover_rejects_invalid_array_shape(self) -> None:
        footprint = vectors([(-5.0, -5.0), (12.0, -4.0), (10.0, 9.0), (-6.0, 7.0)])
        with self.assertRaisesRegex(ValueError, "shape"):
            px.cover_footprint(footprint[:, :2], resolution=2)

    def test_cover_normalizes_arbitrary_vectors(self) -> None:
        footprint = vectors([(-5.0, -5.0), (12.0, -4.0), (10.0, 9.0), (-6.0, 7.0)])
        scales = np.asarray([2.0, 1e300, 1e-300, 7.0])[:, np.newaxis]

        expected = px.cover_footprint(footprint, resolution=3)
        actual = px.cover_footprint(footprint * scales, resolution=3)

        np.testing.assert_array_equal(actual.cells, expected.cells)
        np.testing.assert_array_equal(actual.offsets, expected.offsets)

    def test_cover_rejects_invalid_xyz_vectors(self) -> None:
        valid = vectors([(-5.0, -5.0), (12.0, -4.0), (10.0, 9.0), (-6.0, 7.0)])
        invalid = {
            "non_finite": valid.copy(),
            "zero_length": valid.copy(),
        }
        invalid["non_finite"][0, 0] = np.nan
        invalid["zero_length"][0] = 0.0

        for name, footprint in invalid.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    px.cover_footprint(footprint, resolution=1)
        with self.assertRaisesRegex(TypeError, "complex"):
            px.cover_footprint(valid.astype(np.complex128), resolution=1)

    def test_cover_rejects_invalid_polygon_geometry(self) -> None:
        invalid_polygons = {
            "too_few_vertices": [(-1.0, 0.0), (1.0, 0.0)],
            "duplicate_vertices": [
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
            ],
            "non_convex": [
                (0.0, 0.0),
                (2.0, 0.0),
                (1.0, 1.0),
                (2.0, 2.0),
                (0.0, 2.0),
            ],
        }

        for name, polygon in invalid_polygons.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    px.cover_footprint(vectors(polygon), resolution=1)

    def test_cover_accepts_empty_batches(self) -> None:
        for shape in ((0, 4, 3), (0, 0, 3)):
            with self.subTest(shape=shape):
                coverage = px.cover_footprint(
                    np.empty(shape, dtype=np.float64),
                    resolution=1,
                )
                self.assertEqual(coverage.cells.dtype, np.dtype("uint64"))
                self.assertEqual(coverage.offsets.dtype, np.dtype("uint64"))
                self.assertEqual(coverage.cells.shape, (0,))
                np.testing.assert_array_equal(coverage.offsets, [0])
                self.assertEqual(coverage.counts.shape, (0,))

    def test_cover_rejects_nonempty_zero_vertex_batch(self) -> None:
        with self.assertRaises(ValueError):
            px.cover_footprint(
                np.empty((1, 0, 3), dtype=np.float64),
                resolution=1,
            )

    def test_only_target_public_endpoints_are_exposed(self) -> None:
        self.assertEqual(
            px.__all__,
            [
                "Coverage",
                "__version__",
                "boundaries",
                "centers",
                "cover_footprint",
                "cover_strip",
            ],
        )
        for name in [
            "cell_area_from_resolution",
            "cell_boundary",
            "cell_center",
            "cell_centers",
            "children",
            "cover",
            "decode_cell_id",
            "encode_cell_id",
            "parent",
            "cover_swath",
        ]:
            self.assertFalse(hasattr(px, name), name)


if __name__ == "__main__":
    unittest.main()
