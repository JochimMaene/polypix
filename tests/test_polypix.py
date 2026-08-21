from __future__ import annotations

import math
import unittest
from concurrent.futures import ThreadPoolExecutor
from functools import cache

import numpy as np

import polypix as px


class ArrayOnly:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def __array__(
        self,
        dtype: np.dtype | None = None,
        copy: bool | None = None,
    ) -> np.ndarray:
        return np.asarray(self.values, dtype=dtype)


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


@cache
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
    return independent_cover_xyz(vectors(vertices_lonlat), resolution)


def independent_cover_xyz(vertices_xyz: np.ndarray, resolution: int) -> np.ndarray:
    """Brute-force coverage using independent RING centers and vectorized planes."""
    polygon = np.asarray(vertices_xyz, dtype=np.float64)
    polygon = polygon / np.linalg.norm(polygon, axis=1)[:, np.newaxis]
    if np.allclose(polygon[0], polygon[-1], rtol=0.0, atol=1e-12):
        polygon = polygon[:-1]
    interior = np.sum(polygon, axis=0)
    interior /= np.linalg.norm(interior)
    normals = np.cross(polygon, np.roll(polygon, -1, axis=0))
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]
    if float(np.sum(normals @ interior)) < 0.0:
        polygon = polygon[::-1]
        normals = np.cross(polygon, np.roll(polygon, -1, axis=0))
        normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]

    cells = np.arange(12 * (4**resolution), dtype=np.uint64)
    cell_centers = reference_ring_centers(resolution)
    return cells[np.all(cell_centers @ normals.T >= -1e-14, axis=1)]


def independent_cover_cap(
    center_xyz: np.ndarray,
    radius_rad: float,
    resolution: int,
) -> np.ndarray:
    """Brute-force spherical-cap coverage using independent RING centers."""
    center = np.asarray(center_xyz, dtype=np.float64)
    center /= np.linalg.norm(center)
    cells = np.arange(12 * (4**resolution), dtype=np.uint64)
    cell_centers = reference_ring_centers(resolution)
    cross_norm = np.linalg.norm(np.cross(cell_centers, center), axis=1)
    angular_distance = np.arctan2(cross_norm, cell_centers @ center)
    contained = angular_distance <= radius_rad + 1e-14
    return cells[contained]


def regular_spherical_polygon(
    axis: np.ndarray,
    radius: float,
    vertex_count: int,
) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    seed = (
        np.asarray([1.0, 0.0, 0.0])
        if abs(axis[2]) > 0.9
        else np.asarray([0.0, 0.0, 1.0])
    )
    tangent_x = np.cross(seed, axis)
    tangent_x /= np.linalg.norm(tangent_x)
    tangent_y = np.cross(axis, tangent_x)
    angles = np.arange(vertex_count) * (2.0 * math.pi / vertex_count)
    return math.cos(radius) * axis + math.sin(radius) * (
        np.cos(angles)[:, np.newaxis] * tangent_x
        + np.sin(angles)[:, np.newaxis] * tangent_y
    )


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
            self.assertEqual(
                actual_segment.size,
                np.unique(actual_segment).size,
                "coverage segment contains duplicate cells",
            )
            self.assertCellsEqual(actual_segment, expected_segment)

    def test_centers_match_independent_ring_equations_through_resolution_7(
        self,
    ) -> None:
        for resolution in range(8):
            expected = reference_ring_centers(resolution)
            cells = np.arange(expected.shape[0], dtype=np.uint64)
            # The independent scalar oracle uses sqrt(1 - z*z), whose polar
            # cancellation becomes visible one resolution before the production
            # kernel's factored formulation.
            tolerance = 6e-15 if resolution == 7 else 2e-15
            np.testing.assert_allclose(
                px.cell_centers(cells, resolution),
                expected,
                rtol=0.0,
                atol=tolerance,
            )

    def test_cover_accepts_single_xyz_array(self) -> None:
        polygon = [(-5.0, -5.0), (12.0, -4.0), (10.0, 9.0), (-6.0, 7.0)]

        coverage = px.cover_convex_polygon(vectors(polygon), resolution=2)

        self.assertIsInstance(coverage, px.Coverage)
        self.assertEqual(coverage.resolution, 2)
        self.assertEqual(coverage.cells.dtype, np.dtype("int64"))
        np.testing.assert_array_equal(
            coverage.offsets,
            np.asarray([0, coverage.cells.size], dtype=np.uint64),
        )
        np.testing.assert_array_equal(
            coverage.segment_sizes,
            np.asarray([coverage.cells.size], dtype=np.intp),
        )
        self.assertCellsEqual(
            coverage.cells,
            brute_force_cover(polygon, resolution=2),
        )

    def test_cover_cap_accepts_one_center_and_batched_radii(self) -> None:
        centers = np.asarray(
            [
                lonlat_to_vec(15.0, 25.0),
                7.0 * lonlat_to_vec(179.0, -30.0),
                lonlat_to_vec(-45.0, 89.0),
            ]
        )
        radii = np.radians([0.0, 12.5, 35.0])

        batch = px.cover_cap(centers, radii, resolution=5, threads=1)
        expected = [
            independent_cover_cap(center, float(radius), resolution=5)
            for center, radius in zip(centers, radii, strict=True)
        ]
        self.assertSegmentsEqual(batch, expected)

        single = px.cover_cap(centers[1], radii[1], resolution=5, threads=1)
        self.assertSegmentsEqual(single, [expected[1]])

        shared_radius = px.cover_cap(centers, radii[1], resolution=5, threads=1)
        self.assertSegmentsEqual(
            shared_radius,
            [
                independent_cover_cap(center, float(radii[1]), resolution=5)
                for center in centers
            ],
        )

    def test_cover_cap_handles_exact_boundaries_and_full_sphere(self) -> None:
        resolution = 4
        boundary_cell = np.uint64(321)
        boundary_center = px.cell_centers(boundary_cell, resolution)[0]
        axis = lonlat_to_vec(35.0, -20.0)
        radius = math.atan2(
            float(np.linalg.norm(np.cross(axis, boundary_center))),
            float(np.dot(axis, boundary_center)),
        )

        on_boundary = px.cover_cap(axis, radius, resolution, threads=1)
        just_outside = px.cover_cap(axis, radius - 1e-12, resolution, threads=1)
        point = px.cover_cap(boundary_center, 0.0, resolution, threads=1)
        whole_sphere = px.cover_cap(axis, math.pi, resolution, threads=1)

        self.assertIn(boundary_cell, on_boundary.cells)
        self.assertNotIn(boundary_cell, just_outside.cells)
        self.assertIn(boundary_cell, point.cells)
        np.testing.assert_array_equal(
            np.sort(whole_sphere.cells),
            np.arange(12 * 4**resolution, dtype=np.uint64),
        )

    def test_cover_cap_handles_empty_batches_poles_and_the_longitude_seam(self) -> None:
        resolution = 4
        empty = np.empty((0, 3), dtype=np.float64)
        for radii in (0.2, np.empty(0, dtype=np.float64)):
            with self.subTest(radii=np.asarray(radii).shape):
                coverage = px.cover_cap(empty, radii, resolution)
                np.testing.assert_array_equal(coverage.cells, [])
                np.testing.assert_array_equal(coverage.offsets, [0])
                np.testing.assert_array_equal(
                    px.cover_cap(
                        empty, radii, resolution=resolution, reduce=px.Count()
                    ),
                    np.zeros(12 * 4**resolution, dtype=np.int64),
                )
                np.testing.assert_array_equal(
                    px.cover_cap(
                        empty,
                        radii,
                        resolution=resolution,
                        candidate_cells=[7, 1, 7],
                        reduce=px.Count(),
                    ),
                    [0, 0, 0],
                )

        centers = np.asarray(([0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]))
        radii = np.asarray([0.2, 0.35, 0.4])
        actual = px.cover_cap(centers, radii, resolution, threads=1)
        self.assertSegmentsEqual(
            actual,
            [
                independent_cover_cap(center, float(radius), resolution)
                for center, radius in zip(centers, radii, strict=True)
            ],
        )

    def test_cap_candidates_have_set_semantics(self) -> None:
        resolution = 4
        center = lonlat_to_vec(179.0, 20.0)
        unfiltered = px.cover_cap(center, 0.5, resolution)
        selected = unfiltered.cells[::3]
        candidates = np.concatenate((selected[::-1], selected, selected[:2]))

        restricted = px.cover_cap(
            center,
            0.5,
            resolution,
            candidate_cells=candidates,
        )

        np.testing.assert_array_equal(restricted.cells, np.unique(selected))
        empty = px.cover_cap(center, 0.5, resolution, candidate_cells=[])
        np.testing.assert_array_equal(empty.cells, [])
        np.testing.assert_array_equal(empty.offsets, [0, 0])

    def test_count_caps_per_cell_matches_stored_coverage(self) -> None:
        random = np.random.default_rng(20260812)
        resolution = 4
        centers = random.normal(size=(128, 3))
        radii = random.uniform(0.0, math.pi / 3.0, size=centers.shape[0])
        coverage = px.cover_cap(centers, radii, resolution, threads=1)
        expected = np.bincount(
            coverage.cells.astype(np.int64, copy=False),
            minlength=12 * 4**resolution,
        ).astype(np.int64, copy=False)

        serial = px.cover_cap(
            centers, radii, resolution=resolution, reduce=px.Count(), threads=1
        )
        parallel = px.cover_cap(
            centers, radii, resolution=resolution, reduce=px.Count(), threads=4
        )

        self.assertEqual(serial.dtype, np.dtype("int64"))
        np.testing.assert_array_equal(serial, expected)
        np.testing.assert_array_equal(parallel, expected)
        requested = np.asarray([7, 1, 7, 100, 0], dtype=np.uint64)
        queried = px.cover_cap(
            centers,
            radii,
            resolution=resolution,
            candidate_cells=requested,
            reduce=px.Count(),
            threads=1,
        )
        np.testing.assert_array_equal(
            queried, expected[requested.astype(np.int64, copy=False)]
        )

    def test_count_caps_per_cell_handles_endpoint_radii_and_high_resolution_queries(
        self,
    ) -> None:
        resolution = 3
        centers = px.cell_centers(
            np.asarray([3, 17, 42, 71], dtype=np.uint64), resolution
        )
        radii = np.asarray([0.0, math.pi / 2.0, 3.0 * math.pi / 4.0, math.pi])
        coverage = px.cover_cap(centers, radii, resolution, threads=1)
        expected = np.bincount(
            coverage.cells.astype(np.int64, copy=False),
            minlength=12 * 4**resolution,
        ).astype(np.int64, copy=False)
        np.testing.assert_array_equal(
            px.cover_cap(
                centers, radii, resolution=resolution, reduce=px.Count(), threads=1
            ),
            expected,
        )
        self.assertTrue(np.all(expected >= 1))

        high_resolution = 29
        high_cells = np.asarray([12 * 4**29 - 1, 0, 6 * 4**29, 0], dtype=np.uint64)
        np.testing.assert_array_equal(
            px.cover_cap(
                [-1.0, 0.0, 0.0],
                math.pi,
                resolution=high_resolution,
                candidate_cells=high_cells,
                reduce=px.Count(),
            ),
            np.ones(high_cells.size, dtype=np.int64),
        )
        queried_empty = px.cover_cap(
            [1.0, 0.0, 0.0],
            0.1,
            resolution=high_resolution,
            candidate_cells=[],
            reduce=px.Count(),
        )
        self.assertEqual(queried_empty.shape, (0,))
        self.assertEqual(queried_empty.dtype, np.dtype("int64"))
        with self.assertRaisesRegex(MemoryError, "too large"):
            px.cover_cap(
                [1.0, 0.0, 0.0], 0.1, resolution=high_resolution, reduce=px.Count()
            )
        with self.assertRaisesRegex(MemoryError, "too large"):
            px.cover_cap([1.0, 0.0, 0.0], math.pi, high_resolution)

    def test_cap_inputs_are_validated(self) -> None:
        center = lonlat_to_vec(0.0, 0.0)
        for invalid in (-1.0, math.nextafter(math.pi, math.inf), np.nan, np.inf):
            with self.subTest(radius=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    px.cover_cap(center, invalid, resolution=3)

        with self.assertRaisesRegex(ValueError, "centers_xyz"):
            px.cover_cap(np.zeros(3), 0.1, resolution=3)
        with self.assertRaisesRegex(ValueError, "one radius per center"):
            px.cover_cap(np.stack((center, center)), [0.1], resolution=3)
        with self.assertRaisesRegex(ValueError, "shape"):
            px.cover_cap(np.zeros((2, 2)), 0.1, resolution=3)
        with self.assertRaisesRegex(ValueError, "valid RING indices"):
            px.cover_cap(
                center,
                0.1,
                resolution=3,
                candidate_cells=[12 * 4**3],
                reduce=px.Count(),
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

        coverage = px.cover_convex_polygon(ragged, resolution=2)
        expected = [
            px.cover_convex_polygon(polygon, resolution=2).cells for polygon in ragged
        ]

        self.assertSegmentsEqual(coverage, expected)
        np.testing.assert_array_equal(
            coverage.segment_sizes,
            np.asarray([cells.size for cells in expected], dtype=np.intp),
        )

        dense = np.stack((ragged[0], vectors(polygons[0]) * 7.0))
        dense_coverage = px.cover_convex_polygon(dense, resolution=2)
        self.assertSegmentsEqual(dense_coverage, [expected[0], expected[0]])

        ragged_quads = px.cover_convex_polygon(list(dense), resolution=2)
        np.testing.assert_array_equal(ragged_quads.cells, dense_coverage.cells)
        np.testing.assert_array_equal(ragged_quads.offsets, dense_coverage.offsets)

        nested_quads = px.cover_convex_polygon(dense.tolist(), resolution=2)
        np.testing.assert_array_equal(nested_quads.cells, dense_coverage.cells)
        np.testing.assert_array_equal(nested_quads.offsets, dense_coverage.offsets)

    def test_cover_accepts_array_protocol_inputs(self) -> None:
        polygon = vectors([(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)])
        for values in (polygon, np.stack((polygon, polygon))):
            with self.subTest(ndim=values.ndim):
                expected = px.cover_convex_polygon(values, resolution=3)
                actual = px.cover_convex_polygon(ArrayOnly(values), resolution=3)
                np.testing.assert_array_equal(actual.cells, expected.cells)
                np.testing.assert_array_equal(actual.offsets, expected.offsets)

    def test_repeated_closing_vertex_is_representation_independent(self) -> None:
        triangle = vectors([(-5.0, -5.0), (5.0, -5.0), (0.0, 5.0)])
        closed = np.vstack((triangle, triangle[0]))

        open_coverage = px.cover_convex_polygon(triangle, resolution=3, threads=1)
        dense_closed = px.cover_convex_polygon(closed, resolution=3, threads=1)
        ragged_closed = px.cover_convex_polygon(
            [closed, triangle], resolution=3, threads=1
        )

        np.testing.assert_array_equal(dense_closed.cells, open_coverage.cells)
        self.assertSegmentsEqual(
            ragged_closed,
            [open_coverage.cells, open_coverage.cells],
        )

    def test_redundant_collinear_vertices_are_accepted_at_small_scales(self) -> None:
        for half_size_degrees in (0.35, 0.05, 0.005):
            quad = vectors(
                [
                    (-half_size_degrees, -half_size_degrees),
                    (half_size_degrees, -half_size_degrees),
                    (half_size_degrees, half_size_degrees),
                    (-half_size_degrees, half_size_degrees),
                ]
            )
            midpoint = quad[1] + quad[2]
            midpoint /= np.linalg.norm(midpoint)
            densified = np.vstack((quad[:2], midpoint, quad[2:]))

            with self.subTest(half_size_degrees=half_size_degrees):
                expected = px.cover_convex_polygon(quad, resolution=12, threads=1)
                actual = px.cover_convex_polygon(densified, resolution=12, threads=1)
                np.testing.assert_array_equal(actual.cells, expected.cells)

    def test_cover_sweep_covers_consecutive_edge_intervals(self) -> None:
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

        expected = px.cover_convex_polygon(footprints, resolution=3)
        actual = px.cover_sweep(left, right, resolution=3)

        np.testing.assert_array_equal(actual.offsets, expected.offsets)
        np.testing.assert_array_equal(actual.cells, expected.cells)

    def test_cover_sweep_accepts_a_pinch_on_either_edge(self) -> None:
        pivot = lonlat_to_vec(0.0, 0.0)
        left = np.asarray([pivot, pivot])
        right = np.asarray([lonlat_to_vec(-8.0, 6.0), lonlat_to_vec(8.0, 6.0)])

        left_pinched = px.cover_sweep(left, right, resolution=3)
        right_pinched = px.cover_sweep(right, left, resolution=3)

        np.testing.assert_array_equal(right_pinched.cells, left_pinched.cells)
        np.testing.assert_array_equal(right_pinched.offsets, left_pinched.offsets)

    def test_cover_sweep_uses_minor_arcs_between_samples(self) -> None:
        def edges(step: float) -> tuple[np.ndarray, np.ndarray]:
            return (
                np.asarray([lonlat_to_vec(0.0, -5.0), lonlat_to_vec(step, -5.0)]),
                np.asarray([lonlat_to_vec(0.0, 5.0), lonlat_to_vec(step, 5.0)]),
            )

        # Both counts are the mirror image of each other, because both quads
        # are the same shape reflected across the prime meridian. The minor-arc
        # case used to report 352: its span ended exactly on the meridian, and
        # the longitude bounds then dropped the two cells sitting on it.
        for step, expected_count, expected_y_sign in (
            (179.0, 354, 1.0),
            (181.0, 354, -1.0),
        ):
            with self.subTest(step=step):
                left, right = edges(step)
                actual = px.cover_sweep(left, right, resolution=3, threads=1)
                quad = np.asarray([left[0], right[0], right[1], left[1]])
                expected = px.cover_convex_polygon(quad, resolution=3, threads=1)
                np.testing.assert_array_equal(actual.cells, expected.cells)
                self.assertEqual(actual.cells.size, expected_count)
                centers = px.cell_centers(actual.cells, resolution=3)
                self.assertGreater(expected_y_sign * float(np.mean(centers[:, 1])), 0.0)

        left, right = edges(180.0)
        with self.assertRaises(ValueError):
            px.cover_sweep(left, right, resolution=3, threads=1)

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
                    px.cover_sweep(invalid_left, invalid_right, resolution=3)

    def test_candidate_coverage_matches_post_filtering(self) -> None:
        footprints = np.asarray(
            [
                vectors([(-15.0, -8.0), (8.0, -8.0), (8.0, 8.0), (-15.0, 8.0)]),
                vectors([(20.0, -8.0), (43.0, -8.0), (43.0, 8.0), (20.0, 8.0)]),
            ]
        )
        unfiltered = px.cover_convex_polygon(footprints, resolution=4)
        selected = unfiltered.cells[::2]
        candidates = np.concatenate(
            (selected[::-1], selected[:2], np.asarray([0], dtype=np.int64))
        )

        actual = px.cover_convex_polygon(
            footprints,
            resolution=4,
            candidate_cells=candidates,
        )

        self.assertSegmentsEqual(
            actual,
            post_filter_coverage(unfiltered, candidates),
        )

    def test_empty_candidates_preserve_segment_offsets(self) -> None:
        polygon = vectors([(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)])
        coverage = px.cover_convex_polygon(
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
                    px.cover_convex_polygon(
                        polygon,
                        resolution=4,
                        candidate_cells=candidates,
                    )

        with self.assertRaisesRegex(ValueError, "scalar or one-dimensional"):
            px.cover_convex_polygon(
                polygon,
                resolution=4,
                candidate_cells=np.empty((1, 0), dtype=np.uint64),
            )
        with self.assertRaisesRegex(ValueError, "valid RING indices"):
            px.cover_convex_polygon(
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
                unfiltered = px.cover_convex_polygon(footprint, resolution=4)
                candidates = unfiltered.cells[::2][::-1]
                actual = px.cover_convex_polygon(
                    footprint,
                    resolution=4,
                    candidate_cells=candidates,
                )
                self.assertCellsEqual(actual.cells, candidates)

    def test_resolution_29_scan_keeps_a_center_on_the_longitude_bound(self) -> None:
        resolution = 29
        cell = np.uint64(6 * 4**resolution)
        center = px.cell_centers(cell, resolution)[0]
        tangent = np.cross(np.asarray([0.0, 0.0, 1.0]), center)
        tangent /= np.linalg.norm(tangent)
        inward = np.cross(center, tangent)
        scale = 8.0e-9

        def offset_point(x: float, y: float) -> np.ndarray:
            point = center + scale * (x * tangent + y * inward)
            return point / np.linalg.norm(point)

        footprint = np.asarray(
            [
                offset_point(-1.0, 0.0),
                offset_point(1.0, 0.0),
                offset_point(1.0, 2.0),
                offset_point(-1.0, 2.0),
            ]
        )
        actual = px.cover_convex_polygon(footprint, resolution, threads=1)
        self.assertIn(cell, actual.cells)

    def test_large_cap_vertices_select_the_minor_arc_antipodal_cap(self) -> None:
        north_91 = regular_spherical_polygon(
            np.asarray([0.0, 0.0, 1.0]),
            math.radians(91.0),
            12,
        )
        south_89 = regular_spherical_polygon(
            np.asarray([0.0, 0.0, -1.0]),
            math.radians(89.0),
            12,
        )
        actual = px.cover_convex_polygon(north_91, resolution=3, threads=1)
        expected = px.cover_convex_polygon(south_89, resolution=3, threads=1)
        np.testing.assert_array_equal(actual.cells, expected.cells)

    def test_centimetre_scale_footprint_is_valid_at_resolution_29(self) -> None:
        polygon = regular_spherical_polygon(
            np.asarray([1.0, 0.0, 0.0]),
            1.0e-8,
            4,
        )
        coverage = px.cover_convex_polygon(
            polygon,
            resolution=29,
            candidate_cells=np.empty(0, dtype=np.uint64),
            threads=1,
        )
        np.testing.assert_array_equal(coverage.cells, [])
        np.testing.assert_array_equal(coverage.offsets, [0, 0])

    def test_footprint_below_the_documented_validation_floor_fails_safely(
        self,
    ) -> None:
        polygon = regular_spherical_polygon(
            np.asarray([1.0, 0.0, 0.0]),
            5.0e-9,
            4,
        )
        with self.assertRaisesRegex(ValueError, "degenerate"):
            px.cover_convex_polygon(
                polygon,
                resolution=29,
                candidate_cells=np.empty(0, dtype=np.uint64),
                threads=1,
            )

    def test_small_high_latitude_quad_is_not_rejected_by_endpoint_roundoff(
        self,
    ) -> None:
        longitude = -159.20634920634922
        latitude = -60.0
        polygon = vectors(
            [
                (longitude - 0.05, latitude - 0.05),
                (longitude + 0.05, latitude - 0.05),
                (longitude + 0.05, latitude + 0.05),
                (longitude - 0.05, latitude + 0.05),
            ]
        )
        coverage = px.cover_convex_polygon(
            polygon,
            resolution=29,
            candidate_cells=[],
            threads=1,
        )
        np.testing.assert_array_equal(coverage.offsets, [0, 0])

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

        actual = px.cover_convex_polygon(vectors(polygon), resolution=5, threads=1)
        expected = brute_force_cover(polygon, resolution=5)

        self.assertGreater(expected.size, 0)
        self.assertCellsEqual(actual.cells, expected)

    def test_strip_thread_counts_produce_identical_ordered_results(self) -> None:
        latitudes = np.linspace(-50.0, 50.0, 2049)
        left = np.asarray([lonlat_to_vec(-4.0, value) for value in latitudes])
        right = np.asarray([lonlat_to_vec(4.0, value) for value in latitudes])

        single_threaded = px.cover_sweep(left, right, resolution=5, threads=1)
        parallel = px.cover_sweep(left, right, resolution=5, threads=4)
        automatic = px.cover_sweep(left, right, resolution=5)
        for actual in (parallel, automatic):
            np.testing.assert_array_equal(actual.cells, single_threaded.cells)
            np.testing.assert_array_equal(actual.offsets, single_threaded.offsets)

    def test_strip_candidate_thread_counts_are_deterministic(self) -> None:
        latitudes = np.linspace(-50.0, 50.0, 2049)
        left = np.asarray([lonlat_to_vec(-4.0, value) for value in latitudes])
        right = np.asarray([lonlat_to_vec(4.0, value) for value in latitudes])
        full = px.cover_sweep(left, right, resolution=5, threads=1)
        candidates = np.unique(full.cells[::2])

        single_threaded = px.cover_sweep(
            left,
            right,
            resolution=5,
            candidate_cells=candidates,
            threads=1,
        )
        for threads in (4, None):
            with self.subTest(threads=threads):
                actual = px.cover_sweep(
                    left,
                    right,
                    resolution=5,
                    candidate_cells=candidates,
                    threads=threads,
                )
                np.testing.assert_array_equal(actual.cells, single_threaded.cells)
                np.testing.assert_array_equal(actual.offsets, single_threaded.offsets)

    def test_thread_count_must_be_a_positive_integer(self) -> None:
        polygon = vectors([(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)])
        for threads in (0, -1, True):
            with self.subTest(threads=threads):
                with self.assertRaises((TypeError, ValueError)):
                    px.cover_convex_polygon(polygon, resolution=2, threads=threads)

        sequential = px.cover_convex_polygon(polygon, resolution=2, threads=1)
        bounded = px.cover_convex_polygon(polygon, resolution=2, threads=100_000)
        np.testing.assert_array_equal(bounded.cells, sequential.cells)
        np.testing.assert_array_equal(bounded.offsets, sequential.offsets)

    def test_concurrent_calls_are_deterministic(self) -> None:
        polygon = vectors([(-5.0, -5.0), (12.0, -4.0), (10.0, 9.0), (-6.0, 7.0)])
        footprints = np.repeat(polygon[np.newaxis, :, :], 2048, axis=0)
        expected = px.cover_convex_polygon(
            footprints,
            resolution=3,
            threads=2,
        )

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(
                executor.map(
                    lambda _: px.cover_convex_polygon(
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

    def test_centers_and_corners_return_unit_xyz(self) -> None:
        cells = np.asarray([0, 17, 123], dtype=np.uint64)

        center_vectors = px.cell_centers(cells, resolution=3)
        boundary_vectors = px.cell_corners(cells, resolution=3)
        scalar_center = px.cell_centers(int(cells[0]), resolution=3)
        scalar_boundary = px.cell_corners(int(cells[0]), resolution=3)

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

    def test_cell_at_accepts_scalar_batches_and_strided_inputs(self) -> None:
        cells = np.asarray([0, 7, 31, 191], dtype=np.uint64)
        vectors = px.cell_centers(cells, resolution=2)
        padded = np.zeros((vectors.shape[0], 6), dtype=np.float64)
        padded[:, ::2] = vectors

        actual = px.cell_at(padded[:, ::2], resolution=2)
        self.assertEqual(actual.dtype, np.dtype("int64"))
        self.assertEqual(actual.shape, cells.shape)
        np.testing.assert_array_equal(actual, cells)
        np.testing.assert_array_equal(px.cell_at(vectors[0] * 1e200, 2), cells[:1])
        self.assertEqual(px.cell_at(np.empty((0, 3)), 2).shape, (0,))

        for invalid in ([0.0, 0.0, 0.0], [np.nan, 0.0, 1.0]):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "vectors_xyz"):
                    px.cell_at(invalid, resolution=2)
        with self.assertRaisesRegex(ValueError, "shape"):
            px.cell_at(np.empty((3, 1)), resolution=2)
        with self.assertRaises(TypeError):
            px.cell_at([True, False, True], resolution=2)

    def test_empty_centers_and_corners_have_stable_shapes(self) -> None:
        cells = np.empty(0, dtype=np.uint64)
        self.assertEqual(px.cell_centers(cells, resolution=3).shape, (0, 3))
        self.assertEqual(px.cell_corners(cells, resolution=3).shape, (0, 4, 3))

    def test_resolution_requires_an_integer_in_range(self) -> None:
        footprint = vectors([(-5.0, -5.0), (12.0, -4.0), (10.0, 9.0), (-6.0, 7.0)])
        for resolution in (2.0, "2", True):
            with self.subTest(resolution=resolution):
                with self.assertRaises(TypeError):
                    px.cover_convex_polygon(footprint, resolution=resolution)
        for resolution in (-1, 30):
            with self.subTest(resolution=resolution):
                with self.assertRaisesRegex(ValueError, "between 0 and 29"):
                    px.cover_convex_polygon(footprint, resolution=resolution)

        self.assertIsInstance(
            px.cover_convex_polygon(footprint, resolution=np.int64(2)),
            px.Coverage,
        )

    def test_cell_indices_require_valid_integers(self) -> None:
        for cells in (256.0, [256.9], np.asarray([256.9]), True, [True]):
            with self.subTest(cells=cells):
                with self.assertRaises(TypeError):
                    px.cell_centers(cells, resolution=3)
        for cells in (-1, [-1], np.asarray([-1], dtype=np.int64)):
            with self.subTest(cells=cells):
                with self.assertRaises(ValueError):
                    px.cell_corners(cells, resolution=3)
        with self.assertRaisesRegex(ValueError, "valid RING indices"):
            px.cell_centers([12 * 4**3], resolution=3)
        with self.assertRaises(OverflowError):
            px.cell_centers(np.asarray([2**64], dtype=object), resolution=3)

    def test_imported_indices_are_range_checked_per_input_dtype(self) -> None:
        """Public results skip a redundant pass; other inputs still get one.

        A signed input that holds no negative value is already inside int64, so
        importing a Polypix result costs one copy. Unsigned and object inputs
        can exceed int64 and must still be rejected rather than wrapping to a
        negative public index.
        """
        for name, values in (
            ("cells", (np.asarray([2**63], dtype=np.uint64), [0, 1])),
            ("offsets", (np.asarray([0]), np.asarray([0, 2**63], dtype=np.uint64))),
            ("cells", ([2**63], [0, 1])),
            ("cells", (np.asarray([2**63], dtype=object), [0, 1])),
        ):
            cells, offsets = values
            with self.subTest(name=name, dtype=np.asarray(cells).dtype):
                with self.assertRaisesRegex(OverflowError, "out of range for int64"):
                    px.Coverage.from_arrays(cells, offsets, resolution=0)

        for cells in (
            np.asarray([-1], dtype=np.int64),
            np.asarray([-1], dtype=np.int32),
            [-1],
        ):
            with self.subTest(dtype=np.asarray(cells).dtype):
                with self.assertRaisesRegex(ValueError, "non-negative integers"):
                    px.Coverage.from_arrays(cells, [0, 1], resolution=0)

        # Empty arrays reach neither reduction, whatever their dtype.
        for dtype in (np.int64, np.uint64):
            with self.subTest(dtype=dtype):
                empty = px.Coverage.from_arrays(
                    np.asarray([], dtype=dtype),
                    np.asarray([0], dtype=dtype),
                    resolution=0,
                )
                self.assertEqual(empty.cells.size, 0)
                self.assertEqual(empty.cells.dtype, np.int64)

    def test_negative_indices_are_named_wherever_the_kernel_checks_range(
        self,
    ) -> None:
        """Deferring the scan must not degrade the message it used to produce.

        Every entry point that hands a cell array to a native range check skips
        the separate non-negative scan, so a negative index arrives as a `u64`
        above every cell count. The kernel has to distinguish that from an index
        that is merely too large, at each public argument name.
        """
        coverage = px.cover_cap(np.asarray([[1.0, 0.0, 0.0]]), 0.3, 4)
        negative = np.asarray([-1], dtype=np.int64)
        cases = (
            ("cells", lambda: px.cell_centers(negative, resolution=7)),
            ("cells", lambda: px.cell_corners([-1], resolution=3)),
            ("cells", lambda: px.Coverage.from_arrays(negative, [0, 1], 0)),
            ("offsets", lambda: px.Coverage.from_arrays([0], [0, -1], 0)),
            ("cells", lambda: coverage.reduce(px.Count(), cells=negative)),
            ("cells", lambda: coverage.reduce(px.Sum(1.0), cells=negative)),
            (
                "cells",
                lambda: px.cover_cap(
                    np.asarray([[1.0, 0.0, 0.0]]),
                    0.1,
                    4,
                    candidate_cells=negative,
                    reduce=px.Count(),
                ),
            ),
            (
                "candidate_cells",
                lambda: px.cover_cap(
                    np.asarray([[1.0, 0.0, 0.0]]),
                    0.1,
                    4,
                    candidate_cells=negative,
                ),
            ),
            (
                "vertex_offsets",
                lambda: px.cover_convex_polygon(
                    np.zeros((4, 3)), 4, vertex_offsets=[0, -1]
                ),
            ),
        )
        for name, call in cases:
            with self.subTest(argument=name):
                with self.assertRaisesRegex(
                    ValueError, f"{name} must contain non-negative integers"
                ):
                    call()

        # An index that is too large but not negative keeps its own message.
        for call in (
            lambda: px.cell_centers([12 * 4**3], resolution=3),
            lambda: px.Coverage.from_arrays([12], [0, 1], 0),
            lambda: coverage.reduce(px.Count(), cells=[12 * 4**4]),
        ):
            with self.subTest(kind="out of range"):
                with self.assertRaisesRegex(ValueError, "valid RING indices"):
                    call()

    def test_misaligned_contiguous_inputs_are_accepted(self) -> None:
        """A contiguous array can still sit on an unaligned address.

        Viewing one out of a packed byte buffer produces exactly that, and
        neither ``asarray`` nor ``ascontiguousarray`` repairs it, because dtype
        and contiguity already match. Native code cannot borrow such an array
        as a slice, so the conversion helpers have to force the copy.
        """

        def misaligned(dtype: object, count: int, shape: object = None) -> np.ndarray:
            itemsize = np.dtype(dtype).itemsize
            buffer = np.zeros(itemsize * count + 1, dtype=np.uint8)
            array = np.frombuffer(buffer.data, dtype=dtype, offset=1, count=count)
            return array.reshape(shape) if shape is not None else array

        polygon = misaligned(np.float64, 12, (4, 3))
        polygon[:] = vectors([(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)])
        self.assertFalse(polygon.flags.aligned)
        self.assertTrue(polygon.flags.c_contiguous)

        coverage = px.cover_convex_polygon(polygon, resolution=5)
        self.assertEqual(coverage.cells.size, coverage.offsets[-1])
        self.assertGreater(coverage.cells.size, 0)

        center = misaligned(np.float64, 3, (1, 3))
        center[:] = vectors([(0.0, 0.0)])
        radii = misaligned(np.float64, 1)
        radii[:] = 0.2
        capped = px.cover_cap(center, radii, 5)
        self.assertGreater(capped.cells.size, 0)
        np.testing.assert_array_equal(
            px.cover_cap(center, radii, 5, reduce=px.Count()),
            capped.reduce(px.Count()),
        )

        cells = misaligned(np.int64, 2)
        cells[:] = [1, 2]
        offsets = misaligned(np.int64, 2)
        offsets[:] = [0, 2]
        imported = px.Coverage.from_arrays(cells, offsets, resolution=2)
        np.testing.assert_array_equal(imported.cells, [1, 2])
        np.testing.assert_array_equal(px.cell_centers(cells, 2).shape, (2, 3))
        np.testing.assert_array_equal(px.cell_corners(cells, 2).shape, (2, 4, 3))
        np.testing.assert_array_equal(px.cell_at(px.cell_centers(cells, 2), 2), [1, 2])

        values = misaligned(np.float64, capped.segment_count)
        values[:] = 1.5
        self.assertAlmostEqual(
            float(capped.reduce(px.Sum(values)).sum()), 1.5 * capped.cells.size
        )

    def test_cover_rejects_invalid_array_shape(self) -> None:
        polygon = vectors([(-5.0, -5.0), (12.0, -4.0), (10.0, 9.0), (-6.0, 7.0)])
        for invalid in (polygon[:, :2], polygon[:, :2].tolist()):
            with self.subTest(input_type=type(invalid).__name__):
                with self.assertRaisesRegex(
                    ValueError, r"^polygons_xyz must have shape"
                ):
                    px.cover_convex_polygon(invalid, resolution=2)

    def test_cover_normalizes_arbitrary_vectors(self) -> None:
        footprint = vectors([(-5.0, -5.0), (12.0, -4.0), (10.0, 9.0), (-6.0, 7.0)])
        scales = np.asarray([2.0, 1e300, 1e-300, 7.0])[:, np.newaxis]

        expected = px.cover_convex_polygon(footprint, resolution=3)
        actual = px.cover_convex_polygon(footprint * scales, resolution=3)

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
                    px.cover_convex_polygon(footprint, resolution=1)
        with self.assertRaisesRegex(TypeError, "complex"):
            px.cover_convex_polygon(valid.astype(np.complex128), resolution=1)

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
            "antipodal_edge": [(0.0, 0.0), (180.0, 0.0), (0.0, 20.0)],
            "self_intersecting": [
                (-10.0, -10.0),
                (10.0, 10.0),
                (10.0, -10.0),
                (-10.0, 10.0),
            ],
            "exact_hemisphere": [(0.0, 0.0), (120.0, 0.0), (-120.0, 0.0)],
            "hemisphere_rectangle": [
                (-90.0, -20.0),
                (90.0, -20.0),
                (90.0, 20.0),
                (-90.0, 20.0),
            ],
        }

        for name, polygon in invalid_polygons.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    px.cover_convex_polygon(vectors(polygon), resolution=1)

    def test_batch_geometry_errors_name_the_offending_polygon(self) -> None:
        valid = vectors([(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)])
        invalid = valid.copy()
        invalid[2] = invalid[1]
        batch = np.repeat(valid[np.newaxis, :, :], 4096, axis=0)
        batch[3000] = invalid

        for threads in (1, 4, None):
            with self.subTest(threads=threads):
                with self.assertRaisesRegex(
                    ValueError, r"polygons_xyz\[3000\]: Polygon"
                ):
                    px.cover_convex_polygon(batch, resolution=3, threads=threads)

    def test_parallel_batch_reports_the_first_invalid_polygon(self) -> None:
        valid = vectors([(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)])
        invalid = valid.copy()
        invalid[2] = invalid[1]
        batch = np.repeat(valid[np.newaxis, :, :], 4096, axis=0)
        batch[10] = invalid
        batch[3000] = invalid

        for threads in (4, None):
            with self.subTest(threads=threads):
                with self.assertRaisesRegex(ValueError, r"polygons_xyz\[10\]"):
                    px.cover_convex_polygon(batch, resolution=3, threads=threads)

    def test_cover_accepts_empty_batches(self) -> None:
        for shape in ((0, 4, 3), (0, 0, 3)):
            with self.subTest(shape=shape):
                coverage = px.cover_convex_polygon(
                    np.empty(shape, dtype=np.float64),
                    resolution=1,
                )
                self.assertEqual(coverage.cells.dtype, np.dtype("int64"))
                self.assertEqual(coverage.offsets.dtype, np.dtype("int64"))
                self.assertEqual(coverage.cells.shape, (0,))
                np.testing.assert_array_equal(coverage.offsets, [0])
                self.assertEqual(coverage.segment_sizes.shape, (0,))

        list_coverage = px.cover_convex_polygon([], resolution=1)
        np.testing.assert_array_equal(list_coverage.cells, [])
        np.testing.assert_array_equal(list_coverage.offsets, [0])

    def test_coverage_does_not_claim_array_value_equality(self) -> None:
        first = px.cover_convex_polygon([], resolution=1)
        second = px.cover_convex_polygon([], resolution=1)
        self.assertIsNot(first, second)
        self.assertFalse(first == second)

    def test_coverage_is_a_validated_read_only_segmented_array(self) -> None:
        imported_cells = np.asarray([1, 4, 7], dtype=np.uint64)
        imported_offsets = np.asarray([0, 2, 3], dtype=np.uint64)
        coverage = px.Coverage.from_arrays(
            imported_cells,
            imported_offsets,
            resolution=1,
        )

        imported_cells[0] = 9
        imported_offsets[1] = 1
        np.testing.assert_array_equal(coverage.cells, [1, 4, 7])
        np.testing.assert_array_equal(coverage.offsets, [0, 2, 3])
        self.assertFalse(coverage.cells.flags.writeable)
        self.assertFalse(coverage.offsets.flags.writeable)
        self.assertEqual(len(coverage), 2)
        self.assertEqual(coverage.segment_count, 2)
        np.testing.assert_array_equal(coverage[0], [1, 4])
        np.testing.assert_array_equal(coverage[-1], [7])
        self.assertTrue(np.shares_memory(coverage[0], coverage.cells))
        with self.assertRaises(IndexError):
            _ = coverage[2]

        unsorted = px.Coverage.from_arrays([4, 1, 7], [0, 2, 3], resolution=1)
        np.testing.assert_array_equal(unsorted[0], [4, 1])

        invalid = (
            ([1, 1], [0, 2], "unique"),
            ([1], [1, 1], "start at zero"),
            ([1], [0, 2], "offsets"),
            ([48], [0, 1], "valid RING indices"),
        )
        for cells, offsets, message in invalid:
            with self.subTest(cells=cells, offsets=offsets):
                with self.assertRaisesRegex(ValueError, message):
                    px.Coverage.from_arrays(cells, offsets, resolution=1)

        with self.assertRaisesRegex(TypeError, "not bool"):
            _ = coverage[True]
        with self.assertRaisesRegex(TypeError, "from_arrays"):
            px.Coverage([1], [0, 1], resolution=1)  # type: ignore[call-arg]

        native = px.cover_convex_polygon([], resolution=1)
        self.assertFalse(native.cells.flags.writeable)
        self.assertFalse(native.offsets.flags.writeable)

    def test_cover_rejects_nonempty_zero_vertex_batch(self) -> None:
        for shape in ((0, 3), (1, 0, 3)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    px.cover_convex_polygon(
                        np.empty(shape, dtype=np.float64),
                        resolution=1,
                    )

    def test_only_target_public_endpoints_are_exposed(self) -> None:
        self.assertEqual(
            px.__all__,
            [
                "CellsLike",
                "Count",
                "Coverage",
                "CoverageReducer",
                "EdgesLike",
                "OccupancyRuns",
                "OccupancyStats",
                "OffsetsLike",
                "PolygonsLike",
                "Stats",
                "Sum",
                "ValuesLike",
                "VectorsLike",
                "__version__",
                "cell_at",
                "cell_centers",
                "cell_corners",
                "cell_count",
                "cover_cap",
                "cover_convex_polygon",
                "cover_sweep",
                "occupancy",
            ],
        )
        for name in [
            "OccupancySummary",
            "cell_area_from_resolution",
            "cell_boundary",
            "cell_center",
            "centers",
            "children",
            "corners",
            "cover",
            "cover_footprint",
            "decode_cell_id",
            "encode_cell_id",
            "parent",
            "cover_swath",
            "summarize_occupancy",
        ]:
            self.assertFalse(hasattr(px, name), name)

        coverage = px.Coverage.from_arrays([], [0], resolution=0)
        self.assertFalse(hasattr(coverage, "counts"))


if __name__ == "__main__":
    unittest.main()
