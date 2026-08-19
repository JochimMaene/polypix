import math
import sys
from pathlib import Path

import numpy as np

import polypix as px

# Wheel tests run from outside the checkout. These tests intentionally exercise
# the repository's example modules, which are not part of the installed wheel.
_REPOSITORY_ROOT = str(Path(__file__).resolve().parents[1])
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

from examples.constellation import (  # noqa: E402
    EARTH_RADIUS_KM,
    constellation_centers,
    service_caps,
    swath_edges,
)
from examples.earth_observation_constellation import (  # noqa: E402
    SWATH_HALF_WIDTH_RAD,
    reduce_coverage,
)


def test_swath_edges_are_normalized_at_the_configured_half_width() -> None:
    times_s = np.arange(0.0, 181.0, 60.0)
    centers = constellation_centers(
        times_s,
        satellite_count=10,
        plane_count=5,
        altitude_km=550.0,
        inclination_rad=math.radians(53.0),
    )

    left, right = swath_edges(centers, half_width_rad=SWATH_HALF_WIDTH_RAD)

    assert left.shape == centers.shape
    assert right.shape == centers.shape
    np.testing.assert_allclose(np.linalg.norm(left, axis=-1), 1.0, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(right, axis=-1), 1.0, atol=1e-14)
    np.testing.assert_allclose(
        np.sum(left * centers, axis=-1),
        math.cos(SWATH_HALF_WIDTH_RAD),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        np.sum(right * centers, axis=-1),
        math.cos(SWATH_HALF_WIDTH_RAD),
        atol=1e-14,
    )


def test_service_caps_normalize_positions_and_use_altitude_dependent_radii() -> None:
    minimum_elevation_rad = math.radians(25.0)
    positions_km = np.asarray(
        [
            [EARTH_RADIUS_KM + 350.0, 0.0, 0.0],
            [0.0, 0.0, EARTH_RADIUS_KM + 1200.0],
        ]
    )
    orbit_radii_km = np.linalg.norm(positions_km, axis=1)

    centers, radii = service_caps(
        positions_km,
        body_radius_km=EARTH_RADIUS_KM,
        minimum_elevation_rad=minimum_elevation_rad,
    )

    np.testing.assert_allclose(np.linalg.norm(centers, axis=1), 1.0)
    np.testing.assert_allclose(
        radii,
        np.arccos(EARTH_RADIUS_KM / orbit_radii_km * math.cos(minimum_elevation_rad))
        - minimum_elevation_rad,
    )
    assert np.all((radii >= 0.0) & (radii <= math.pi))


def test_constellation_rejects_uneven_plane_distribution() -> None:
    with np.testing.assert_raises_regex(ValueError, "divisible"):
        constellation_centers(
            np.asarray([0.0]),
            satellite_count=10,
            plane_count=3,
            altitude_km=550.0,
            inclination_rad=math.radians(53.0),
        )


def test_reduce_coverage_counts_windows_and_mean_uncovered_gaps() -> None:
    first = px.Coverage.from_arrays(
        cells=np.asarray([1, 2, 1, 1, 2], dtype=np.uint64),
        offsets=np.asarray([0, 2, 3, 3, 5], dtype=np.uint64),
        resolution=1,
    )
    second = px.Coverage.from_arrays(
        cells=np.asarray([1, 2, 2], dtype=np.uint64),
        offsets=np.asarray([0, 1, 1, 2, 3], dtype=np.uint64),
        resolution=1,
    )

    observations, mean_internal_gap_s, max_internal_gap_s, gap_counts = reduce_coverage(
        [first, second],
        cell_count=48,
        cadence_s=60,
    )

    assert observations[1] == 2
    assert observations[2] == 2
    assert mean_internal_gap_s[1] == 60
    assert mean_internal_gap_s[2] == 60
    assert max_internal_gap_s[1] == 60
    assert max_internal_gap_s[2] == 60
    assert gap_counts[1] == 1
    assert gap_counts[2] == 1
    assert np.isnan(mean_internal_gap_s[3])
    assert np.isnan(max_internal_gap_s[3])


def test_reduce_coverage_excludes_horizon_edge_gaps() -> None:
    coverage = px.Coverage.from_arrays(
        cells=[4, 4],
        offsets=[0, 0, 1, 1, 2, 2],
        resolution=1,
    )

    observations, mean_internal_gap_s, max_internal_gap_s, gap_counts = reduce_coverage(
        [coverage], cell_count=48, cadence_s=60
    )

    assert observations[4] == 2
    assert gap_counts[4] == 1
    assert mean_internal_gap_s[4] == 60
    assert max_internal_gap_s[4] == 60
    # The unobserved first and last bins are deliberately not part of this
    # complete internal-gap statistic.


def test_reduce_coverage_rejects_mismatched_interval_counts() -> None:
    first = px.Coverage.from_arrays(
        cells=np.asarray([1], dtype=np.uint64),
        offsets=np.asarray([0, 1], dtype=np.uint64),
        resolution=1,
    )
    second = px.Coverage.from_arrays(
        cells=np.asarray([1], dtype=np.uint64),
        offsets=np.asarray([0, 1, 1], dtype=np.uint64),
        resolution=1,
    )

    with np.testing.assert_raises_regex(ValueError, "same interval count"):
        reduce_coverage([first, second], cell_count=48, cadence_s=60)
