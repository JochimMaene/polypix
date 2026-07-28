import math

import numpy as np

import polypix as px
from examples.communication_constellation import service_radius_rad
from examples.constellation import cap_footprints, constellation_centers, swath_edges
from examples.earth_observation_constellation import (
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


def test_communications_footprints_use_the_service_radius() -> None:
    centers = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    radius = service_radius_rad()

    footprints = cap_footprints(centers, radius_rad=radius, vertex_count=16)

    assert 0.0 < radius < math.pi / 2
    assert footprints.shape == (2, 16, 3)
    np.testing.assert_allclose(np.linalg.norm(footprints, axis=-1), 1.0)
    np.testing.assert_allclose(
        np.sum(footprints * centers[:, np.newaxis, :], axis=-1),
        math.cos(radius),
    )


def test_constellation_rejects_uneven_plane_distribution() -> None:
    with np.testing.assert_raises_regex(ValueError, "divisible"):
        constellation_centers(
            np.asarray([0.0]),
            satellite_count=10,
            plane_count=3,
            altitude_km=550.0,
            inclination_rad=math.radians(53.0),
        )


def test_reduce_coverage_counts_observations_and_mean_revisit_gaps() -> None:
    first = px.Coverage(
        cells=np.asarray([1, 2, 1, 1, 2], dtype=np.uint64),
        offsets=np.asarray([0, 2, 3, 3, 5], dtype=np.uint64),
        resolution=1,
    )
    second = px.Coverage(
        cells=np.asarray([1, 2, 2], dtype=np.uint64),
        offsets=np.asarray([0, 1, 1, 2, 3], dtype=np.uint64),
        resolution=1,
    )

    observations, mean_revisit_s, revisit_counts = reduce_coverage(
        [first, second],
        cell_count=48,
        cadence_s=60,
    )

    assert observations[1] == 3
    assert observations[2] == 3
    assert mean_revisit_s[1] == 60
    assert mean_revisit_s[2] == 60
    assert revisit_counts[1] == 1
    assert revisit_counts[2] == 1
    assert np.isnan(mean_revisit_s[3])


def test_reduce_coverage_rejects_mismatched_interval_counts() -> None:
    first = px.Coverage(
        cells=np.asarray([1], dtype=np.uint64),
        offsets=np.asarray([0, 1], dtype=np.uint64),
        resolution=1,
    )
    second = px.Coverage(
        cells=np.asarray([1], dtype=np.uint64),
        offsets=np.asarray([0, 1, 1], dtype=np.uint64),
        resolution=1,
    )

    with np.testing.assert_raises_regex(ValueError, "same interval count"):
        reduce_coverage([first, second], cell_count=48, cadence_s=60)
