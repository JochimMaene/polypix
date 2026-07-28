"""Shared geometry and plotting helpers for constellation examples."""

from __future__ import annotations

import math
from pathlib import Path
from typing import BinaryIO

import numpy as np
import numpy.typing as npt

import polypix as px

EARTH_RADIUS_KM = 6_378.137
EARTH_MU_KM3_S2 = 398_600.4418
EARTH_ROTATION_RAD_S = 7.2921150e-5


def constellation_centers(
    times_s: npt.NDArray[np.float64],
    *,
    satellite_count: int,
    plane_count: int,
    altitude_km: float,
    inclination_rad: float,
) -> npt.NDArray[np.float64]:
    """Return idealized Walker-like sub-satellite vectors in an Earth-fixed frame."""
    if satellite_count % plane_count:
        raise ValueError("satellite_count must be divisible by plane_count")

    satellites_per_plane = satellite_count // plane_count
    plane = np.repeat(np.arange(plane_count), satellites_per_plane)
    slot = np.tile(np.arange(satellites_per_plane), plane_count)
    raan = 2.0 * math.pi * plane / plane_count
    initial_phase = (
        2.0 * math.pi * slot / satellites_per_plane
        + 2.0 * math.pi * plane / satellite_count
    )
    orbit_radius_km = EARTH_RADIUS_KM + altitude_km
    mean_motion = math.sqrt(EARTH_MU_KM3_S2 / orbit_radius_km**3)
    argument_of_latitude = times_s[:, np.newaxis] * mean_motion + initial_phase

    cos_raan = np.cos(raan)
    sin_raan = np.sin(raan)
    cos_u = np.cos(argument_of_latitude)
    sin_u = np.sin(argument_of_latitude)
    cos_i = math.cos(inclination_rad)
    sin_i = math.sin(inclination_rad)

    inertial_x = cos_raan * cos_u - sin_raan * sin_u * cos_i
    inertial_y = sin_raan * cos_u + cos_raan * sin_u * cos_i
    inertial_z = sin_u * sin_i

    earth_angle = times_s[:, np.newaxis] * EARTH_ROTATION_RAD_S
    cos_earth = np.cos(earth_angle)
    sin_earth = np.sin(earth_angle)
    earth_fixed = np.stack(
        (
            cos_earth * inertial_x + sin_earth * inertial_y,
            -sin_earth * inertial_x + cos_earth * inertial_y,
            inertial_z,
        ),
        axis=-1,
    )
    return earth_fixed / np.linalg.norm(earth_fixed, axis=-1, keepdims=True)


def swath_edges(
    centers: npt.NDArray[np.float64],
    *,
    half_width_rad: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return constant-width left and right ground-swath edge vectors."""
    along_track = np.gradient(centers, axis=0)
    along_track -= np.sum(along_track * centers, axis=-1, keepdims=True) * centers
    along_track /= np.linalg.norm(along_track, axis=-1, keepdims=True)
    cross_track = np.cross(centers, along_track)
    cross_track /= np.linalg.norm(cross_track, axis=-1, keepdims=True)

    left = math.cos(half_width_rad) * centers + math.sin(half_width_rad) * cross_track
    right = math.cos(half_width_rad) * centers - math.sin(half_width_rad) * cross_track
    return left, right


def cap_footprints(
    centers: npt.NDArray[np.float64],
    *,
    radius_rad: float,
    vertex_count: int,
) -> npt.NDArray[np.float64]:
    """Return inscribed regular polygons approximating spherical caps."""
    reference = np.zeros_like(centers)
    reference[:, 2] = 1.0
    near_pole = np.abs(centers[:, 2]) > 0.9
    reference[near_pole] = (1.0, 0.0, 0.0)

    first_tangent = np.cross(reference, centers)
    first_tangent /= np.linalg.norm(first_tangent, axis=1, keepdims=True)
    second_tangent = np.cross(centers, first_tangent)

    angles = np.linspace(0.0, 2.0 * math.pi, vertex_count, endpoint=False)
    boundary_direction = (
        np.cos(angles)[np.newaxis, :, np.newaxis] * first_tangent[:, np.newaxis, :]
        + np.sin(angles)[np.newaxis, :, np.newaxis] * second_tangent[:, np.newaxis, :]
    )
    return (
        math.cos(radius_rad) * centers[:, np.newaxis, :]
        + math.sin(radius_rad) * boundary_direction
    )


def map_coordinates(
    *,
    resolution: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return longitude and latitude for every cell center."""
    cell_count = 12 * 4**resolution
    centers = px.centers(np.arange(cell_count, dtype=np.uint64), resolution)
    return (
        np.arctan2(centers[:, 1], centers[:, 0]),
        np.arcsin(np.clip(centers[:, 2], -1.0, 1.0)),
    )


def plot_global_map(
    values: npt.NDArray[np.float64] | npt.NDArray[np.int64],
    output: Path | BinaryIO,
    *,
    coordinates: tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ],
    visible: npt.NDArray[np.bool_],
    title: str,
    subtitle: str,
    colorbar_label: str,
    footer: str,
    cmap: str,
    norm: object,
    dpi: int = 150,
) -> None:
    """Render cell values on a consistently styled global equal-area map."""
    import matplotlib.pyplot as plt

    longitude, latitude = coordinates
    figure = plt.figure(figsize=(12.0, 6.8), facecolor="#07111f")
    axes = figure.add_subplot(projection="mollweide", facecolor="#101c2d")
    points = axes.scatter(
        longitude[visible],
        latitude[visible],
        c=values[visible],
        cmap=cmap,
        norm=norm,
        marker=".",
        s=2.2,
        linewidths=0,
        rasterized=True,
    )
    axes.grid(color="white", alpha=0.16, linewidth=0.6)
    axes.tick_params(colors="#a9b7ca", labelsize=8)
    figure.suptitle(
        title,
        color="white",
        fontsize=18,
        fontweight="bold",
        y=0.965,
    )
    figure.text(
        0.5,
        0.915,
        subtitle,
        color="#b8c5d6",
        fontsize=10,
        ha="center",
    )
    colorbar = figure.colorbar(
        points,
        ax=axes,
        orientation="horizontal",
        fraction=0.055,
        pad=0.075,
        aspect=45,
    )
    colorbar.set_label(colorbar_label, color="white", fontsize=10)
    colorbar.ax.tick_params(colors="#c6d1df", labelsize=8)
    colorbar.outline.set_edgecolor("#53657a")
    figure.text(
        0.5,
        0.035,
        footer,
        color="#b8c5d6",
        fontsize=9,
        ha="center",
    )
    figure.subplots_adjust(left=0.035, right=0.965, top=0.86, bottom=0.14)

    if isinstance(output, Path):
        output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        dpi=dpi,
        facecolor=figure.get_facecolor(),
        pil_kwargs={"compress_level": 6},
    )
    plt.close(figure)
