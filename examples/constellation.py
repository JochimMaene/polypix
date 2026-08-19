"""Shared geometry and plotting helpers for constellation examples."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import numpy.typing as npt

import polypix as px
from examples.palette import (
    MAP_BACKGROUND,
    MAP_GRID,
    MAP_MUTED,
    MAP_PANEL,
    MAP_RULE,
    MAP_TEXT,
)

EARTH_RADIUS_KM = 6_378.137
EARTH_MU_KM3_S2 = 398_600.4418
EARTH_ROTATION_RAD_S = 7.2921150e-5

# Figures for the documentation build are written here and copied into the
# generated site. The path is relative to the repository root, which is the
# working directory during a Sphinx build.
DOC_FIGURE_DIR = Path("docs/assets/generated")

# Path from a built page at examples/<name>.html to the assets copied by Sphinx.
DOC_FIGURE_URL = "../generated"


def write_measurements(path: Path, measurements: dict[str, Any]) -> None:
    """Record one run's measurements next to the figures it produced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(measurements, indent=2, sort_keys=True) + "\n")


def read_measurements(path: Path) -> dict[str, Any]:
    """Load measurements recorded by the documentation asset step."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `pixi run --environment docs docs-figures` "
            "to execute the examples before building the documentation."
        )
    loaded: dict[str, Any] = json.loads(path.read_text())
    return loaded


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


def service_caps(
    positions_km: npt.NDArray[np.float64],
    *,
    body_radius_km: float,
    minimum_elevation_rad: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return ground-service cap centers and altitude-dependent radii."""
    orbit_radii_km = np.linalg.norm(positions_km, axis=1)
    centers = positions_km / orbit_radii_km[:, np.newaxis]
    radii = (
        np.arccos(
            np.clip(
                body_radius_km / orbit_radii_km * math.cos(minimum_elevation_rad),
                -1.0,
                1.0,
            )
        )
        - minimum_elevation_rad
    )
    return centers, radii


def map_coordinates(
    *,
    resolution: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return longitude and latitude for every cell center."""
    cell_count = px.cell_count(resolution)
    centers = px.cell_centers(np.arange(cell_count, dtype=np.int64), resolution)
    return (
        np.arctan2(centers[:, 1], centers[:, 0]),
        np.arcsin(np.clip(centers[:, 2], -1.0, 1.0)),
    )


def tiling_marker_size(
    figure: object,
    axes: object,
    *,
    resolution: int,
) -> float:
    """Return a scatter area in points² that makes cell markers tile the map.

    A HEALPix cell at this resolution occupies a fixed angular width, and the
    Mollweide axes map 360 degrees across their full width. Markers narrower
    than that spacing leave the background visible between cells, which reads
    as noise rather than as a coverage field.
    """
    figure.canvas.draw()  # type: ignore[attr-defined]
    width_pt = axes.get_window_extent().width / figure.dpi * 72.0  # type: ignore[attr-defined]
    cell_deg = math.degrees(math.sqrt(4.0 * math.pi / (12 * 4**resolution)))
    spacing_pt = width_pt * cell_deg / 360.0
    # Overlap neighboring markers enough to avoid white seams after PNG
    # downsampling in the documentation site.
    return float((2.1 * spacing_pt) ** 2)


def plot_global_map(
    values: npt.NDArray[np.float64] | npt.NDArray[np.int64],
    output: Path | BinaryIO,
    *,
    coordinates: tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ],
    visible: npt.NDArray[np.bool_],
    resolution: int,
    colorbar_label: str,
    cmap: Any,
    norm: object,
    colorbar_ticks: Sequence[float] | None = None,
    extend: str = "neither",
    dpi: int = 150,
) -> None:
    """Render cell values on a consistently styled global equal-area map."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator, ScalarFormatter

    longitude, latitude = coordinates
    figure = plt.figure(figsize=(12.0, 5.75), facecolor=MAP_BACKGROUND)
    axes = figure.add_subplot(projection="mollweide", facecolor=MAP_PANEL)
    points = axes.scatter(
        longitude[visible],
        latitude[visible],
        c=values[visible],
        cmap=cmap,
        norm=norm,
        marker="o",
        s=tiling_marker_size(figure, axes, resolution=resolution),
        linewidths=0,
        rasterized=True,
    )
    axes.grid(color=MAP_GRID, alpha=0.28, linewidth=0.55)
    axes.tick_params(colors=MAP_MUTED, labelsize=8)
    axes.spines["geo"].set_edgecolor(MAP_RULE)
    colorbar = figure.colorbar(
        points,
        ax=axes,
        orientation="horizontal",
        fraction=0.055,
        pad=0.075,
        aspect=45,
        extend=extend,
    )
    colorbar.set_label(colorbar_label, color=MAP_TEXT, fontsize=10)
    colorbar.ax.tick_params(colors=MAP_MUTED, labelsize=8)
    colorbar.outline.set_edgecolor(MAP_RULE)
    if colorbar_ticks is not None:
        # A logarithmic scale otherwise labels an hours axis "10^0".
        colorbar.ax.xaxis.set_minor_locator(NullLocator())
        colorbar.set_ticks(list(colorbar_ticks))
        colorbar.ax.xaxis.set_major_formatter(ScalarFormatter())
    figure.subplots_adjust(left=0.025, right=0.975, top=0.97, bottom=0.13)

    encoded = BytesIO()
    figure.savefig(encoded, format="png", dpi=dpi, facecolor=figure.get_facecolor())
    plt.close(figure)

    data = _quantized_png(encoded.getvalue())
    if isinstance(output, Path):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(data)
    else:
        output.write(data)


def _quantized_png(data: bytes) -> bytes:
    """Re-encode a colormapped map to a palette PNG.

    These figures draw a single colormap over a flat background, so 256 palette
    entries are visually indistinguishable from truecolor at roughly a third of
    the bytes.
    """
    from PIL import Image

    image = Image.open(BytesIO(data)).convert("RGB")
    palette = image.quantize(colors=256, dither=Image.Dither.NONE)
    out = BytesIO()
    palette.save(out, format="PNG", optimize=True)
    return out.getvalue() if out.tell() < len(data) else data
