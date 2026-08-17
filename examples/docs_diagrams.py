"""Small explanatory diagrams for the documentation.

These are deliberately plain: a patch of the HEALPix grid, a region drawn on
top, and the cells Polypix selected. They exist for readers who work with
swaths and footprints every day but have never seen a HEALPix cell.

Every diagram is drawn with Polypix itself, so a picture cannot disagree with
the library. Output is SVG on a transparent background, which keeps the figures
readable under both the light and dark documentation themes.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

import polypix as px
from examples.constellation import DOC_FIGURE_DIR
from examples.palette import (
    COVERED_CENTER,
    COVERED_FILL,
    GRID_CENTER,
    GRID_EDGE,
    LABEL,
    MISSED_FILL,
    REGION_LINE,
)

# Chosen so a diagram holds roughly forty cells: large enough to see one, small
# enough that drawing a flat longitude/latitude patch stays honest.
RESOLUTION = 4
WINDOW_LON = (-17.0, 17.0)
WINDOW_LAT = (-13.0, 13.0)


def to_lonlat(vectors: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return degrees longitude and latitude for unit vectors."""
    longitude = np.degrees(np.arctan2(vectors[..., 1], vectors[..., 0]))
    latitude = np.degrees(np.arcsin(np.clip(vectors[..., 2], -1.0, 1.0)))
    return np.stack((longitude, latitude), axis=-1)


def from_lonlat(
    longitude_deg: npt.ArrayLike,
    latitude_deg: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Return unit vectors for longitudes and latitudes in degrees."""
    return np.asarray(unit_vector(longitude_deg, latitude_deg), dtype=np.float64)


# --8<-- [start:unit-vector]
def unit_vector(lon_deg, lat_deg):
    """Cartesian directions for longitudes and latitudes in degrees."""
    lon, lat = np.radians(lon_deg), np.radians(lat_deg)
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        axis=-1,
    )


# --8<-- [end:unit-vector]


def window_cells(resolution: int) -> npt.NDArray[np.uint64]:
    """Return every cell whose centre falls inside the drawing window."""
    longitude, latitude = np.meshgrid(
        np.linspace(WINDOW_LON[0], WINDOW_LON[1], 400),
        np.linspace(WINDOW_LAT[0], WINDOW_LAT[1], 400),
    )
    directions = from_lonlat(longitude.ravel(), latitude.ravel())
    return np.unique(px.cell_at(directions, resolution))


def cell_polygons(
    cells: npt.NDArray[np.uint64],
    resolution: int,
) -> list[npt.NDArray[np.float64]]:
    """Return each cell's four corners as a longitude/latitude polygon."""
    corners = to_lonlat(px.corners(cells, resolution))
    centers = to_lonlat(px.centers(cells, resolution))
    polygons = []
    for corner, center in zip(corners, centers, strict=True):
        # Keep every corner on the same side of the seam as its own centre.
        longitude = corner[:, 0] - 360.0 * np.round((corner[:, 0] - center[0]) / 360.0)
        polygons.append(np.stack((longitude, corner[:, 1]), axis=-1))
    return polygons


def cap_outline(
    center_lon: float,
    center_lat: float,
    radius_deg: float,
    samples: int = 240,
) -> npt.NDArray[np.float64]:
    """Return the longitude/latitude outline of a spherical cap."""
    center = from_lonlat(center_lon, center_lat)
    first = np.cross([0.0, 0.0, 1.0], center)
    first /= np.linalg.norm(first)
    second = np.cross(center, first)
    angles = np.linspace(0.0, 2.0 * math.pi, samples)
    radius = math.radians(radius_deg)
    points = math.cos(radius) * center + math.sin(radius) * (
        np.cos(angles)[:, None] * first + np.sin(angles)[:, None] * second
    )
    return to_lonlat(points)


def new_axes(
    width: float = 5.6,
    height: float = 4.3,
    panels: int = 1,
) -> tuple[Any, Any]:
    """Return bare transparent axes sized for the documentation column."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, panels, figsize=(width, height))
    figure.patch.set_alpha(0.0)
    for panel in np.atleast_1d(axes):
        panel.set_facecolor("none")
        panel.set_xlim(*WINDOW_LON)
        panel.set_ylim(*WINDOW_LAT)
        panel.set_aspect("equal")
        panel.axis("off")
    figure.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.04)
    return figure, axes


def draw_grid(
    axes: Any,
    cells: npt.NDArray[np.uint64],
    resolution: int,
    *,
    covered: set[int] | None = None,
    highlight: dict[int, str] | None = None,
    centers: bool = True,
) -> None:
    """Draw cell outlines, fill the covered cells, and mark cell centres."""
    from matplotlib.patches import Polygon

    covered = covered or set()
    highlight = highlight or {}
    for cell, polygon in zip(cells, cell_polygons(cells, resolution), strict=True):
        key = int(cell)
        color = highlight.get(key) or (COVERED_FILL if key in covered else None)
        axes.add_patch(
            Polygon(
                polygon,
                closed=True,
                facecolor=color or "none",
                edgecolor=GRID_EDGE,
                alpha=0.38 if color else 1.0,
                linewidth=0.7,
                zorder=1 if color else 2,
            )
        )
    if centers:
        points = to_lonlat(px.centers(cells, resolution))
        # Only genuinely covered cells get the emphasised centre. A highlighted
        # near-miss keeps the plain marker, because its centre is what excluded it.
        inside = np.array([int(c) in covered for c in cells])
        axes.scatter(
            points[~inside, 0], points[~inside, 1], s=5, color=GRID_CENTER, zorder=3
        )
        axes.scatter(
            points[inside, 0], points[inside, 1], s=9, color=COVERED_CENTER, zorder=4
        )


def save(figure: Any, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="svg", transparent=True)
    plt.close(figure)


def outline(axes: Any, points: npt.NDArray[np.float64]) -> None:
    axes.plot(points[:, 0], points[:, 1], color=REGION_LINE, linewidth=2.0, zorder=5)


def center_sampling(path: Path) -> None:
    """A cap, the cells it selects, and one cell it overlaps but misses."""
    cells = window_cells(RESOLUTION)
    center, radius = from_lonlat(0.0, 0.0), 7.5
    coverage = px.cover_cap(center, math.radians(radius), RESOLUTION)
    covered = {int(c) for c in coverage.cells}

    missed: tuple[int, npt.NDArray[np.float64]] | None = None
    for cell, polygon in zip(cells, cell_polygons(cells, RESOLUTION), strict=True):
        if int(cell) in covered:
            continue
        corner_angles = np.degrees(
            np.arccos(np.clip(px.corners(cell, RESOLUTION)[0] @ center, -1.0, 1.0))
        )
        if (corner_angles < radius).any():
            missed = (int(cell), polygon)
            break

    figure, axes = new_axes()
    draw_grid(
        axes,
        cells,
        RESOLUTION,
        covered=covered,
        highlight={missed[0]: MISSED_FILL} if missed else None,
    )
    outline(axes, cap_outline(0.0, 0.0, radius))
    if missed is not None:
        anchor = missed[1].mean(axis=0)
        axes.annotate(
            "overlapped,\ncentre outside",
            xy=(anchor[0], anchor[1]),
            xytext=(anchor[0] + 2.4, anchor[1] + 1.4),
            color=LABEL,
            fontsize=8.5,
            ha="left",
            va="bottom",
            arrowprops={"arrowstyle": "-", "color": LABEL, "linewidth": 0.8},
            zorder=6,
        )
    save(figure, path)


def cover_cap(path: Path) -> None:
    """Two caps of different radii and the cells each selects."""
    # --8<-- [start:cover-cap]
    lon, lat, radius_deg = [-7.5, 8.0], [3.0, -4.0], [6.0, 4.0]

    coverage = px.cover_cap(unit_vector(lon, lat), np.radians(radius_deg), resolution=4)
    assert coverage.counts.tolist() == [9, 4]
    # --8<-- [end:cover-cap]

    figure, axes = new_axes()
    draw_grid(
        axes,
        window_cells(RESOLUTION),
        RESOLUTION,
        covered={int(c) for c in coverage.cells},
    )
    for cap in zip(lon, lat, radius_deg, strict=True):
        outline(axes, cap_outline(*cap))
    save(figure, path)


def cover_footprint(path: Path) -> None:
    """A convex polygon and the cells it selects."""
    # --8<-- [start:cover-footprint]
    lon = [-9.0, 7.0, 11.0, -2.0]
    lat = [-6.0, -8.0, 4.0, 8.0]

    coverage = px.cover_footprint(unit_vector(lon, lat), resolution=4)
    assert len(coverage) == 1
    # --8<-- [end:cover-footprint]

    figure, axes = new_axes()
    draw_grid(
        axes,
        window_cells(RESOLUTION),
        RESOLUTION,
        covered={int(c) for c in coverage.cells},
    )
    outline(axes, np.array([*zip(lon, lat, strict=True), (lon[0], lat[0])]))
    save(figure, path)


def cover_sweep(path: Path) -> None:
    """A sampled sweep, its quadrilaterals, and the cells they select."""
    # --8<-- [start:cover-sweep]
    track_lon = np.linspace(-13.0, 13.0, 7)
    track_lat = 5.0 * np.sin(np.radians(track_lon * 7.0))

    left = unit_vector(track_lon, track_lat + 3.2)
    right = unit_vector(track_lon, track_lat - 3.2)
    coverage = px.cover_sweep(left, right, resolution=4)
    assert len(coverage) == len(track_lon) - 1
    # --8<-- [end:cover-sweep]

    figure, axes = new_axes()
    draw_grid(
        axes,
        window_cells(RESOLUTION),
        RESOLUTION,
        covered={int(c) for c in coverage.cells},
    )
    left_ll, right_ll = to_lonlat(left), to_lonlat(right)
    for i in range(len(track_lon) - 1):
        quad = np.array(
            [left_ll[i], right_ll[i], right_ll[i + 1], left_ll[i + 1], left_ll[i]]
        )
        outline(axes, quad)
    axes.scatter(
        np.concatenate([left_ll[:, 0], right_ll[:, 0]]),
        np.concatenate([left_ll[:, 1], right_ll[:, 1]]),
        s=14,
        color=REGION_LINE,
        zorder=6,
    )
    save(figure, path)


def cell_at(path: Path) -> None:
    """Scattered directions, the cell each lands in, and that cell's centre."""
    # --8<-- [start:cell-at]
    lon = [-11.0, -3.0, 5.0, 12.0]
    lat = [6.0, -7.0, 2.0, -9.0]

    cells = px.cell_at(unit_vector(lon, lat), resolution=4)
    cell_centers = px.centers(cells, resolution=4)
    assert cells.shape == (4,) and cell_centers.shape == (4, 3)
    # --8<-- [end:cell-at]

    points = np.stack([lon, lat], axis=-1)
    centers = to_lonlat(cell_centers)

    figure, axes = new_axes()
    draw_grid(
        axes, window_cells(RESOLUTION), RESOLUTION, covered={int(c) for c in cells}
    )
    axes.scatter(points[:, 0], points[:, 1], s=34, color=REGION_LINE, zorder=6)
    for point, center in zip(points, centers, strict=True):
        axes.annotate(
            "",
            xy=(center[0], center[1]),
            xytext=(point[0], point[1]),
            arrowprops={"arrowstyle": "->", "color": LABEL, "linewidth": 0.9},
            zorder=6,
        )
    save(figure, path)


def resolution_steps(path: Path) -> None:
    """The same cap on three grids, one resolution apart."""
    center, radius = from_lonlat(0.0, 0.0), 7.5
    figure, panels = new_axes(width=7.4, height=2.1, panels=3)
    for panel, resolution in zip(panels, (3, 4, 5), strict=True):
        coverage = px.cover_cap(center, math.radians(radius), resolution)
        draw_grid(
            panel,
            window_cells(resolution),
            resolution,
            covered={int(c) for c in coverage.cells},
            centers=False,
        )
        outline(panel, cap_outline(0.0, 0.0, radius))
        panel.set_title(f"resolution {resolution}", color=LABEL, fontsize=8.5, pad=4)
    save(figure, path)


def main() -> None:
    center_sampling(DOC_FIGURE_DIR / "center-sampling.svg")
    cover_cap(DOC_FIGURE_DIR / "cover-cap.svg")
    cover_footprint(DOC_FIGURE_DIR / "cover-footprint.svg")
    cover_sweep(DOC_FIGURE_DIR / "cover-sweep.svg")
    cell_at(DOC_FIGURE_DIR / "cell-at.svg")
    resolution_steps(DOC_FIGURE_DIR / "resolution-steps.svg")


if __name__ == "__main__":
    main()
