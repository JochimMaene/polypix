"""Small explanatory diagrams for the documentation.

These are deliberately plain: a patch of the HEALPix grid, a region drawn on
top, and the cells Polypix selected. They exist for readers who work with
swaths and footprints every day but have never seen a HEALPix cell.

Every diagram is drawn with Polypix itself, so a picture cannot disagree with
the library. Output is SVG on a transparent background, which keeps the figures
readable under both the light and dark documentation themes.

The tutorial geometry is not defined here. Every figure on the getting-started
page is drawn from the namespace of that page's own doctest blocks, so a figure
cannot disagree with the code printed above it. Edit `docs/guide.md` to change
what these pictures show. Coordinates there are chosen to frame well inside the
drawing window below.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

import polypix as px
from examples.constellation import DOC_FIGURE_DIR
from examples.doc_snippets import guide
from examples.palette import (
    COVERED_CENTER,
    COVERED_FILL,
    CYAN,
    GRID_CENTER,
    GRID_EDGE,
    LABEL,
    MISSED_FILL,
    NAVY_DEEP,
    REGION_LINE,
)

# Chosen so a diagram holds roughly forty cells: large enough to see one, small
# enough that drawing a flat longitude/latitude patch stays honest.
RESOLUTION = 4
WINDOW_LON = (-17.0, 17.0)
WINDOW_LAT = (-13.0, 13.0)


def to_lonlat(vectors: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Degrees longitude and latitude, using the helper the guide publishes."""
    return np.asarray(guide()["to_lonlat"](vectors), dtype=np.float64)


def from_lonlat(
    longitude_deg: npt.ArrayLike,
    latitude_deg: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Unit vectors, using the helper the guide publishes."""
    return np.asarray(
        guide()["unit_vector"](longitude_deg, latitude_deg), dtype=np.float64
    )


def window_cells(resolution: int) -> npt.NDArray[np.int64]:
    """Return every cell whose center falls inside the drawing window."""
    longitude, latitude = np.meshgrid(
        np.linspace(WINDOW_LON[0], WINDOW_LON[1], 400),
        np.linspace(WINDOW_LAT[0], WINDOW_LAT[1], 400),
    )
    directions = from_lonlat(longitude.ravel(), latitude.ravel())
    return np.unique(px.cell_at(directions, resolution))


def cell_polygons(
    cells: npt.NDArray[np.int64],
    resolution: int,
) -> list[npt.NDArray[np.float64]]:
    """Return each cell's four corners as a longitude/latitude polygon."""
    corners = to_lonlat(px.cell_corners(cells, resolution))
    centers = to_lonlat(px.cell_centers(cells, resolution))
    polygons = []
    for corner, center in zip(corners, centers, strict=True):
        # Keep every corner on the same side of the seam as its own center.
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
    cells: npt.NDArray[np.int64],
    resolution: int,
    *,
    covered: set[int] | None = None,
    highlight: dict[int, str] | None = None,
    centers: bool = True,
) -> None:
    """Draw cell outlines, fill the covered cells, and mark cell centers."""
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
        points = to_lonlat(px.cell_centers(cells, resolution))
        # Only genuinely covered cells get the emphasised center. A highlighted
        # near-miss keeps the plain marker, because its center is what excluded it.
        inside = np.array([int(c) in covered for c in cells])
        axes.scatter(
            points[~inside, 0], points[~inside, 1], s=5, color=GRID_CENTER, zorder=3
        )
        axes.scatter(
            points[inside, 0], points[inside, 1], s=9, color=COVERED_CENTER, zorder=4
        )


def save(figure: Any, path: Path) -> None:
    """Write a line-art diagram as SVG."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="svg", transparent=True)
    plt.close(figure)


def save_raster(figure: Any, path: Path, dpi: int = 170) -> None:
    """Write a diagram as PNG.

    The sphere views draw tens of thousands of facets. As SVG that is tens of
    megabytes of vector paths, so those go out as raster instead.
    """
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="png", dpi=dpi, transparent=True)
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
            np.arccos(np.clip(px.cell_corners(cell, RESOLUTION)[0] @ center, -1.0, 1.0))
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
            "overlapped,\ncenter outside",
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
    page = guide()
    lon, lat = page["cap_lon"], page["cap_lat"]
    radius_deg = page["cap_radius_deg"]
    coverage = page["cap_coverage"]

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


def cover_polygon(path: Path) -> None:
    """A convex polygon and the cells it selects."""
    page = guide()
    lon, lat = page["scene_lon"], page["scene_lat"]
    coverage = page["scene_coverage"]

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
    page = guide()
    track_lon = page["track_lon"]
    left, right = page["left_edge"], page["right_edge"]
    coverage = page["swath_coverage"]

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
    """Scattered directions, the cell each lands in, and that cell's center."""
    page = guide()
    lon, lat = page["point_lon"], page["point_lat"]
    cells = page["point_cells"]
    cell_centers = page["point_centers"]

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


def touching_cells(resolution: int) -> dict[int, set[int]]:
    """Return which cells share an edge, keyed by cell.

    Cells sharing an edge share two corners, so counting shared corners is
    enough. `cell_neighbors()` is deliberately not used here: it also returns
    cells meeting at a single point, and those are not counted. A shared vertex
    is not visible enough to read as one region, and including it makes the
    graph too dense to color from a small palette.
    """
    cells = np.arange(12 * 4**resolution, dtype=np.int64)
    corners = np.round(px.cell_corners(cells, resolution), 7)
    at_corner: dict[tuple[float, float, float], set[int]] = {}
    for index, cell_corners in enumerate(corners):
        for corner in cell_corners:
            at_corner.setdefault(tuple(corner), set()).add(index)

    shared_corners: dict[tuple[int, int], int] = {}
    for group in at_corner.values():
        for first in group:
            for second in group:
                if first < second:
                    key = (first, second)
                    shared_corners[key] = shared_corners.get(key, 0) + 1

    touching: dict[int, set[int]] = {int(c): set() for c in cells}
    for (first, second), count in shared_corners.items():
        if count >= 2:
            touching[first].add(second)
            touching[second].add(first)
    return touching


def distinct_colors(resolution: int, palette: list[str]) -> list[str]:
    """Assign palette entries so no two touching cells get the same one.

    Greedy, worst first. A HEALPix grid is planar, so four colors always
    suffice and the palette has room to spare.
    """
    touching = touching_cells(resolution)
    assigned: dict[int, int] = {}
    used = [0] * len(palette)
    for cell in sorted(touching, key=lambda c: -len(touching[c])):
        taken = {assigned[n] for n in touching[cell] if n in assigned}
        # Take the least-used free color, not the first one. Picking the first
        # leaves the dark end of the palette unused and the sphere washed out.
        choice = min(
            (i for i in range(len(palette)) if i not in taken),
            key=lambda i: used[i],
        )
        assigned[cell] = choice
        used[choice] += 1
    for cell, neighbours in touching.items():
        for neighbour in neighbours:
            assert assigned[cell] != assigned[neighbour], "adjacent cells match"
    return [palette[assigned[cell]] for cell in sorted(assigned)]


def sphere_levels(path: Path) -> None:
    """The whole sphere, partitioned at four resolutions.

    HEALPix cell edges are curved, and `cell_corners()` returns only four points per
    cell, so drawing a coarse cell as a flat quadrilateral would misrepresent
    its shape. Instead this draws a fine mesh and colors each fine cell by the
    coarse cell its center falls in, which recovers the true curved boundaries
    using nothing but the public API.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    mesh_resolution = 6
    mesh = np.arange(12 * 4**mesh_resolution, dtype=np.int64)
    mesh_centers = px.cell_centers(mesh, mesh_resolution)
    mesh_corners = px.cell_corners(mesh, mesh_resolution)

    # Slightly rotated view so the pole and the equatorial band are both visible.
    view = np.array([0.62, 0.45, 0.64])
    view /= np.linalg.norm(view)
    facing = mesh_centers @ view > 0.0

    # Five clearly separated steps; the closest pair is 75 apart in RGB, where
    # the previous ramp had two entries only 30 apart and put them on
    # neighbouring cells.
    palette = ["#d7f2f8", "#8fe0ea", CYAN, "#17607d", NAVY_DEEP]
    figure, panels = plt.subplots(
        1, 4, figsize=(7.6, 2.2), subplot_kw={"projection": "3d"}
    )
    figure.patch.set_alpha(0.0)

    for panel, resolution in zip(panels, (0, 1, 2, 3), strict=True):
        assigned = distinct_colors(resolution, palette)
        parents = px.cell_at(mesh_centers[facing], resolution)
        colors = [to_rgb(assigned[int(p)]) for p in parents]
        panel.add_collection3d(
            Poly3DCollection(
                mesh_corners[facing],
                facecolors=colors,
                linewidths=0,
                shade=False,
            )
        )
        panel.set_xlim(-0.72, 0.72)
        panel.set_ylim(-0.72, 0.72)
        panel.set_zlim(-0.72, 0.72)
        panel.set_box_aspect((1, 1, 1))
        panel.view_init(elev=26, azim=36)
        panel.set_axis_off()
        panel.patch.set_alpha(0.0)
        cells = 12 * 4**resolution
        panel.set_title(
            f"resolution {resolution}\n{cells:,} cells",
            color=LABEL,
            fontsize=8,
            pad=-2,
        )

    figure.subplots_adjust(left=0.0, right=1.0, top=0.86, bottom=0.0, wspace=0.0)
    save_raster(figure, path)


def main() -> None:
    center_sampling(DOC_FIGURE_DIR / "center-sampling.svg")
    cover_cap(DOC_FIGURE_DIR / "cover-cap.svg")
    cover_polygon(DOC_FIGURE_DIR / "cover-convex-polygon.svg")
    cover_sweep(DOC_FIGURE_DIR / "cover-sweep.svg")
    cell_at(DOC_FIGURE_DIR / "cell-at.svg")
    resolution_steps(DOC_FIGURE_DIR / "resolution-steps.svg")
    sphere_levels(DOC_FIGURE_DIR / "sphere-levels.png")


if __name__ == "__main__":
    main()
