"""Map one hour of geometric visibility from a historical Starlink snapshot."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import astroz
import numpy as np
import numpy.typing as npt

import polypix as px
from examples.constellation import (
    DOC_FIGURE_DIR,
    DOC_FIGURE_INCLUDE_DIR,
    EARTH_RADIUS_KM,
    map_coordinates,
    plot_global_map,
    service_caps,
)
from examples.palette import (
    COUNT_CMAP,
    MAP_BACKGROUND,
    MAP_GRID,
    MAP_MUTED,
    MAP_PANEL,
    MAP_RULE,
    MAP_TEXT,
    REGION_LINE,
)

FIGURE_PATH = DOC_FIGURE_DIR / "communications-availability.png"
GERMANY_FIGURE_PATH = DOC_FIGURE_DIR / "communications-germany.png"

# Permanent CelesTrak STARLINK group snapshot, retrieved 2026-07-29 from
# https://celestrak.org/NORAD/elements/gp.php?GROUP=STARLINK&FORMAT=TLE.
# The example intentionally has no download or refresh path.
TLE_PATH = Path(__file__).with_name("data") / "starlink-2026-07-29.tle"
MAP_PATH = Path(__file__).with_name("data") / "central-europe-borders.json"
ANALYSIS_START = datetime(2026, 7, 29, tzinfo=UTC)

MINIMUM_ELEVATION_RAD = math.radians(25.0)

DURATION_MIN = 60
CADENCE_MIN = 1
HEALPIX_RESOLUTION = 6
GERMANY_RESOLUTION = 9


@dataclass(frozen=True)
class CommunicationsAnalysis:
    """Availability results and timings for the communications scenario."""

    mean_visible: npt.NDArray[np.float64]
    satellite_count: int
    snapshot_count: int
    covered_pair_count: int
    tle_parsing_elapsed_s: float
    propagation_elapsed_s: float
    coverage_elapsed_s: float
    reduction_elapsed_s: float
    analysis_elapsed_s: float


# --8<-- [start:communications-coverage]
def count_visible(
    snapshot_positions_km: npt.NDArray[np.float64],
    *,
    resolution: int = HEALPIX_RESOLUTION,
    candidate_cells: npt.NDArray[np.int64] | None = None,
) -> npt.NDArray[np.int64]:
    """Count the objects in view of every cell, at one timestamp."""
    centers, radii_rad = service_caps(
        snapshot_positions_km,
        body_radius_km=EARTH_RADIUS_KM,
        minimum_elevation_rad=MINIMUM_ELEVATION_RAD,
    )
    return px.cover_cap(
        centers,
        radii_rad,
        resolution=resolution,
        candidate_cells=candidate_cells,
        reduce=px.Count(),
    )


# --8<-- [end:communications-coverage]


CENTRAL_EUROPE_BORDERS = json.loads(MAP_PATH.read_text())
GERMANY_BOUNDARY_LON_LAT_DEG = np.asarray(
    CENTRAL_EUROPE_BORDERS["DEU"][0], dtype=np.float64
)


# --8<-- [start:germany-aoi]
def germany_cells() -> npt.NDArray[np.int64]:
    """Return the resolution-9 cells whose centres lie inside Germany."""
    longitude, latitude = np.radians(GERMANY_BOUNDARY_LON_LAT_DEG).T
    boundary = np.column_stack(
        (
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        )
    )
    return px.cover_polygon(px.Polygon(boundary), GERMANY_RESOLUTION).cells


# --8<-- [end:germany-aoi]


def analyze(
    *,
    resolution: int = HEALPIX_RESOLUTION,
    candidate_cells: npt.NDArray[np.int64] | None = None,
) -> CommunicationsAnalysis:
    """Sample instantaneous satellites in view for the requested cells."""
    analysis_started = time.perf_counter()

    parsing_started = time.perf_counter()
    constellation = astroz.Constellation(str(TLE_PATH))
    tle_parsing_elapsed_s = time.perf_counter() - parsing_started

    times_min = np.arange(
        0,
        DURATION_MIN + CADENCE_MIN,
        CADENCE_MIN,
        dtype=np.float64,
    )
    propagation_started = time.perf_counter()
    positions_km = astroz.propagate(
        constellation,
        times_min,
        start_time=ANALYSIS_START,
        output="ecef",
    )
    propagation_elapsed_s = time.perf_counter() - propagation_started

    if not np.all(np.isfinite(positions_km)):
        raise ValueError("Astroz returned a non-finite propagated position")

    cell_count = (
        px.cell_count(resolution) if candidate_cells is None else candidate_cells.size
    )
    visible_sum = np.zeros(cell_count, dtype=np.int64)
    covered_pair_count = 0
    coverage_elapsed_s = 0.0
    reduction_elapsed_s = 0.0

    # One timestamp at a time, so that the fused count never builds the much
    # larger list of repeated cap-cell pairs.
    for snapshot_positions_km in positions_km:
        coverage_started = time.perf_counter()
        visible = count_visible(
            snapshot_positions_km,
            resolution=resolution,
            candidate_cells=candidate_cells,
        )
        coverage_elapsed_s += time.perf_counter() - coverage_started

        reduction_started = time.perf_counter()
        covered_pair_count += int(visible.sum())
        visible_sum += visible
        reduction_elapsed_s += time.perf_counter() - reduction_started

    return CommunicationsAnalysis(
        mean_visible=visible_sum / times_min.size,
        satellite_count=constellation.num_satellites,
        snapshot_count=times_min.size,
        covered_pair_count=covered_pair_count,
        tle_parsing_elapsed_s=tle_parsing_elapsed_s,
        propagation_elapsed_s=propagation_elapsed_s,
        coverage_elapsed_s=coverage_elapsed_s,
        reduction_elapsed_s=reduction_elapsed_s,
        analysis_elapsed_s=time.perf_counter() - analysis_started,
    )


def plot_availability(
    result: CommunicationsAnalysis,
    output: Path | BytesIO,
    *,
    coordinates: tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]
    | None = None,
    dpi: int = 150,
) -> None:
    """Render the time-averaged number of catalogued objects in view."""
    import matplotlib.colors as colors

    if coordinates is None:
        coordinates = map_coordinates(resolution=HEALPIX_RESOLUTION)
    maximum = float(result.mean_visible.max())
    plot_global_map(
        result.mean_visible,
        output,
        coordinates=coordinates,
        visible=np.ones(result.mean_visible.size, dtype=bool),
        resolution=HEALPIX_RESOLUTION,
        colorbar_label="Mean catalogued objects in view",
        cmap=COUNT_CMAP,
        norm=colors.PowerNorm(gamma=0.75, vmin=0.0, vmax=maximum),
        dpi=dpi,
    )


def plot_germany_availability(
    result: CommunicationsAnalysis,
    cells: npt.NDArray[np.int64],
    output: Path,
    *,
    dpi: int = 150,
) -> None:
    """Render availability in the Germany area of interest."""
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    corners = px.cell_corners(cells, GERMANY_RESOLUTION)
    polygons = np.stack(
        (
            np.degrees(np.arctan2(corners[..., 1], corners[..., 0])),
            np.degrees(np.arcsin(np.clip(corners[..., 2], -1.0, 1.0))),
        ),
        axis=-1,
    )
    minimum = float(result.mean_visible.min())
    maximum = float(result.mean_visible.max())
    norm = colors.PowerNorm(gamma=0.75, vmin=minimum, vmax=maximum)

    figure, axes = plt.subplots(
        figsize=(7.0, 6.0), facecolor=MAP_BACKGROUND, constrained_layout=True
    )
    axes.set_facecolor(MAP_PANEL)
    country_polygons = [
        np.asarray(ring) for rings in CENTRAL_EUROPE_BORDERS.values() for ring in rings
    ]
    countries = PolyCollection(
        country_polygons,
        facecolors=MAP_BACKGROUND,
        edgecolors=MAP_RULE,
        linewidths=0.7,
    )
    axes.add_collection(countries)
    collection = PolyCollection(
        polygons,
        array=result.mean_visible,
        cmap=COUNT_CMAP,
        norm=norm,
        edgecolors="none",
        rasterized=True,
    )
    axes.add_collection(collection)
    outline = np.vstack((GERMANY_BOUNDARY_LON_LAT_DEG, GERMANY_BOUNDARY_LON_LAT_DEG[0]))
    axes.plot(outline[:, 0], outline[:, 1], color=REGION_LINE, linewidth=2.0)
    axes.set_xlim(4.8, 15.7)
    axes.set_ylim(46.7, 55.3)
    axes.set_aspect(1.0 / math.cos(math.radians(51.0)))
    axes.set_xlabel("Longitude", color=MAP_TEXT)
    axes.set_ylabel("Latitude", color=MAP_TEXT)
    axes.grid(color=MAP_GRID, alpha=0.25, linewidth=0.55)
    axes.tick_params(colors=MAP_MUTED)
    for spine in axes.spines.values():
        spine.set_edgecolor(MAP_RULE)
    colorbar = figure.colorbar(collection, ax=axes, orientation="horizontal", pad=0.1)
    colorbar.set_label("Mean catalogued objects in view", color=MAP_TEXT)
    colorbar.ax.tick_params(colors=MAP_MUTED)
    colorbar.outline.set_edgecolor(MAP_RULE)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, facecolor=figure.get_facecolor())
    plt.close(figure)


def build_documentation_assets() -> str:
    """Run the scenario, write its map, and return its documentation page."""
    result = analyze()
    coordinates = map_coordinates(resolution=HEALPIX_RESOLUTION)
    plot_availability(result, FIGURE_PATH, coordinates=coordinates, dpi=100)
    aoi_cells = germany_cells()
    germany = analyze(resolution=GERMANY_RESOLUTION, candidate_cells=aoi_cells)
    plot_germany_availability(germany, aoi_cells, GERMANY_FIGURE_PATH, dpi=120)

    return f"""
```{{figure}} {DOC_FIGURE_INCLUDE_DIR}/{FIGURE_PATH.name}
:alt: Global map of mean catalogued Starlink objects geometrically visible
:figclass: example-figure

Mean simultaneous catalogued Starlink objects geometrically visible above a 25°
elevation mask, sampled once per minute for one hour.
```
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("communication-availability.png"),
    )
    args = parser.parse_args()

    result = analyze()
    plotting_started = time.perf_counter()
    plot_availability(result, args.output)
    plotting_elapsed_s = time.perf_counter() - plotting_started
    print(
        f"Processed {result.snapshot_count * result.satellite_count:,} exact service "
        f"caps into per-cell counts in {result.coverage_elapsed_s:.3f} s."
    )
    print(f"Counted {result.covered_pair_count:,} cap-cell hits without storing any.")
    print(f"Complete availability analysis took {result.analysis_elapsed_s:.3f} s.")
    print(f"Rendering the map took {plotting_elapsed_s:.3f} s.")
    print(f"Saved {args.output}.")


if __name__ == "__main__":
    main()
