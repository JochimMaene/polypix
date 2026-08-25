"""Map one hour of geometric visibility from a historical Starlink snapshot."""

from __future__ import annotations

import argparse
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
from examples.palette import COUNT_CMAP

FIGURE_PATH = DOC_FIGURE_DIR / "communications-availability.png"

# Permanent CelesTrak STARLINK group snapshot, retrieved 2026-07-29 from
# https://celestrak.org/NORAD/elements/gp.php?GROUP=STARLINK&FORMAT=TLE.
# The example intentionally has no download or refresh path.
TLE_PATH = Path(__file__).with_name("data") / "starlink-2026-07-29.tle"
ANALYSIS_START = datetime(2026, 7, 29, tzinfo=UTC)

MINIMUM_ELEVATION_RAD = math.radians(25.0)

DURATION_MIN = 60
CADENCE_MIN = 1
HEALPIX_RESOLUTION = 6


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
        resolution=HEALPIX_RESOLUTION,
        reduce=px.Count(),
    )


# --8<-- [end:communications-coverage]


def analyze() -> CommunicationsAnalysis:
    """Sample instantaneous satellites in view for every HEALPix cell."""
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

    cell_count = px.cell_count(HEALPIX_RESOLUTION)
    visible_sum = np.zeros(cell_count, dtype=np.int64)
    covered_pair_count = 0
    coverage_elapsed_s = 0.0
    reduction_elapsed_s = 0.0

    # One timestamp at a time, so that the fused count never builds the much
    # larger list of repeated cap-cell pairs.
    for snapshot_positions_km in positions_km:
        coverage_started = time.perf_counter()
        visible = count_visible(snapshot_positions_km)
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


def build_documentation_assets() -> str:
    """Run the scenario, write its map, and return its documentation page."""
    result = analyze()
    coordinates = map_coordinates(resolution=HEALPIX_RESOLUTION)
    plot_availability(result, FIGURE_PATH, coordinates=coordinates, dpi=100)

    return f"""
```{{figure}} {DOC_FIGURE_INCLUDE_DIR}/{FIGURE_PATH.name}
:alt: Global map of mean catalogued Starlink objects geometrically visible
:figclass: example-figure

Mean simultaneous catalogued Starlink objects geometrically visible above a 25°
elevation mask, sampled once per minute for one hour.
```

```{{list-table}} One measured run
:header-rows: 1
:class: example-timings
:widths: 70 30

* - Stage
  - Time
* - Parse pinned TLE snapshot
  - {result.tle_parsing_elapsed_s * 1_000:.0f} ms
* - SGP4 propagation
  - {result.propagation_elapsed_s * 1_000:.0f} ms
* - {result.snapshot_count} cap builds and `cover_cap(reduce=Count())` calls
  - **{result.coverage_elapsed_s * 1_000:.0f} ms**
* - Complete analysis
  - {result.analysis_elapsed_s * 1_000:.0f} ms
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
