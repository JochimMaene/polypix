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
    DOC_FIGURE_URL,
    EARTH_RADIUS_KM,
    map_coordinates,
    plot_global_map,
    read_measurements,
    service_caps,
    write_measurements,
)
from examples.palette import sequential_colormap

FIGURE_PATH = DOC_FIGURE_DIR / "communications-availability.png"
MEASUREMENTS_PATH = DOC_FIGURE_DIR / "communications-availability.json"

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
    minimum_visible: npt.NDArray[np.int64]
    maximum_visible: npt.NDArray[np.int64]
    satellite_count: int
    snapshot_count: int
    covered_pair_count: int
    tle_parsing_elapsed_s: float
    propagation_elapsed_s: float
    cap_geometry_elapsed_s: float
    coverage_elapsed_s: float
    reduction_elapsed_s: float
    analysis_elapsed_s: float


def analyze() -> CommunicationsAnalysis:
    """Sample instantaneous satellites in view for every HEALPix cell."""
    analysis_started = time.perf_counter()

    # --8<-- [start:communications-orbits]
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
    # --8<-- [end:communications-orbits]

    if not np.all(np.isfinite(positions_km)):
        raise ValueError("Astroz returned a non-finite propagated position")

    cell_count = px.cell_count(HEALPIX_RESOLUTION)
    visible_sum = np.zeros(cell_count, dtype=np.int64)
    minimum_visible = np.full(
        cell_count,
        constellation.num_satellites,
        dtype=np.int64,
    )
    maximum_visible = np.zeros(cell_count, dtype=np.int64)
    covered_pair_count = 0
    cap_geometry_elapsed_s = 0.0
    coverage_elapsed_s = 0.0
    reduction_elapsed_s = 0.0

    # Exact caps and their dense per-cell counts are processed one timestamp at
    # a time. The fused count operation never builds the much larger list
    # of repeated cap-cell pairs.
    # --8<-- [start:communications-coverage]
    for snapshot_positions_km in positions_km:
        cap_geometry_started = time.perf_counter()
        centers, radii_rad = service_caps(
            snapshot_positions_km,
            body_radius_km=EARTH_RADIUS_KM,
            minimum_elevation_rad=MINIMUM_ELEVATION_RAD,
        )
        cap_geometry_elapsed_s += time.perf_counter() - cap_geometry_started

        coverage_started = time.perf_counter()
        visible = px.cover_cap(
            centers,
            radii_rad,
            resolution=HEALPIX_RESOLUTION,
            into=px.Count(),
        )
        coverage_elapsed_s += time.perf_counter() - coverage_started

        reduction_started = time.perf_counter()
        covered_pair_count += int(visible.sum())
        visible_sum += visible
        np.minimum(minimum_visible, visible, out=minimum_visible)
        np.maximum(maximum_visible, visible, out=maximum_visible)
        reduction_elapsed_s += time.perf_counter() - reduction_started
    # --8<-- [end:communications-coverage]

    return CommunicationsAnalysis(
        mean_visible=visible_sum / times_min.size,
        minimum_visible=minimum_visible,
        maximum_visible=maximum_visible,
        satellite_count=constellation.num_satellites,
        snapshot_count=times_min.size,
        covered_pair_count=covered_pair_count,
        tle_parsing_elapsed_s=tle_parsing_elapsed_s,
        propagation_elapsed_s=propagation_elapsed_s,
        cap_geometry_elapsed_s=cap_geometry_elapsed_s,
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
        cmap=sequential_colormap(),
        norm=colors.PowerNorm(gamma=0.75, vmin=0.0, vmax=maximum),
        dpi=dpi,
    )


def build_documentation_assets() -> None:
    """Run the scenario, write its map, and record the measurements."""
    result = analyze()
    plotting_started = time.perf_counter()
    coordinates = map_coordinates(resolution=HEALPIX_RESOLUTION)
    plot_availability(result, FIGURE_PATH, coordinates=coordinates, dpi=100)
    plotting_elapsed_s = time.perf_counter() - plotting_started

    write_measurements(
        MEASUREMENTS_PATH,
        {
            "satellite_count": result.satellite_count,
            "snapshot_count": result.snapshot_count,
            "cap_count": result.snapshot_count * result.satellite_count,
            "covered_pair_count": result.covered_pair_count,
            "tle_parsing_ms": result.tle_parsing_elapsed_s * 1_000,
            "propagation_ms": result.propagation_elapsed_s * 1_000,
            "cap_geometry_ms": result.cap_geometry_elapsed_s * 1_000,
            "coverage_ms": result.coverage_elapsed_s * 1_000,
            "reduction_ms": result.reduction_elapsed_s * 1_000,
            "analysis_ms": result.analysis_elapsed_s * 1_000,
            "plotting_ms": plotting_elapsed_s * 1_000,
            "mean_visible_min": float(result.mean_visible.min()),
            "mean_visible_max": float(result.mean_visible.max()),
            "sample_min": int(result.minimum_visible.min()),
            "sample_max": int(result.maximum_visible.max()),
        },
    )


def documentation_html() -> str:
    """Return the recorded figure and measurements as HTML for the docs page."""
    m = read_measurements(MEASUREMENTS_PATH)
    return f"""
<figure class="example-figure">
  <img
    src="{DOC_FIGURE_URL}/{FIGURE_PATH.name}"
    alt="Global map of mean catalogued Starlink objects geometrically visible"
    loading="lazy"
  >
  <figcaption>
    Mean simultaneous catalogued Starlink objects geometrically visible above
    a 25° elevation mask, sampled once per minute for one hour. This is not a
    map of operational Starlink service.
  </figcaption>
</figure>

<div class="example-metrics">
  <div><strong>{m["satellite_count"]:,}</strong><span>catalogued objects</span></div>
  <div><strong>{m["cap_count"]:,}</strong><span>caps evaluated</span></div>
  <div><strong>{m["covered_pair_count"]:,}</strong><span>cap–cell hits counted without storing them</span></div>
</div>

<table class="example-timings">
  <caption>One measured run</caption>
  <thead>
    <tr><th>Stage</th><th>Time</th></tr>
  </thead>
  <tbody>
    <tr><td>Parse pinned TLE snapshot</td>
        <td>{m["tle_parsing_ms"]:.0f} ms</td></tr>
    <tr><td>SGP4 propagation</td>
        <td>{m["propagation_ms"]:.0f} ms</td></tr>
    <tr><td>Service-cap geometry</td>
        <td>{m["cap_geometry_ms"]:.0f} ms</td></tr>
    <tr><td>{m["snapshot_count"]} <code>cover_cap(into=Count())</code> calls</td>
        <td><strong>{m["coverage_ms"]:.0f} ms</strong></td></tr>
    <tr><td>Availability reduction</td>
        <td>{m["reduction_ms"]:.0f} ms</td></tr>
    <tr><td>Complete analysis</td>
        <td>{m["analysis_ms"]:.0f} ms</td></tr>
    <tr><td>Plot and PNG encoding</td>
        <td>{m["plotting_ms"]:.0f} ms</td></tr>
  </tbody>
</table>
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
    print(f"Complete availability analysis took {result.analysis_elapsed_s:.3f} s.")
    print(f"Rendering the map took {plotting_elapsed_s:.3f} s.")
    print(f"Saved {args.output}.")


if __name__ == "__main__":
    main()
