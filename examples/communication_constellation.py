"""Map one hour of availability from a Starlink-like communications constellation."""

from __future__ import annotations

import argparse
import base64
import math
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import numpy.typing as npt

import polypix as px
from examples.constellation import (
    EARTH_RADIUS_KM,
    cap_footprints,
    constellation_centers,
    map_coordinates,
    plot_global_map,
)

SATELLITE_COUNT = 500
PLANE_COUNT = 20
ALTITUDE_KM = 550.0
INCLINATION_RAD = math.radians(53.0)
MINIMUM_ELEVATION_RAD = math.radians(25.0)
FOOTPRINT_VERTEX_COUNT = 16

DURATION_S = 60 * 60
CADENCE_S = 60
HEALPIX_RESOLUTION = 6


@dataclass(frozen=True)
class CommunicationsAnalysis:
    """Availability results and timings for the communications scenario."""

    mean_visible: npt.NDArray[np.float64]
    minimum_visible: npt.NDArray[np.int64]
    maximum_visible: npt.NDArray[np.int64]
    materialized_count: int
    geometry_elapsed_s: float
    coverage_elapsed_s: float
    reduction_elapsed_s: float
    analysis_elapsed_s: float


def service_radius_rad() -> float:
    """Return the Earth-centered service radius at the elevation mask."""
    orbit_radius_km = EARTH_RADIUS_KM + ALTITUDE_KM
    return (
        math.acos(EARTH_RADIUS_KM / orbit_radius_km * math.cos(MINIMUM_ELEVATION_RAD))
        - MINIMUM_ELEVATION_RAD
    )


def analyze() -> CommunicationsAnalysis:
    """Sample instantaneous satellites in view for every HEALPix cell."""
    analysis_started = time.perf_counter()
    geometry_started = time.perf_counter()

    # --8<-- [start:communications-orbits]
    times_s = np.arange(
        0,
        DURATION_S + CADENCE_S,
        CADENCE_S,
        dtype=np.float64,
    )
    centers = constellation_centers(
        times_s,
        satellite_count=SATELLITE_COUNT,
        plane_count=PLANE_COUNT,
        altitude_km=ALTITUDE_KM,
        inclination_rad=INCLINATION_RAD,
    )
    # --8<-- [end:communications-orbits]
    geometry_elapsed_s = time.perf_counter() - geometry_started

    cell_count = 12 * 4**HEALPIX_RESOLUTION
    visible_sum = np.zeros(cell_count, dtype=np.int64)
    minimum_visible = np.full(cell_count, SATELLITE_COUNT, dtype=np.int64)
    maximum_visible = np.zeros(cell_count, dtype=np.int64)
    materialized_count = 0
    coverage_elapsed_s = 0.0
    reduction_elapsed_s = 0.0
    radius_rad = service_radius_rad()

    # Work on one timestamp at a time so only 500 footprints are materialized.
    # --8<-- [start:communications-coverage]
    for snapshot_centers in centers:
        geometry_started = time.perf_counter()
        footprints = cap_footprints(
            snapshot_centers,
            radius_rad=radius_rad,
            vertex_count=FOOTPRINT_VERTEX_COUNT,
        )
        geometry_elapsed_s += time.perf_counter() - geometry_started

        coverage_started = time.perf_counter()
        coverage = px.cover_footprint(
            footprints,
            resolution=HEALPIX_RESOLUTION,
        )
        coverage_elapsed_s += time.perf_counter() - coverage_started
        materialized_count += int(coverage.cells.size)

        reduction_started = time.perf_counter()
        visible = np.bincount(
            coverage.cells.astype(np.intp, copy=False),
            minlength=cell_count,
        )
        visible_sum += visible
        np.minimum(minimum_visible, visible, out=minimum_visible)
        np.maximum(maximum_visible, visible, out=maximum_visible)
        reduction_elapsed_s += time.perf_counter() - reduction_started
    # --8<-- [end:communications-coverage]

    return CommunicationsAnalysis(
        mean_visible=visible_sum / times_s.size,
        minimum_visible=minimum_visible,
        maximum_visible=maximum_visible,
        materialized_count=materialized_count,
        geometry_elapsed_s=geometry_elapsed_s,
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
    """Render the time-averaged number of satellites in view."""
    import matplotlib.colors as colors

    if coordinates is None:
        coordinates = map_coordinates(resolution=HEALPIX_RESOLUTION)
    maximum = float(result.mean_visible.max())
    plot_global_map(
        result.mean_visible,
        output,
        coordinates=coordinates,
        visible=np.ones(result.mean_visible.size, dtype=bool),
        title="Starlink-like communications availability",
        subtitle=(
            "500 satellites · one-hour mean · "
            "25° minimum elevation · one-minute samples"
        ),
        colorbar_label="Mean satellites in view",
        footer=(
            f"Minimum sampled availability: {int(result.minimum_visible.min())}  ·  "
            f"Peak sampled availability: {int(result.maximum_visible.max())}  ·  "
            f"HEALPix resolution {HEALPIX_RESOLUTION}"
        ),
        cmap="plasma",
        norm=colors.PowerNorm(gamma=0.75, vmin=0.0, vmax=maximum),
        dpi=dpi,
    )


def render_documentation() -> str:
    """Run the scenario and return its live figure and measurements as HTML."""
    documentation_started = time.perf_counter()
    result = analyze()
    plotting_started = time.perf_counter()
    coordinates = map_coordinates(resolution=HEALPIX_RESOLUTION)
    image = BytesIO()
    plot_availability(result, image, coordinates=coordinates, dpi=80)
    plotting_elapsed_s = time.perf_counter() - plotting_started
    encoded_image = base64.b64encode(image.getvalue()).decode("ascii")
    documentation_elapsed_s = time.perf_counter() - documentation_started

    snapshot_count = DURATION_S // CADENCE_S + 1
    footprint_count = snapshot_count * SATELLITE_COUNT
    return f"""
<figure>
  <img
    src="data:image/png;base64,{encoded_image}"
    alt="Global map of mean communications satellites in view over one hour"
  >
  <figcaption>
    Mean simultaneous satellites above a 25° elevation mask, sampled once per
    minute for one hour.
  </figcaption>
</figure>

<table>
  <thead>
    <tr><th>Measurement</th><th>Result from this build</th></tr>
  </thead>
  <tbody>
    <tr><td>Orbit and footprint generation</td>
        <td>{result.geometry_elapsed_s * 1_000:.1f} ms</td></tr>
    <tr><td>Batched <code>cover_footprint()</code> calls</td>
        <td>{result.coverage_elapsed_s * 1_000:.1f} ms</td></tr>
    <tr><td>Availability reduction</td>
        <td>{result.reduction_elapsed_s * 1_000:.1f} ms</td></tr>
    <tr><td>Complete numerical analysis</td>
        <td>{result.analysis_elapsed_s * 1_000:.1f} ms</td></tr>
    <tr><td>Map and PNG encoding</td>
        <td>{plotting_elapsed_s * 1_000:.1f} ms</td></tr>
    <tr><td>Complete executable example</td>
        <td>{documentation_elapsed_s * 1_000:.1f} ms</td></tr>
    <tr><td>Service footprints covered</td><td>{footprint_count:,}</td></tr>
    <tr><td>Materialized footprint-cell observations</td>
        <td>{result.materialized_count:,}</td></tr>
    <tr><td>Mean satellites in view, global range</td>
        <td>{result.mean_visible.min():.2f}–{result.mean_visible.max():.2f}</td></tr>
    <tr><td>Minimum satellites in view at any sample</td>
        <td>{int(result.minimum_visible.min())}</td></tr>
    <tr><td>Maximum satellites in view at any sample</td>
        <td>{int(result.maximum_visible.max())}</td></tr>
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
        f"Covered {(DURATION_S // CADENCE_S + 1) * SATELLITE_COUNT:,} service "
        f"footprints in {result.coverage_elapsed_s:.3f} s."
    )
    print(f"Complete availability analysis took {result.analysis_elapsed_s:.3f} s.")
    print(f"Rendering the map took {plotting_elapsed_s:.3f} s.")
    print(f"Saved {args.output}.")


if __name__ == "__main__":
    main()
