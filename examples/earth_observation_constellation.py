"""Map ten days of observations and revisit time for an EO constellation."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import numpy.typing as npt

import polypix as px
from examples.constellation import (
    DOC_FIGURE_DIR,
    DOC_FIGURE_URL,
    clipped_range,
    constellation_centers,
    map_coordinates,
    plot_global_map,
    read_measurements,
    swath_edges,
    write_measurements,
)

OBSERVATIONS_FIGURE_PATH = DOC_FIGURE_DIR / "earth-observation-count.png"
REVISIT_FIGURE_PATH = DOC_FIGURE_DIR / "earth-observation-revisit.png"
MEASUREMENTS_PATH = DOC_FIGURE_DIR / "earth-observation.json"

SATELLITE_COUNT = 10
PLANE_COUNT = 5
ALTITUDE_KM = 550.0
INCLINATION_RAD = math.radians(53.0)
SWATH_HALF_WIDTH_RAD = math.radians(7.5)

DURATION_S = 10 * 24 * 60 * 60
CADENCE_S = 60
HEALPIX_RESOLUTION = 6


@dataclass(frozen=True)
class EarthObservationAnalysis:
    """Observation and revisit results with stage timings."""

    observations: npt.NDArray[np.int64]
    mean_revisit_s: npt.NDArray[np.float64]
    revisit_counts: npt.NDArray[np.int64]
    materialized_count: int
    swath_elapsed_s: float
    coverage_elapsed_s: float
    reduction_elapsed_s: float
    analysis_elapsed_s: float


def cover_constellation(
    left_edges: npt.NDArray[np.float64],
    right_edges: npt.NDArray[np.float64],
) -> tuple[list[px.Coverage], int, float]:
    """Cover each satellite's ten-day swept strip."""
    coverages: list[px.Coverage] = []
    materialized_count = 0
    elapsed_s = 0.0

    # --8<-- [start:eo-cover]
    for satellite in range(SATELLITE_COUNT):
        started = time.perf_counter()
        coverage = px.cover_sweep(
            left_edges[:, satellite],
            right_edges[:, satellite],
            resolution=HEALPIX_RESOLUTION,
        )
        elapsed_s += time.perf_counter() - started
        coverages.append(coverage)
        materialized_count += int(coverage.cells.size)
    # --8<-- [end:eo-cover]
    return coverages, materialized_count, elapsed_s


def reduce_coverage(
    coverages: list[px.Coverage],
    *,
    cell_count: int,
    cadence_s: int,
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
]:
    """Count observations and mean gaps between constellation accesses."""
    if not coverages:
        empty_int = np.zeros(cell_count, dtype=np.int64)
        return (
            empty_int,
            np.full(cell_count, np.nan, dtype=np.float64),
            empty_int.copy(),
        )
    expected_cell_count = 12 * 4 ** coverages[0].resolution
    if cell_count != expected_cell_count:
        raise ValueError("cell_count must match the coverage resolution")
    interval_count = coverages[0].offsets.size - 1
    if any(coverage.offsets.size - 1 != interval_count for coverage in coverages):
        raise ValueError("all satellite strips must have the same interval count")

    # Each source's observation runs and the constellation-wide merged gaps
    # are reduced natively without sorting or Python work per interval.
    # --8<-- [start:eo-reduce]
    summary = px.summarize_occupancy(coverages)
    observations = np.zeros(cell_count, dtype=np.int64)
    revisit_counts = np.zeros(cell_count, dtype=np.int64)
    mean_revisit_s = np.full(cell_count, np.nan, dtype=np.float64)
    cells = summary.cells.astype(np.int64, copy=False)
    observations[cells] = summary.run_counts
    revisit_counts[cells] = summary.merged_gap_counts
    mean_revisit_s[cells] = summary.mean_merged_gap_steps * cadence_s
    # --8<-- [end:eo-reduce]
    return observations, mean_revisit_s, revisit_counts


def analyze() -> EarthObservationAnalysis:
    """Run the ten-day swept-swath analysis."""
    analysis_started = time.perf_counter()
    swath_started = time.perf_counter()

    # --8<-- [start:eo-swaths]
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
    left_edges, right_edges = swath_edges(
        centers,
        half_width_rad=SWATH_HALF_WIDTH_RAD,
    )
    # --8<-- [end:eo-swaths]
    swath_elapsed_s = time.perf_counter() - swath_started

    coverages, materialized_count, coverage_elapsed_s = cover_constellation(
        left_edges,
        right_edges,
    )
    reduction_started = time.perf_counter()
    observations, mean_revisit_s, revisit_counts = reduce_coverage(
        coverages,
        cell_count=12 * 4**HEALPIX_RESOLUTION,
        cadence_s=CADENCE_S,
    )
    reduction_elapsed_s = time.perf_counter() - reduction_started
    return EarthObservationAnalysis(
        observations=observations,
        mean_revisit_s=mean_revisit_s,
        revisit_counts=revisit_counts,
        materialized_count=materialized_count,
        swath_elapsed_s=swath_elapsed_s,
        coverage_elapsed_s=coverage_elapsed_s,
        reduction_elapsed_s=reduction_elapsed_s,
        analysis_elapsed_s=time.perf_counter() - analysis_started,
    )


def plot_observations(
    result: EarthObservationAnalysis,
    output: Path | BytesIO,
    *,
    coordinates: tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ],
    dpi: int = 150,
) -> None:
    """Render distinct satellite observations per cell."""
    import matplotlib.colors as colors

    visible = result.observations > 0
    _, high = clipped_range(
        result.observations[visible].astype(np.float64),
        low_percentile=0.0,
        high_percentile=99.0,
    )
    plot_global_map(
        result.observations,
        output,
        coordinates=coordinates,
        visible=visible,
        resolution=HEALPIX_RESOLUTION,
        colorbar_label="Distinct satellite observations per cell",
        cmap="plasma",
        norm=colors.PowerNorm(gamma=0.65, vmin=1, vmax=high),
        extend="max",
        dpi=dpi,
    )


def plot_revisit(
    result: EarthObservationAnalysis,
    output: Path | BytesIO,
    *,
    coordinates: tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ],
    dpi: int = 150,
) -> None:
    """Render mean gaps between constellation access windows."""
    import matplotlib.colors as colors

    measured = np.isfinite(result.mean_revisit_s)
    revisit_hours = result.mean_revisit_s / 3_600
    finite_hours = revisit_hours[measured]
    # A thin polar tail reaches several hours and would otherwise flatten every
    # inhabited latitude into one color.
    low, high = clipped_range(finite_hours, low_percentile=1.0, high_percentile=99.0)
    plot_global_map(
        revisit_hours,
        output,
        coordinates=coordinates,
        visible=measured,
        resolution=HEALPIX_RESOLUTION,
        colorbar_label="Mean revisit gap (hours)",
        cmap="viridis_r",
        norm=colors.LogNorm(vmin=low, vmax=high),
        colorbar_ticks=[t for t in (0.5, 0.75, 1, 1.5, 2, 3, 5, 8) if low <= t <= high],
        extend="both",
        dpi=dpi,
    )


def build_documentation_assets() -> None:
    """Run the scenario, write both maps, and record the measurements."""
    result = analyze()
    plotting_started = time.perf_counter()
    coordinates = map_coordinates(resolution=HEALPIX_RESOLUTION)
    plot_observations(
        result, OBSERVATIONS_FIGURE_PATH, coordinates=coordinates, dpi=100
    )
    plot_revisit(result, REVISIT_FIGURE_PATH, coordinates=coordinates, dpi=100)
    plotting_elapsed_s = time.perf_counter() - plotting_started

    measured = np.isfinite(result.mean_revisit_s)
    revisit_hours = result.mean_revisit_s[measured] / 3_600
    write_measurements(
        MEASUREMENTS_PATH,
        {
            "interval_count": DURATION_S // CADENCE_S * SATELLITE_COUNT,
            "materialized_count": result.materialized_count,
            "swath_ms": result.swath_elapsed_s * 1_000,
            "coverage_ms": result.coverage_elapsed_s * 1_000,
            "reduction_ms": result.reduction_elapsed_s * 1_000,
            "analysis_ms": result.analysis_elapsed_s * 1_000,
            "plotting_ms": plotting_elapsed_s * 1_000,
            "cells_observed": int(np.count_nonzero(result.observations)),
            "cell_count": int(result.observations.size),
            "observations_max": int(result.observations.max()),
            "revisit_median_h": float(np.median(revisit_hours)),
            "revisit_min_h": float(revisit_hours.min()),
            "revisit_max_h": float(revisit_hours.max()),
        },
    )


def documentation_html() -> str:
    """Return the recorded figures and measurements as HTML for the docs page."""
    m = read_measurements(MEASUREMENTS_PATH)
    return f"""
<figure class="example-figure">
  <img src="{DOC_FIGURE_URL}/{OBSERVATIONS_FIGURE_PATH.name}"
       alt="Global map of distinct Earth observations over ten days"
       loading="lazy">
  <figcaption>
    Consecutive one-minute hits by the same satellite are one observation.
  </figcaption>
</figure>

<figure class="example-figure">
  <img src="{DOC_FIGURE_URL}/{REVISIT_FIGURE_PATH.name}"
       alt="Global map of mean Earth-observation revisit time over ten days"
       loading="lazy">
  <figcaption>
    Mean uncovered gap between globally merged constellation access windows.
  </figcaption>
</figure>

<div class="example-metrics">
  <div><strong>{m["interval_count"]:,}</strong><span>swept intervals</span></div>
  <div><strong>{m["materialized_count"]:,}</strong><span>interval–cell hits</span></div>
  <div><strong>{m["cells_observed"]:,}</strong><span>cells observed</span></div>
</div>

<table class="example-timings">
  <caption>One measured run</caption>
  <thead>
    <tr><th>Stage</th><th>Time</th></tr>
  </thead>
  <tbody>
    <tr><td>Ten <code>cover_sweep()</code> calls</td>
        <td><strong>{m["coverage_ms"]:.0f} ms</strong></td></tr>
    <tr><td>Orbits and swath edges</td>
        <td>{m["swath_ms"]:.0f} ms</td></tr>
    <tr><td>Occupancy summary and scatter</td>
        <td>{m["reduction_ms"]:.0f} ms</td></tr>
    <tr><td>Complete analysis</td>
        <td>{m["analysis_ms"]:.0f} ms</td></tr>
    <tr><td>Two plots and PNG encoding</td>
        <td>{m["plotting_ms"]:.0f} ms</td></tr>
  </tbody>
</table>
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observations-output",
        type=Path,
        default=Path("earth-observation-count.png"),
    )
    parser.add_argument(
        "--revisit-output",
        type=Path,
        default=Path("earth-observation-revisit.png"),
    )
    args = parser.parse_args()

    result = analyze()
    coordinates = map_coordinates(resolution=HEALPIX_RESOLUTION)
    plotting_started = time.perf_counter()
    plot_observations(result, args.observations_output, coordinates=coordinates)
    plot_revisit(result, args.revisit_output, coordinates=coordinates)
    plotting_elapsed_s = time.perf_counter() - plotting_started
    print(
        f"Covered {DURATION_S // CADENCE_S * SATELLITE_COUNT:,} swept intervals "
        f"in {result.coverage_elapsed_s:.3f} s."
    )
    print(
        f"Complete observation and revisit analysis took "
        f"{result.analysis_elapsed_s:.3f} s."
    )
    print(f"Rendering both maps took {plotting_elapsed_s:.3f} s.")
    print(f"Saved {args.observations_output} and {args.revisit_output}.")


if __name__ == "__main__":
    main()
