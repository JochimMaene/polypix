"""Map ten days of observations and revisit time for an EO constellation."""

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
    constellation_centers,
    map_coordinates,
    plot_global_map,
    swath_edges,
)

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
        coverage = px.cover_strip(
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

    interval_count = int(coverages[0].counts.size)
    if any(coverage.counts.size != interval_count for coverage in coverages):
        raise ValueError("all satellite strips must have the same interval count")

    observations = np.zeros(cell_count, dtype=np.int64)
    revisit_gap_sum = np.zeros(cell_count, dtype=np.int64)
    revisit_counts = np.zeros(cell_count, dtype=np.int64)
    constellation_last_seen = np.full(cell_count, -2, dtype=np.int32)
    interval_stamp = np.full(cell_count, -1, dtype=np.int32)
    satellite_last_seen = np.full(
        (len(coverages), cell_count),
        -2,
        dtype=np.int32,
    )

    # Each segment already contains unique cells. Integer timestamps therefore
    # replace global sorting and per-interval set construction.
    # --8<-- [start:eo-reduce]
    for interval in range(interval_count):
        observed_parts: list[npt.NDArray[np.intp]] = []
        for satellite, coverage in enumerate(coverages):
            cells = coverage.cells[
                int(coverage.offsets[interval]) : int(coverage.offsets[interval + 1])
            ].astype(np.intp, copy=False)

            previous_satellite = satellite_last_seen[satellite, cells]
            starts = previous_satellite < interval - 1
            observations[cells[starts]] += 1
            satellite_last_seen[satellite, cells] = interval

            first_in_interval = interval_stamp[cells] != interval
            observed_parts.append(cells[first_in_interval])
            interval_stamp[cells] = interval

        observed = np.concatenate(observed_parts)
        previous = constellation_last_seen[observed]
        revisited = (previous >= 0) & (previous < interval - 1)
        revisited_cells = observed[revisited]
        revisit_gap_sum[revisited_cells] += interval - previous[revisited] - 1
        revisit_counts[revisited_cells] += 1
        constellation_last_seen[observed] = interval
    # --8<-- [end:eo-reduce]

    mean_revisit_s = np.full(cell_count, np.nan, dtype=np.float64)
    measured = revisit_counts > 0
    mean_revisit_s[measured] = (
        revisit_gap_sum[measured] * cadence_s / revisit_counts[measured]
    )
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
    plot_global_map(
        result.observations,
        output,
        coordinates=coordinates,
        visible=visible,
        title="Ten days of Earth-observation coverage",
        subtitle=(
            "10 satellites · one-minute swept intervals · 7.5° ground half-width"
        ),
        colorbar_label="Distinct satellite observations per cell",
        footer=(
            f"{np.count_nonzero(visible):,} of {result.observations.size:,} cells "
            f"observed  ·  maximum {int(result.observations.max())} observations"
        ),
        cmap="plasma",
        norm=colors.PowerNorm(
            gamma=0.65,
            vmin=1,
            vmax=int(result.observations.max()),
        ),
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
    plot_global_map(
        revisit_hours,
        output,
        coordinates=coordinates,
        visible=measured,
        title="Mean Earth-observation revisit time",
        subtitle=(
            "Gap between globally merged access windows · "
            "ten-day analysis · one-minute resolution"
        ),
        colorbar_label="Mean revisit gap (hours, logarithmic scale)",
        footer=(
            f"{np.count_nonzero(measured):,} cells with measured revisit  ·  "
            f"median {np.median(finite_hours):.2f} hours"
        ),
        cmap="viridis_r",
        norm=colors.LogNorm(
            vmin=float(finite_hours.min()),
            vmax=float(finite_hours.max()),
        ),
        dpi=dpi,
    )


def render_documentation() -> str:
    """Run the scenario and return two live figures and measurements as HTML."""
    documentation_started = time.perf_counter()
    result = analyze()
    plotting_started = time.perf_counter()
    coordinates = map_coordinates(resolution=HEALPIX_RESOLUTION)

    observations_image = BytesIO()
    plot_observations(result, observations_image, coordinates=coordinates, dpi=80)
    revisit_image = BytesIO()
    plot_revisit(result, revisit_image, coordinates=coordinates, dpi=80)
    plotting_elapsed_s = time.perf_counter() - plotting_started
    encoded_observations = base64.b64encode(observations_image.getvalue()).decode(
        "ascii"
    )
    encoded_revisit = base64.b64encode(revisit_image.getvalue()).decode("ascii")
    documentation_elapsed_s = time.perf_counter() - documentation_started

    measured = np.isfinite(result.mean_revisit_s)
    revisit_hours = result.mean_revisit_s[measured] / 3_600
    interval_count = DURATION_S // CADENCE_S * SATELLITE_COUNT
    return f"""
<figure>
  <img src="data:image/png;base64,{encoded_observations}"
       alt="Global map of distinct Earth observations over ten days">
  <figcaption>
    Consecutive one-minute hits by the same satellite are one observation.
  </figcaption>
</figure>

<figure>
  <img src="data:image/png;base64,{encoded_revisit}"
       alt="Global map of mean Earth-observation revisit time over ten days">
  <figcaption>
    Mean uncovered gap between globally merged constellation access windows.
  </figcaption>
</figure>

<table>
  <thead>
    <tr><th>Measurement</th><th>Result from this build</th></tr>
  </thead>
  <tbody>
    <tr><td>Orbit and swath generation</td>
        <td>{result.swath_elapsed_s * 1_000:.1f} ms</td></tr>
    <tr><td>Ten sequential <code>cover_strip()</code> calls</td>
        <td>{result.coverage_elapsed_s * 1_000:.1f} ms</td></tr>
    <tr><td>Observation and revisit reduction</td>
        <td>{result.reduction_elapsed_s * 1_000:.1f} ms</td></tr>
    <tr><td>Complete numerical analysis</td>
        <td>{result.analysis_elapsed_s * 1_000:.1f} ms</td></tr>
    <tr><td>Coordinates, two maps, and PNG encoding</td>
        <td>{plotting_elapsed_s * 1_000:.1f} ms</td></tr>
    <tr><td>Complete executable example</td>
        <td>{documentation_elapsed_s * 1_000:.1f} ms</td></tr>
    <tr><td>Swept intervals covered</td><td>{interval_count:,}</td></tr>
    <tr><td>Materialized interval-cell observations</td>
        <td>{result.materialized_count:,}</td></tr>
    <tr><td>Observed cells</td>
        <td>{np.count_nonzero(result.observations):,} of
            {result.observations.size:,}</td></tr>
    <tr><td>Distinct observations per cell, maximum</td>
        <td>{int(result.observations.max())}</td></tr>
    <tr><td>Median per-cell mean revisit</td>
        <td>{np.median(revisit_hours):.2f} hours</td></tr>
    <tr><td>Range of per-cell mean revisit</td>
        <td>{revisit_hours.min():.2f}–{revisit_hours.max():.2f} hours</td></tr>
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
