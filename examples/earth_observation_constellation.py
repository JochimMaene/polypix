"""Map ten days of occupied-bin runs and internal gaps for an EO constellation."""

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
    constellation_centers,
    map_coordinates,
    plot_global_map,
    swath_edges,
)
from examples.palette import COUNT_CMAP, GAP_CMAP

OBSERVATIONS_FIGURE_PATH = DOC_FIGURE_DIR / "earth-observation-count.png"
REVISIT_FIGURE_PATH = DOC_FIGURE_DIR / "earth-observation-revisit.png"

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
    """Sampled occupied-bin and internal-gap results with stage timings."""

    observations: npt.NDArray[np.int64]
    mean_internal_gap_s: npt.NDArray[np.float64]
    max_internal_gap_s: npt.NDArray[np.float64]
    gap_counts: npt.NDArray[np.int64]
    stored_hit_count: int
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
    stored_hit_count = 0
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
        stored_hit_count += int(coverage.cells.size)
    # --8<-- [end:eo-cover]
    return coverages, stored_hit_count, elapsed_s


def reduce_coverage(
    coverages: list[px.Coverage],
    *,
    cell_count: int,
    cadence_s: int,
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
]:
    """Count merged occupied-bin runs and their complete internal gaps."""
    if not coverages:
        empty_int = np.zeros(cell_count, dtype=np.int64)
        return (
            empty_int,
            np.full(cell_count, np.nan, dtype=np.float64),
            np.full(cell_count, np.nan, dtype=np.float64),
            empty_int.copy(),
        )
    expected_cell_count = px.cell_count(coverages[0].resolution)
    if cell_count != expected_cell_count:
        raise ValueError("cell_count must match the coverage resolution")
    interval_count = coverages[0].offsets.size - 1
    if any(coverage.offsets.size - 1 != interval_count for coverage in coverages):
        raise ValueError("all satellite strips must have the same interval count")

    # Runs are only an intermediate here, so fuse the per-cell statistics and
    # never build run boundaries. Physical time and the choice of which
    # gaps to summarize remain downstream of the ordinal occupancy operation.
    # --8<-- [start:eo-reduce]
    stats = px.revisit(coverages)
    observations = np.zeros(cell_count, dtype=np.int64)
    gap_counts = np.zeros(cell_count, dtype=np.int64)
    mean_internal_gap_s = np.full(cell_count, np.nan, dtype=np.float64)
    max_internal_gap_s = np.full(cell_count, np.nan, dtype=np.float64)
    observations[stats.cells] = stats.run_counts
    observed_gap_counts = stats.run_counts - 1
    gap_counts[stats.cells] = observed_gap_counts
    measured = observed_gap_counts > 0
    mean_internal_gap_s[stats.cells[measured]] = (
        stats.internal_gap_steps_sum[measured]
        / observed_gap_counts[measured]
        * cadence_s
    )
    max_internal_gap_s[stats.cells[measured]] = (
        stats.maximum_internal_gap_steps[measured] * cadence_s
    )
    # --8<-- [end:eo-reduce]
    return observations, mean_internal_gap_s, max_internal_gap_s, gap_counts


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

    coverages, stored_hit_count, coverage_elapsed_s = cover_constellation(
        left_edges,
        right_edges,
    )
    reduction_started = time.perf_counter()
    observations, mean_internal_gap_s, max_internal_gap_s, gap_counts = reduce_coverage(
        coverages,
        cell_count=px.cell_count(HEALPIX_RESOLUTION),
        cadence_s=CADENCE_S,
    )
    reduction_elapsed_s = time.perf_counter() - reduction_started
    return EarthObservationAnalysis(
        observations=observations,
        mean_internal_gap_s=mean_internal_gap_s,
        max_internal_gap_s=max_internal_gap_s,
        gap_counts=gap_counts,
        stored_hit_count=stored_hit_count,
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
    """Render merged constellation occupied-bin runs per cell."""
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    visible = result.observations > 0
    # Same reason as the revisit map: the bulk of the sphere sits in a narrow
    # band of counts while a thin peak reaches far higher, so bands show the
    # latitude structure that a continuous ramp flattens.
    levels = [1, 50, 100, 150, 200, 250, 350, 450]
    plot_global_map(
        result.observations,
        output,
        coordinates=coordinates,
        visible=visible,
        resolution=HEALPIX_RESOLUTION,
        colorbar_label="Merged occupied-bin runs per cell",
        cmap=plt.get_cmap(COUNT_CMAP, len(levels)),
        norm=colors.BoundaryNorm(levels, len(levels), extend="max"),
        colorbar_ticks=levels,
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
    """Render mean internal gaps between sampled occupied-bin runs."""
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    measured = np.isfinite(result.mean_internal_gap_s)
    gap_hours = result.mean_internal_gap_s / 3_600
    # Most measured cells sit close to the median, so a continuous ramp spends
    # its range on differences nobody can see. Banding by hour reads like a
    # contour map and can be taken straight off the colorbar.
    levels = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    plot_global_map(
        gap_hours,
        output,
        coordinates=coordinates,
        visible=measured,
        resolution=HEALPIX_RESOLUTION,
        colorbar_label="Mean internal uncovered gap (hours)",
        cmap=plt.get_cmap(GAP_CMAP, len(levels)),
        norm=colors.BoundaryNorm(levels, len(levels), extend="max"),
        colorbar_ticks=levels,
        extend="max",
        dpi=dpi,
    )


def build_documentation_assets() -> str:
    """Run the scenario, write both maps, and return its documentation HTML."""
    result = analyze()
    plotting_started = time.perf_counter()
    coordinates = map_coordinates(resolution=HEALPIX_RESOLUTION)
    plot_observations(
        result, OBSERVATIONS_FIGURE_PATH, coordinates=coordinates, dpi=100
    )
    plot_revisit(result, REVISIT_FIGURE_PATH, coordinates=coordinates, dpi=100)
    plotting_elapsed_s = time.perf_counter() - plotting_started

    cells_observed = int(np.count_nonzero(result.observations))
    return f"""
<figure class="example-figure">
  <img src="{DOC_FIGURE_URL}/{OBSERVATIONS_FIGURE_PATH.name}"
       alt="Global map of merged Earth-observation occupied-bin runs over ten days"
       loading="lazy">
  <figcaption>
    Consecutive one-minute hits by any satellite form one merged occupied-bin run.
  </figcaption>
</figure>

<figure class="example-figure">
  <img src="{DOC_FIGURE_URL}/{REVISIT_FIGURE_PATH.name}"
       alt="Observed-cell map of mean internal uncovered gaps over ten days"
       loading="lazy">
  <figcaption>
    Mean internal gap between sampled occupied-bin runs. Horizon-edge gaps are
    excluded, and cells with fewer than two runs have no value.
  </figcaption>
</figure>

<div class="example-metrics">
  <div><strong>{DURATION_S // CADENCE_S * SATELLITE_COUNT:,}</strong><span>swept intervals</span></div>
  <div><strong>{result.stored_hit_count:,}</strong><span>interval–cell hits</span></div>
  <div><strong>{cells_observed:,}</strong><span>cells observed</span></div>
  <div><strong>{result.observations.size - cells_observed:,}</strong><span>cells never observed</span></div>
</div>

<table class="example-timings">
  <caption>One measured run</caption>
  <thead>
    <tr><th>Stage</th><th>Time</th></tr>
  </thead>
  <tbody>
    <tr><td>Ten <code>cover_sweep()</code> calls</td>
        <td><strong>{result.coverage_elapsed_s * 1_000:.0f} ms</strong></td></tr>
    <tr><td>Orbits and swath edges</td>
        <td>{result.swath_elapsed_s * 1_000:.0f} ms</td></tr>
    <tr><td>Occupancy runs and scatter</td>
        <td>{result.reduction_elapsed_s * 1_000:.0f} ms</td></tr>
    <tr><td>Complete analysis</td>
        <td>{result.analysis_elapsed_s * 1_000:.0f} ms</td></tr>
    <tr><td>Two plots and PNG encoding</td>
        <td>{plotting_elapsed_s * 1_000:.0f} ms</td></tr>
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
        f"Complete sampled-occupancy and internal-gap analysis took "
        f"{result.analysis_elapsed_s:.3f} s."
    )
    print(f"Rendering both maps took {plotting_elapsed_s:.3f} s.")
    print(f"Saved {args.observations_output} and {args.revisit_output}.")


if __name__ == "__main__":
    main()
