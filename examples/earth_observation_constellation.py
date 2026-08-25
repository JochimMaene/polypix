"""Map 14 days of Sentinel-2 overflight gaps from a pinned TLE snapshot."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import numpy.typing as npt

import polypix as px
from examples.constellation import (
    DOC_FIGURE_DIR,
    DOC_FIGURE_INCLUDE_DIR,
    EARTH_RADIUS_KM,
    map_coordinates,
    plot_global_map,
    swath_edges,
)
from examples.palette import COUNT_CMAP, GAP_CMAP, RAMP_START

VISITS_FIGURE_PATH = DOC_FIGURE_DIR / "earth-observation-count.png"
MEAN_GAP_FIGURE_PATH = DOC_FIGURE_DIR / "earth-observation-revisit.png"
WORST_GAP_FIGURE_PATH = DOC_FIGURE_DIR / "earth-observation-worst-gap.png"

# Permanent CelesTrak snapshot of the three Sentinel-2 spacecraft, fetched
# 2026-08-24 in three-line element format. The example never goes to the
# network, so the analysis is reproducible from the repository alone.
TLE_PATH = Path(__file__).with_name("data") / "sentinel-2-2026-08-24.tle"
ANALYSIS_START = datetime(2026, 8, 24, tzinfo=UTC)

# The Multi-Spectral Instrument images a 290 km ground swath.
SWATH_WIDTH_KM = 290.0
SWATH_HALF_WIDTH_RAD = 0.5 * SWATH_WIDTH_KM / EARTH_RADIUS_KM

DURATION_DAYS = 14
CADENCE_S = 60
HEALPIX_RESOLUTION = 7


@dataclass(frozen=True)
class EarthObservationAnalysis:
    """Per-cell overflight gaps, in hours, with stage timings."""

    visits: npt.NDArray[np.int64]
    mean_gap_h: npt.NDArray[np.float64]
    worst_gap_h: npt.NDArray[np.float64]
    satellite_count: int
    interval_count: int
    stored_hit_count: int
    observed_cell_count: int
    swath_elapsed_s: float
    coverage_elapsed_s: float
    reduction_elapsed_s: float
    analysis_elapsed_s: float


# --8<-- [start:eo-swaths]
def ground_swath(
    times_min: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the left and right swath edges of every spacecraft."""
    import astroz

    constellation = astroz.Constellation(str(TLE_PATH))
    positions_km = astroz.propagate(
        constellation,
        times_min,
        start_time=ANALYSIS_START,
        output="ecef",
    )
    sub_satellite = positions_km / np.linalg.norm(positions_km, axis=-1, keepdims=True)
    return swath_edges(sub_satellite, half_width_rad=SWATH_HALF_WIDTH_RAD)


# --8<-- [end:eo-swaths]


# --8<-- [start:eo-cover]
def cover_constellation(
    left_edges: npt.NDArray[np.float64],
    right_edges: npt.NDArray[np.float64],
) -> list[px.Coverage]:
    """Cover every one-minute interval of every spacecraft's swath."""
    return [
        px.cover_sweep(
            left_edges[:, satellite],
            right_edges[:, satellite],
            resolution=HEALPIX_RESOLUTION,
        )
        for satellite in range(left_edges.shape[1])
    ]


# --8<-- [end:eo-cover]


# --8<-- [start:eo-reduce]
def overflight_gaps(
    coverages: list[px.Coverage],
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Return visited cells, their visit counts, and their gaps in hours."""
    stats = px.revisit(coverages)
    gap_counts = stats.run_counts - 1
    measured = gap_counts > 0
    hours_per_bin = CADENCE_S / 3_600
    mean_gap_h = (
        stats.internal_gap_steps_sum[measured] / gap_counts[measured] * hours_per_bin
    )
    worst_gap_h = stats.maximum_internal_gap_steps[measured] * hours_per_bin
    return stats.cells, stats.run_counts, mean_gap_h, worst_gap_h


# --8<-- [end:eo-reduce]


def analyze() -> EarthObservationAnalysis:
    """Run the 14-day Sentinel-2 overflight analysis."""
    analysis_started = time.perf_counter()

    times_min = np.arange(
        0.0,
        DURATION_DAYS * 24 * 60 + CADENCE_S / 60,
        CADENCE_S / 60,
        dtype=np.float64,
    )
    swath_started = time.perf_counter()
    left_edges, right_edges = ground_swath(times_min)
    swath_elapsed_s = time.perf_counter() - swath_started

    coverage_started = time.perf_counter()
    coverages = cover_constellation(left_edges, right_edges)
    coverage_elapsed_s = time.perf_counter() - coverage_started

    reduction_started = time.perf_counter()
    cells, run_counts, measured_mean_h, measured_worst_h = overflight_gaps(coverages)
    cell_count = px.cell_count(HEALPIX_RESOLUTION)
    visits = np.zeros(cell_count, dtype=np.int64)
    visits[cells] = run_counts
    mean_gap_h = np.full(cell_count, np.nan, dtype=np.float64)
    worst_gap_h = np.full(cell_count, np.nan, dtype=np.float64)
    measured_cells = cells[run_counts > 1]
    mean_gap_h[measured_cells] = measured_mean_h
    worst_gap_h[measured_cells] = measured_worst_h
    reduction_elapsed_s = time.perf_counter() - reduction_started

    return EarthObservationAnalysis(
        visits=visits,
        mean_gap_h=mean_gap_h,
        worst_gap_h=worst_gap_h,
        satellite_count=len(coverages),
        interval_count=sum(len(coverage) for coverage in coverages),
        stored_hit_count=sum(int(coverage.cells.size) for coverage in coverages),
        observed_cell_count=int(cells.size),
        swath_elapsed_s=swath_elapsed_s,
        coverage_elapsed_s=coverage_elapsed_s,
        reduction_elapsed_s=reduction_elapsed_s,
        analysis_elapsed_s=time.perf_counter() - analysis_started,
    )


def _plot_gap_map(
    values_h: npt.NDArray[np.float64],
    output: Path | BytesIO,
    *,
    coordinates: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    levels: list[float],
    colorbar_label: str,
    dpi: int,
) -> None:
    """Render one banded gap map over the cells that have a gap at all."""
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    ramp = plt.get_cmap(GAP_CMAP)
    steps = np.linspace(RAMP_START, 1.0, len(levels))
    plot_global_map(
        values_h,
        output,
        coordinates=coordinates,
        visible=np.isfinite(values_h),
        resolution=HEALPIX_RESOLUTION,
        colorbar_label=colorbar_label,
        cmap=colors.ListedColormap([ramp(step) for step in steps]),
        norm=colors.BoundaryNorm(levels, len(levels), extend="max"),
        colorbar_ticks=levels,
        extend="max",
        dpi=dpi,
    )


def plot_mean_gap(
    result: EarthObservationAnalysis,
    output: Path | BytesIO,
    *,
    coordinates: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    dpi: int = 150,
) -> None:
    """Render the mean time between overflights."""
    # Banding by hours reads like a contour map and can be taken straight off
    # the colorbar; a continuous ramp spends its range on differences that
    # nobody can see.
    _plot_gap_map(
        result.mean_gap_h,
        output,
        coordinates=coordinates,
        levels=[8, 16, 22, 28, 32, 36, 40],
        colorbar_label="Mean time between overflights (hours)",
        dpi=dpi,
    )


def plot_worst_gap(
    result: EarthObservationAnalysis,
    output: Path | BytesIO,
    *,
    coordinates: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    dpi: int = 150,
) -> None:
    """Render the longest single wait between overflights."""
    _plot_gap_map(
        result.worst_gap_h,
        output,
        coordinates=coordinates,
        levels=[24, 36, 48, 60, 72, 84, 96],
        colorbar_label="Longest wait between overflights (hours)",
        dpi=dpi,
    )


def plot_visits(
    result: EarthObservationAnalysis,
    output: Path | BytesIO,
    *,
    coordinates: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    dpi: int = 150,
) -> None:
    """Render how many separate overflights each cell received."""
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    levels = [12, 18, 24, 30, 40, 60, 100]
    ramp = plt.get_cmap(COUNT_CMAP)
    steps = np.linspace(RAMP_START, 1.0, len(levels))
    plot_global_map(
        result.visits,
        output,
        coordinates=coordinates,
        visible=result.visits > 0,
        resolution=HEALPIX_RESOLUTION,
        colorbar_label="Overflights per cell",
        cmap=colors.ListedColormap([ramp(step) for step in steps]),
        norm=colors.BoundaryNorm(levels, len(levels), extend="max"),
        colorbar_ticks=levels,
        extend="max",
        dpi=dpi,
    )


def build_documentation_assets() -> str:
    """Run the scenario, write the maps, and return its documentation page."""
    result = analyze()
    coordinates = map_coordinates(resolution=HEALPIX_RESOLUTION)
    plot_mean_gap(result, MEAN_GAP_FIGURE_PATH, coordinates=coordinates, dpi=170)
    plot_worst_gap(result, WORST_GAP_FIGURE_PATH, coordinates=coordinates, dpi=170)
    plot_visits(result, VISITS_FIGURE_PATH, coordinates=coordinates, dpi=170)

    never_observed = result.visits.size - result.observed_cell_count
    return f"""
```{{figure}} {DOC_FIGURE_INCLUDE_DIR}/{MEAN_GAP_FIGURE_PATH.name}
:alt: Global map of the mean time between Sentinel-2 overflights
:figclass: example-figure

Mean time between overflights over {DURATION_DAYS} days. Consecutive
one-minute intervals covered by any of the three spacecraft count as one
overflight.
```

```{{figure}} {DOC_FIGURE_INCLUDE_DIR}/{WORST_GAP_FIGURE_PATH.name}
:alt: Global map of the longest wait between Sentinel-2 overflights
:figclass: example-figure

The longest single wait in the same {DURATION_DAYS} days. Waits running past
the start or the end of the window are excluded, and {never_observed:,} cells
above 83° latitude are never overflown at all.
```

```{{list-table}} One measured run
:header-rows: 1
:class: example-timings
:widths: 70 30

* - Stage
  - Time
* - SGP4 propagation and swath edges
  - {result.swath_elapsed_s * 1_000:.0f} ms
* - {result.satellite_count} `cover_sweep()` calls
  - **{result.coverage_elapsed_s * 1_000:.0f} ms**
* - `revisit()` and gap conversion
  - {result.reduction_elapsed_s * 1_000:.0f} ms
* - Complete analysis
  - {result.analysis_elapsed_s * 1_000:.0f} ms
```
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mean-gap-output",
        type=Path,
        default=Path("earth-observation-revisit.png"),
    )
    parser.add_argument(
        "--worst-gap-output",
        type=Path,
        default=Path("earth-observation-worst-gap.png"),
    )
    args = parser.parse_args()

    result = analyze()
    coordinates = map_coordinates(resolution=HEALPIX_RESOLUTION)
    plot_mean_gap(result, args.mean_gap_output, coordinates=coordinates)
    plot_worst_gap(result, args.worst_gap_output, coordinates=coordinates)
    measured = np.isfinite(result.mean_gap_h)
    print(
        f"Covered {result.interval_count:,} swath intervals into "
        f"{result.stored_hit_count:,} interval-cell hits in "
        f"{result.coverage_elapsed_s:.3f} s."
    )
    print(
        "Mean time between overflights: "
        f"{np.nanmin(result.mean_gap_h):.1f} h at best, "
        f"{np.nanmax(result.mean_gap_h):.1f} h at worst, "
        f"over {int(measured.sum()):,} cells."
    )
    print(f"Complete analysis took {result.analysis_elapsed_s:.3f} s.")
    print(f"Saved {args.mean_gap_output} and {args.worst_gap_output}.")


if __name__ == "__main__":
    main()
