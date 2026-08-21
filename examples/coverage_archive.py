"""Persist swept coverage, reload it, and analyse it in a later process.

Coverage is the expensive product of a campaign: covering six satellites over
two days here costs far more than every query afterwards. So the realistic shape
of the work is to compute it once, store it, and answer questions against the
archive for as long as the campaign stays interesting.

That round trip is what :meth:`polypix.Coverage.from_arrays` exists for. A
`Coverage` is two flat arrays plus a resolution, so it stores in any container
that holds arrays -- ``.npz`` here -- and `from_arrays()` is the way back, one
validation at the boundary rather than a check on every later hit.

Reloaded coverages are ordinary coverages: this runs :func:`polypix.revisit`
over them and asserts the statistics match the in-process originals exactly.

It then asks a regional question with ``reduce(..., cells=)``, which answers
about a named set of cells without ever allocating the grid. That matters as
resolution rises: the array-level equivalent, ``np.bincount(cells,
minlength=cell_count)[region]``, has to build the whole grid to index a few
thousand cells out of it, and at resolution 13 that is six gibibytes to answer a
question about a city.
"""

from __future__ import annotations

import argparse
import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

import polypix as px
from examples.constellation import constellation_centers, swath_edges

SATELLITE_COUNT = 6
PLANE_COUNT = 3
ALTITUDE_KM = 550.0
INCLINATION_RAD = math.radians(53.0)
SWATH_HALF_WIDTH_RAD = math.radians(7.5)

DURATION_S = 2 * 24 * 60 * 60
CADENCE_S = 120
HEALPIX_RESOLUTION = 7

# A region of interest, as a cap over central Europe.
REGION_CENTER_LONLAT = (10.0, 50.0)
REGION_RADIUS_RAD = math.radians(5.0)


@dataclass(frozen=True)
class ArchiveAnalysis:
    """What the archive round trip produced, with stage timings."""

    satellite_count: int
    stored_hit_count: int
    archive_bytes: int
    region_cell_count: int
    region_observations: npt.NDArray[np.int64]
    cover_elapsed_s: float
    store_elapsed_s: float
    load_elapsed_s: float
    revisit_elapsed_s: float


def cover_campaign() -> list[px.Coverage]:
    """Cover each satellite's swept ground track for the whole campaign."""
    times_s = np.arange(0.0, float(DURATION_S), float(CADENCE_S))
    centers = constellation_centers(
        times_s,
        satellite_count=SATELLITE_COUNT,
        plane_count=PLANE_COUNT,
        altitude_km=ALTITUDE_KM,
        inclination_rad=INCLINATION_RAD,
    )
    left, right = swath_edges(centers, half_width_rad=SWATH_HALF_WIDTH_RAD)
    return [
        px.cover_sweep(
            left[:, satellite], right[:, satellite], resolution=HEALPIX_RESOLUTION
        )
        for satellite in range(SATELLITE_COUNT)
    ]


def store(coverages: list[px.Coverage], path: Path) -> None:
    """Write coverages to a single compressed archive.

    A `Coverage` is two flat arrays and a resolution, so nothing here is
    Polypix-specific: the same three fields go into Parquet, HDF5, or a blob
    store just as readily.
    """
    # --8<-- [start:archive-store]
    arrays: dict[str, npt.NDArray[np.int64]] = {}
    for satellite, coverage in enumerate(coverages):
        arrays[f"cells_{satellite}"] = coverage.cells
        arrays[f"offsets_{satellite}"] = coverage.offsets
    np.savez_compressed(path, resolution=HEALPIX_RESOLUTION, **arrays)
    # --8<-- [end:archive-store]


def load(path: Path) -> list[px.Coverage]:
    """Rebuild coverages from the archive, validating once on the way in."""
    # --8<-- [start:archive-load]
    with np.load(path) as archive:
        resolution = int(archive["resolution"])
        satellite_count = sum(1 for name in archive.files if name.startswith("cells_"))
        return [
            px.Coverage.from_arrays(
                archive[f"cells_{satellite}"],
                archive[f"offsets_{satellite}"],
                resolution=resolution,
            )
            for satellite in range(satellite_count)
        ]
    # --8<-- [end:archive-load]


def region_cells(resolution: int) -> npt.NDArray[np.uint64]:
    """RING indices of the region of interest."""
    longitude, latitude = np.radians(REGION_CENTER_LONLAT)
    center = np.array(
        [
            math.cos(latitude) * math.cos(longitude),
            math.cos(latitude) * math.sin(longitude),
            math.sin(latitude),
        ]
    )
    return np.asarray(px.cover_cap(center, REGION_RADIUS_RAD, resolution).cells)


def analyze() -> ArchiveAnalysis:
    """Cover, store, reload, and verify that the reloaded coverage still answers."""
    started = time.perf_counter()
    coverages = cover_campaign()
    cover_elapsed_s = time.perf_counter() - started
    stored_hit_count = sum(int(coverage.cells.size) for coverage in coverages)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "campaign.npz"
        started = time.perf_counter()
        store(coverages, path)
        store_elapsed_s = time.perf_counter() - started
        archive_bytes = path.stat().st_size

        started = time.perf_counter()
        reloaded = load(path)
        load_elapsed_s = time.perf_counter() - started

    # The whole point of the round trip: a reloaded coverage is an ordinary
    # coverage, so every operation still applies and agrees exactly.
    started = time.perf_counter()
    stats = px.revisit(reloaded)
    revisit_elapsed_s = time.perf_counter() - started
    original = px.revisit(coverages)
    np.testing.assert_array_equal(stats.cells, original.cells)
    np.testing.assert_array_equal(stats.run_counts, original.run_counts)
    np.testing.assert_array_equal(
        stats.maximum_internal_gap_steps, original.maximum_internal_gap_steps
    )

    # --8<-- [start:archive-region]
    region = region_cells(HEALPIX_RESOLUTION)
    region_observations = np.zeros(region.size, dtype=np.int64)
    for coverage in reloaded:
        region_observations += coverage.reduce(px.Count(), cells=region)
    # --8<-- [end:archive-region]

    return ArchiveAnalysis(
        satellite_count=len(reloaded),
        stored_hit_count=stored_hit_count,
        archive_bytes=archive_bytes,
        region_cell_count=int(region.size),
        region_observations=region_observations,
        cover_elapsed_s=cover_elapsed_s,
        store_elapsed_s=store_elapsed_s,
        load_elapsed_s=load_elapsed_s,
        revisit_elapsed_s=revisit_elapsed_s,
    )


def main() -> None:
    """Report the round trip and the regional query."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    result = analyze()
    grid = px.cell_count(HEALPIX_RESOLUTION)
    print(
        f"Covered {result.satellite_count} satellites over "
        f"{DURATION_S // 86400} days at resolution {HEALPIX_RESOLUTION} "
        f"({result.stored_hit_count:,} hits) in {result.cover_elapsed_s:.2f} s."
    )
    print(
        f"Stored {result.archive_bytes / 2**20:.1f} MiB in "
        f"{result.store_elapsed_s:.2f} s, reloaded in {result.load_elapsed_s:.2f} s."
    )
    print(
        f"revisit() over the reloaded coverages took "
        f"{result.revisit_elapsed_s:.2f} s and matched the originals exactly."
    )
    print(
        f"Region of interest: {result.region_cell_count} cells, "
        f"{result.region_observations.sum():,} observations, "
        f"peak {result.region_observations.max():,} on one cell -- answered "
        f"without allocating any of the {grid:,}-cell grid."
    )


if __name__ == "__main__":
    main()
