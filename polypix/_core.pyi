import numpy as np
import numpy.typing as npt

__version__: str
_MAX_RESOLUTION: int
__all__ = [
    "__version__",
    "_MAX_RESOLUTION",
    "_corner_many",
    "_cell_at",
    "_center",
    "_neighbors",
    "_count_caps_per_cell",
    "_count_coverage_per_cell",
    "_cover",
    "_cover_prepared_regions",
    "_cover_cap",
    "_cover_sweep",
    "_revisit_stats",
    "_sum_coverage_per_cell",
    "_validate_coverage",
    "_prepare_polygon",
]

def _cover(
    vertices_xyz: npt.NDArray[np.float64],
    offsets: npt.NDArray[np.uint64],
    resolution: int,
    candidate_cells: npt.NDArray[np.uint64] | None = None,
    restrict_output: bool = True,
    threads: int | None = None,
    overlap: bool = False,
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]: ...
def _cover_prepared_regions(
    regions: list[list[object]],
    resolution: int,
    candidate_cells: npt.NDArray[np.uint64] | None = None,
    restrict_output: bool = True,
    threads: int | None = None,
    overlap: bool = False,
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]: ...
def _cover_cap(
    centers_xyz: npt.NDArray[np.float64],
    radii_rad: npt.NDArray[np.float64],
    resolution: int,
    candidate_cells: npt.NDArray[np.uint64] | None = None,
    restrict_output: bool = True,
    threads: int | None = None,
    overlap: bool = False,
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]: ...
def _count_caps_per_cell(
    centers_xyz: npt.NDArray[np.float64],
    radii_rad: npt.NDArray[np.float64],
    resolution: int,
    cells: npt.NDArray[np.uint64] | None = None,
    threads: int | None = None,
) -> npt.NDArray[np.int64] | None: ...
def _count_coverage_per_cell(
    cells: npt.NDArray[np.uint64],
    resolution: int,
    requested_cells: npt.NDArray[np.uint64] | None = None,
) -> npt.NDArray[np.int64]: ...
def _sum_coverage_per_cell(
    cells: npt.NDArray[np.uint64],
    offsets: npt.NDArray[np.uint64],
    values: npt.NDArray[np.float64],
    resolution: int,
    requested_cells: npt.NDArray[np.uint64] | None = None,
) -> npt.NDArray[np.float64]: ...
def _cover_sweep(
    left_edge_xyz: npt.NDArray[np.float64],
    right_edge_xyz: npt.NDArray[np.float64],
    resolution: int,
    candidate_cells: npt.NDArray[np.uint64] | None = None,
    restrict_output: bool = True,
    threads: int | None = None,
    overlap: bool = False,
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]: ...
def _revisit_stats(
    cell_arrays: list[npt.NDArray[np.uint64]],
    offset_arrays: list[npt.NDArray[np.uint64]],
    resolution: int,
    minimum_sources: int,
) -> tuple[
    npt.NDArray[np.uint64],
    npt.NDArray[np.uint64],
    npt.NDArray[np.uint64],
    npt.NDArray[np.uint64],
    npt.NDArray[np.uint64],
    npt.NDArray[np.uint64],
]: ...
def _validate_coverage(
    cells: npt.NDArray[np.uint64],
    offsets: npt.NDArray[np.uint64],
    resolution: int,
) -> None: ...
def _prepare_polygon(
    vertices_xyz: npt.NDArray[np.float64],
    ring_offsets: npt.NDArray[np.uint64],
) -> object: ...
def _center(
    cells: npt.NDArray[np.uint64],
    resolution: int,
) -> npt.NDArray[np.float64]: ...
def _cell_at(
    vectors_xyz: npt.NDArray[np.float64],
    resolution: int,
) -> npt.NDArray[np.uint64]: ...
def _corner_many(
    cells: npt.NDArray[np.uint64],
    resolution: int,
) -> npt.NDArray[np.float64]: ...
def _neighbors(
    cells: npt.NDArray[np.uint64],
    resolution: int,
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]: ...
