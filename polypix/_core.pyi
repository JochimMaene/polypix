import numpy as np
import numpy.typing as npt

__version__: str
_MAX_RESOLUTION: int
__all__ = [
    "__version__",
    "_MAX_RESOLUTION",
    "_boundary_many",
    "_center",
    "_cover",
    "_cover_strip",
]

def _cover(
    vertices_xyz: npt.NDArray[np.float64],
    offsets: npt.NDArray[np.uint64],
    resolution: int,
    candidate_cells: npt.NDArray[np.uint64] | None = None,
    threads: int | None = None,
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]: ...
def _cover_strip(
    left_edge_xyz: npt.NDArray[np.float64],
    right_edge_xyz: npt.NDArray[np.float64],
    resolution: int,
    candidate_cells: npt.NDArray[np.uint64] | None = None,
    threads: int | None = None,
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]: ...
def _center(
    cells: npt.NDArray[np.uint64],
    resolution: int,
) -> npt.NDArray[np.float64]: ...
def _boundary_many(
    cells: npt.NDArray[np.uint64],
    resolution: int,
) -> npt.NDArray[np.float64]: ...
