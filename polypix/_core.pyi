from typing import Any

import numpy as np
import numpy.typing as npt

__version__: str

def _cover(
    vertices_xyz: npt.NDArray[np.float64],
    offsets: npt.NDArray[np.uint64],
    resolution: int,
    candidate_cells: npt.NDArray[np.uint64] | None = None,
    threads: int | None = None,
) -> dict[str, Any]: ...
def _cover_strip(
    left_edge_xyz: npt.NDArray[np.float64],
    right_edge_xyz: npt.NDArray[np.float64],
    resolution: int,
    candidate_cells: npt.NDArray[np.uint64] | None = None,
    threads: int | None = None,
) -> dict[str, Any]: ...
def _center(
    cells: npt.NDArray[np.uint64],
    resolution: int,
) -> npt.NDArray[np.float64]: ...
def _boundary_many(
    cells: npt.NDArray[np.uint64],
    resolution: int,
) -> npt.NDArray[np.float64]: ...
