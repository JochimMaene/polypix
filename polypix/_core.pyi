from typing import Any

import numpy as np
import numpy.typing as npt

__version__: str

def _cover(
    vertices_xyz: npt.NDArray[np.float64],
    offsets: npt.NDArray[np.uint64],
    resolution: int,
) -> dict[str, Any]: ...

def _cover_lonlat(
    vertices_lonlat_deg: npt.NDArray[np.float64],
    offsets: npt.NDArray[np.uint64],
    resolution: int,
) -> dict[str, Any]: ...

def _center(cell_ids: npt.NDArray[np.uint64]) -> npt.NDArray[np.float64]: ...
def _boundary(cell_id: int) -> npt.NDArray[np.float64]: ...
def _boundary_many(cell_ids: npt.NDArray[np.uint64]) -> npt.NDArray[np.float64]: ...
