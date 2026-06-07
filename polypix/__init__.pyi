from dataclasses import dataclass
from typing import Sequence, overload

import numpy as np
import numpy.typing as npt

__version__: str

@dataclass(frozen=True)
class Coverage:
    cell_ids: npt.NDArray[np.uint64]
    offsets: npt.NDArray[np.uint64]
    @property
    def counts(self) -> npt.NDArray[np.intp]: ...

def cover_footprint(
    footprints_xyz: Sequence[Sequence[float]] | npt.NDArray[np.float64],
    resolution: int,
) -> Coverage: ...

def cover_swath(
    left_edge_xyz: Sequence[Sequence[float]] | npt.NDArray[np.float64],
    right_edge_xyz: Sequence[Sequence[float]] | npt.NDArray[np.float64],
    resolution: int,
) -> Coverage: ...

@overload
def centers(cell_ids: int) -> tuple[float, float]: ...
@overload
def centers(cell_ids: Sequence[int] | npt.NDArray[np.uint64]) -> npt.NDArray[np.float64]: ...

def boundaries(cell_ids: int | Sequence[int] | npt.NDArray[np.uint64]) -> npt.NDArray[np.float64]: ...
