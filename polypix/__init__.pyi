from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

__version__: str

@dataclass(frozen=True)
class Coverage:
    cells: npt.NDArray[np.uint64]
    offsets: npt.NDArray[np.uint64]
    resolution: int
    @property
    def counts(self) -> npt.NDArray[np.intp]: ...

def cover_footprint(
    footprints_xyz: Sequence[Sequence[float]]
    | Sequence[Sequence[Sequence[float]]]
    | npt.ArrayLike,
    resolution: int,
    *,
    candidate_cells: Sequence[int] | npt.NDArray[np.integer[Any]] | None = None,
    threads: int | None = None,
) -> Coverage: ...
def cover_strip(
    left_edge_xyz: Sequence[Sequence[float]] | npt.ArrayLike,
    right_edge_xyz: Sequence[Sequence[float]] | npt.ArrayLike,
    resolution: int,
    *,
    candidate_cells: Sequence[int] | npt.NDArray[np.integer[Any]] | None = None,
    threads: int | None = None,
) -> Coverage: ...
def centers(
    cells: int | Sequence[int] | npt.NDArray[np.integer[Any]],
    resolution: int,
) -> npt.NDArray[np.float64]: ...
def boundaries(
    cells: int | Sequence[int] | npt.NDArray[np.integer[Any]],
    resolution: int,
) -> npt.NDArray[np.float64]: ...
