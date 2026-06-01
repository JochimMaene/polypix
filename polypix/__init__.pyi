from dataclasses import dataclass
from typing import Sequence, overload

import numpy as np
import numpy.typing as npt

__version__: str

@dataclass(frozen=True)
class Polygon:
    vertices: npt.NDArray[np.float64]
    coordinates: str
    offsets: npt.NDArray[np.uint64]
    def __init__(
        self,
        vertices: Sequence[Sequence[float]] | npt.NDArray[np.float64],
        coordinates: str,
    ) -> None: ...
    @classmethod
    def from_xyz(
        cls,
        vertices: Sequence[Sequence[float]] | npt.NDArray[np.float64],
    ) -> Polygon: ...
    @classmethod
    def from_lonlat(
        cls,
        vertices: Sequence[Sequence[float]] | npt.NDArray[np.float64],
    ) -> Polygon: ...

@dataclass(frozen=True)
class MultiPolygon:
    vertices: npt.NDArray[np.float64]
    offsets: npt.NDArray[np.uint64]
    coordinates: str
    @classmethod
    def from_xyz(
        cls,
        vertices: Sequence[Sequence[float]] | Sequence[npt.NDArray[np.float64]] | npt.NDArray[np.float64],
    ) -> MultiPolygon: ...
    @classmethod
    def from_lonlat(
        cls,
        vertices: Sequence[Sequence[float]] | Sequence[npt.NDArray[np.float64]] | npt.NDArray[np.float64],
    ) -> MultiPolygon: ...
    def __len__(self) -> int: ...

@overload
def cover(
    polygons: Polygon,
    resolution: int,
) -> npt.NDArray[np.uint64]: ...
@overload
def cover(
    polygons: MultiPolygon,
    resolution: int,
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.intp]]: ...

@overload
def center(cell_ids: int) -> tuple[float, float]: ...
@overload
def center(cell_ids: Sequence[int] | npt.NDArray[np.uint64]) -> npt.NDArray[np.float64]: ...

def boundary(cell_ids: int | Sequence[int] | npt.NDArray[np.uint64]) -> npt.NDArray[np.float64]: ...
