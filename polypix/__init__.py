from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ._core import __version__, _boundary, _boundary_many, _center, _cover, _cover_lonlat


@dataclass(frozen=True)
class Polygon:
    vertices: np.ndarray
    coordinates: str
    offsets: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        width = _coordinate_width(self.coordinates)
        vertices = _as_float_matrix(self.vertices, width, "vertices")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "offsets", np.asarray([0, vertices.shape[0]], dtype=np.uint64))

    @classmethod
    def from_xyz(cls, vertices: Sequence[Sequence[float]] | np.ndarray) -> Polygon:
        return cls(vertices, "xyz")

    @classmethod
    def from_lonlat(cls, vertices: Sequence[Sequence[float]] | np.ndarray) -> Polygon:
        return cls(vertices, "lonlat")


@dataclass(frozen=True)
class MultiPolygon:
    vertices: np.ndarray
    offsets: np.ndarray
    coordinates: str

    def __post_init__(self) -> None:
        width = _coordinate_width(self.coordinates)
        vertices = _as_float_matrix(self.vertices, width, "vertices")
        offsets = _as_offsets(self.offsets, vertices.shape[0])
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "offsets", offsets)

    @classmethod
    def from_xyz(cls, vertices: Sequence[Sequence[float]] | Sequence[np.ndarray] | np.ndarray) -> MultiPolygon:
        flat_vertices, offsets = _as_polygon_vertices(vertices, 3, "polygons")
        return cls(flat_vertices, offsets, "xyz")

    @classmethod
    def from_lonlat(cls, vertices: Sequence[Sequence[float]] | Sequence[np.ndarray] | np.ndarray) -> MultiPolygon:
        flat_vertices, offsets = _as_polygon_vertices(vertices, 2, "polygons")
        return cls(flat_vertices, offsets, "lonlat")

    def __len__(self) -> int:
        return max(self.offsets.shape[0] - 1, 0)


def _coordinate_width(coordinates: str) -> int:
    if coordinates == "xyz":
        return 3
    if coordinates == "lonlat":
        return 2
    raise ValueError("coordinates must be 'xyz' or 'lonlat'.")


def _as_resolution(value: int) -> int:
    if isinstance(value, bool):
        raise TypeError("resolution must be an integer, not bool.")
    try:
        resolution = operator.index(value)
    except TypeError as exc:
        raise TypeError("resolution must be an integer.") from exc
    return int(resolution)


def _as_uint64_scalar(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool.")
    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must contain integers.") from exc
    if integer < 0:
        raise ValueError(f"{name} must contain non-negative integers.")
    if integer > np.iinfo(np.uint64).max:
        raise OverflowError(f"{name} value is out of range for uint64.")
    return int(integer)


def _as_uint64_vector(values: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional uint64 array.")
    if np.issubdtype(array.dtype, np.bool_):
        raise TypeError(f"{name} must contain integers, not bool.")
    if np.issubdtype(array.dtype, np.unsignedinteger):
        return np.ascontiguousarray(array.astype(np.uint64, copy=False))
    if np.issubdtype(array.dtype, np.signedinteger):
        if np.any(array < 0):
            raise ValueError(f"{name} must contain non-negative integers.")
        return np.ascontiguousarray(array.astype(np.uint64, copy=False))
    if array.dtype == np.dtype("O"):
        integers = [_as_uint64_scalar(value, name) for value in array.tolist()]
        return np.ascontiguousarray(np.asarray(integers, dtype=np.uint64))
    raise TypeError(f"{name} must contain integers.")


def _as_float_matrix(values: Sequence[Sequence[float]] | np.ndarray, width: int, name: str) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must be a contiguous array of shape (vertices, {width}).")
    return array


def _as_polygon_vertices(
    values: Sequence[Sequence[float]] | np.ndarray,
    width: int,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        array = np.asarray(values, dtype=np.float64)
    except ValueError:
        array = None

    if array is not None:
        if array.ndim == 2 and array.shape[1] == width:
            vertices = np.ascontiguousarray(array)
            return vertices, np.array([0, vertices.shape[0]], dtype=np.uint64)
        if array.ndim == 3 and array.shape[2] == width:
            polygon_count, vertex_count, _ = array.shape
            vertices = np.ascontiguousarray(array.reshape(polygon_count * vertex_count, width))
            offsets = np.arange(0, vertices.shape[0] + 1, vertex_count, dtype=np.uint64)
            return vertices, offsets

    polygons = [_as_float_matrix(polygon, width, name) for polygon in values]
    offsets = np.empty(len(polygons) + 1, dtype=np.uint64)
    offsets[0] = 0
    if not polygons:
        return np.empty((0, width), dtype=np.float64), offsets
    lengths = np.fromiter((polygon.shape[0] for polygon in polygons), dtype=np.uint64, count=len(polygons))
    np.cumsum(lengths, out=offsets[1:])
    return np.ascontiguousarray(np.concatenate(polygons)), offsets


def _as_offsets(values: Sequence[int] | np.ndarray, vertex_count: int) -> np.ndarray:
    offsets = _as_uint64_vector(values, "offsets")
    if offsets.size == 0:
        raise ValueError("offsets must contain at least the starting zero offset.")
    if int(offsets[0]) != 0:
        raise ValueError("offsets must start at 0.")
    if int(offsets[-1]) != vertex_count:
        raise ValueError("offsets must end at the total vertex count.")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("offsets must be nondecreasing.")
    return offsets


def _coverage_result(payload: dict) -> tuple[np.ndarray, np.ndarray]:
    cell_ids = np.asarray(payload["cell_ids"])
    offsets = np.asarray(payload["offsets"])
    counts = np.diff(offsets).astype(np.intp, copy=False)
    return cell_ids, counts


def _cell_ids(values: int | Sequence[int] | np.ndarray) -> tuple[np.ndarray, bool]:
    array = np.asarray(values)
    if array.ndim == 0:
        return np.asarray([_as_uint64_scalar(array.item(), "cell_ids")], dtype=np.uint64), True
    if array.ndim != 1:
        raise ValueError("cell_ids must be a scalar or one-dimensional uint64 array.")
    return _as_uint64_vector(array, "cell_ids"), False


def cover(
    polygons: Polygon | MultiPolygon,
    resolution: int,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    if not isinstance(polygons, (Polygon, MultiPolygon)):
        raise TypeError(
            "cover() requires a Polygon or MultiPolygon. "
            "Use Polygon.from_lonlat(), Polygon.from_xyz(), MultiPolygon.from_lonlat(), or MultiPolygon.from_xyz()."
        )

    resolved = _as_resolution(resolution)
    if polygons.coordinates == "xyz":
        cell_ids, _counts = _coverage_result(_cover(polygons.vertices, polygons.offsets, resolved))
    elif polygons.coordinates == "lonlat":
        cell_ids, _counts = _coverage_result(_cover_lonlat(polygons.vertices, polygons.offsets, resolved))
    else:
        raise ValueError("unsupported polygon coordinate system.")

    if isinstance(polygons, Polygon):
        return cell_ids
    return cell_ids, _counts


def _cover_flat(
    vertices: Sequence[Sequence[float]] | np.ndarray,
    resolution: int,
    offsets: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    vertices_xyz = _as_float_matrix(vertices, 3, "vertices")
    polygon_offsets = _as_offsets(offsets, vertices_xyz.shape[0])
    resolved = _as_resolution(resolution)
    return _coverage_result(_cover(vertices_xyz, polygon_offsets, resolved))


def center(cell_ids: int | Sequence[int] | np.ndarray) -> tuple[float, float] | np.ndarray:
    ids, is_scalar = _cell_ids(cell_ids)
    lonlat = np.asarray(_center(ids))
    if is_scalar:
        return float(lonlat[0, 0]), float(lonlat[0, 1])
    return lonlat


def boundary(cell_ids: int | Sequence[int] | np.ndarray) -> np.ndarray:
    ids, is_scalar = _cell_ids(cell_ids)
    if is_scalar:
        return np.asarray(_boundary(int(ids[0])))
    return np.asarray(_boundary_many(ids))


__all__ = ["MultiPolygon", "Polygon", "__version__", "boundary", "center", "cover"]
