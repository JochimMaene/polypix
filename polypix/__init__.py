from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ._core import __version__, _boundary, _boundary_many, _center, _cover


@dataclass(frozen=True)
class Coverage:
    cell_ids: np.ndarray
    offsets: np.ndarray

    @property
    def counts(self) -> np.ndarray:
        return np.diff(self.offsets).astype(np.intp, copy=False)


def _as_resolution(value: int) -> int:
    if isinstance(value, bool):
        raise TypeError("resolution must be an integer, not bool.")
    try:
        resolution = operator.index(value)
    except TypeError as exc:
        raise TypeError("resolution must be an integer.") from exc
    return resolution


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
    return integer


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
        raise ValueError(f"{name} must have shape (items, {width}).")
    return array


def _as_footprints(values: Sequence[Sequence[float]] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if array.ndim == 2 and array.shape[1] == 3:
        return array, np.asarray([0, array.shape[0]], dtype=np.uint64)
    if array.ndim == 3 and array.shape[2] == 3:
        footprint_count, vertex_count, _ = array.shape
        if vertex_count == 0:
            if footprint_count == 0:
                return np.empty((0, 3), dtype=np.float64), np.asarray([0], dtype=np.uint64)
            raise ValueError("footprints must contain at least one vertex per footprint.")
        vertices = np.ascontiguousarray(array.reshape(footprint_count * vertex_count, 3))
        offsets = np.arange(0, vertices.shape[0] + 1, vertex_count, dtype=np.uint64)
        return vertices, offsets
    raise ValueError("footprints must have shape (vertices, 3) or (footprints, vertices, 3).")


def _coverage(payload: dict) -> Coverage:
    return Coverage(
        cell_ids=payload["cell_ids"],
        offsets=payload["offsets"],
    )


def _cell_ids(values: int | Sequence[int] | np.ndarray) -> tuple[np.ndarray, bool]:
    array = np.asarray(values)
    if array.ndim == 0:
        return np.asarray([_as_uint64_scalar(array.item(), "cell_ids")], dtype=np.uint64), True
    if array.ndim != 1:
        raise ValueError("cell_ids must be a scalar or one-dimensional uint64 array.")
    return _as_uint64_vector(array, "cell_ids"), False


def cover_footprint(footprints_xyz: Sequence[Sequence[float]] | np.ndarray, resolution: int) -> Coverage:
    resolved = _as_resolution(resolution)
    vertices, offsets = _as_footprints(footprints_xyz)
    return _coverage(_cover(vertices, offsets, resolved))


def cover_swath(
    left_edge_xyz: Sequence[Sequence[float]] | np.ndarray,
    right_edge_xyz: Sequence[Sequence[float]] | np.ndarray,
    resolution: int,
) -> Coverage:
    resolved = _as_resolution(resolution)
    left = _as_float_matrix(left_edge_xyz, 3, "left_edge_xyz")
    right = _as_float_matrix(right_edge_xyz, 3, "right_edge_xyz")
    if left.shape[0] != right.shape[0]:
        raise ValueError("left_edge_xyz and right_edge_xyz must contain the same number of samples.")
    if left.shape[0] < 2:
        raise ValueError("cover_swath() requires at least two edge samples.")

    footprints = np.empty((left.shape[0] - 1, 4, 3), dtype=np.float64)
    footprints[:, 0, :] = left[:-1]
    footprints[:, 1, :] = right[:-1]
    footprints[:, 2, :] = right[1:]
    footprints[:, 3, :] = left[1:]
    vertices, offsets = _as_footprints(footprints)
    return _coverage(_cover(vertices, offsets, resolved))


def centers(cell_ids: int | Sequence[int] | np.ndarray) -> tuple[float, float] | np.ndarray:
    ids, is_scalar = _cell_ids(cell_ids)
    lonlat = np.asarray(_center(ids))
    if is_scalar:
        return float(lonlat[0, 0]), float(lonlat[0, 1])
    return lonlat


def boundaries(cell_ids: int | Sequence[int] | np.ndarray) -> np.ndarray:
    ids, is_scalar = _cell_ids(cell_ids)
    if is_scalar:
        return np.asarray(_boundary(int(ids[0])))
    return np.asarray(_boundary_many(ids))


__all__ = ["Coverage", "__version__", "boundaries", "centers", "cover_footprint", "cover_swath"]
