from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ._core import __version__, _boundary_many, _center, _cover, _cover_strip

_MAX_RESOLUTION = 29


@dataclass(frozen=True)
class Coverage:
    cells: np.ndarray
    offsets: np.ndarray
    resolution: int

    @property
    def counts(self) -> np.ndarray:
        return np.diff(self.offsets).astype(np.intp, copy=False)


def _as_resolution(value: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("resolution must be an integer, not bool.")
    try:
        resolution = operator.index(value)
    except TypeError as exc:
        raise TypeError("resolution must be an integer.") from exc
    if resolution < 0 or resolution > _MAX_RESOLUTION:
        raise ValueError(f"resolution must be between 0 and {_MAX_RESOLUTION}.")
    return resolution


def _as_threads(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("threads must be a positive integer, not bool.")
    try:
        threads = operator.index(value)
    except TypeError as exc:
        raise TypeError("threads must be a positive integer or None.") from exc
    if threads < 1:
        raise ValueError("threads must be a positive integer.")
    return threads


def _as_uint64_scalar(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must contain integers, not bool.")
    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must contain integers.") from exc
    if integer < 0:
        raise ValueError(f"{name} must contain non-negative integers.")
    if integer > np.iinfo(np.uint64).max:
        raise OverflowError(f"{name} value is out of range for uint64.")
    return integer


def _as_uint64_vector(
    values: int | Sequence[int] | np.ndarray,
    name: str,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0:
        return np.asarray([_as_uint64_scalar(array.item(), name)], dtype=np.uint64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a scalar or one-dimensional integer array.")
    if array.size == 0:
        return np.empty(0, dtype=np.uint64)
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


def _as_float_array(values: object, name: str) -> np.ndarray:
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{name} must contain real numbers, not complex values.")
    if np.issubdtype(array.dtype, np.bool_):
        raise TypeError(f"{name} must contain real numbers, not bool.")
    if not np.issubdtype(array.dtype, np.number) or array.dtype == np.dtype("O"):
        raise TypeError(f"{name} must contain real numbers.")
    return np.ascontiguousarray(array, dtype=np.float64)


def _as_float_matrix(values: object, width: int, name: str) -> np.ndarray:
    array = _as_float_array(values, name)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape (items, {width}).")
    return array


def _dense_footprints(array: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if array.ndim == 2 and array.shape[1] == 3:
        return array, np.asarray([0, array.shape[0]], dtype=np.uint64)
    if array.ndim == 3 and array.shape[2] == 3:
        footprint_count, vertex_count, _ = array.shape
        vertices = np.ascontiguousarray(
            array.reshape(footprint_count * vertex_count, 3)
        )
        if vertex_count == 0:
            offsets = np.zeros(footprint_count + 1, dtype=np.uint64)
        else:
            offsets = np.arange(0, vertices.shape[0] + 1, vertex_count, dtype=np.uint64)
        return vertices, offsets
    return None


def _as_footprints(
    values: Sequence[Sequence[float]]
    | Sequence[Sequence[Sequence[float]]]
    | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(values, np.ndarray):
        if values.dtype == np.dtype("O"):
            raise TypeError("footprints_xyz must contain real numbers.")
        dense_array: np.ndarray | None = _as_float_array(values, "footprints_xyz")
    elif isinstance(values, Sequence) and len(values) > 0:
        first = np.asarray(values[0])
        dense_array = (
            None
            if first.ndim == 2
            else _as_float_array(values, "footprints_xyz")
        )
    else:
        dense_array = None

    if dense_array is not None:
        dense = _dense_footprints(dense_array)
        if dense is None:
            raise ValueError(
                "footprints_xyz must have shape (vertices, 3), "
                "(footprints, vertices, 3), or be a sequence of (vertices, 3) arrays."
            )
        vertices, offsets = dense
        return vertices, offsets

    if not isinstance(values, Sequence) or len(values) == 0:
        raise ValueError(
            "footprints_xyz must have shape (vertices, 3), "
            "(footprints, vertices, 3), or be a sequence of (vertices, 3) arrays."
        )

    if all(
        isinstance(footprint, np.ndarray)
        and footprint.dtype == np.float64
        and footprint.ndim == 2
        and footprint.shape[1] == 3
        and footprint.flags.c_contiguous
        for footprint in values
    ):
        footprints = list(values)
    else:
        footprints = [
            _as_float_matrix(footprint, 3, f"footprints_xyz[{index}]")
            for index, footprint in enumerate(values)
        ]
    counts = np.asarray(
        [footprint.shape[0] for footprint in footprints], dtype=np.uint64
    )
    offsets = np.concatenate(
        (np.zeros(1, dtype=np.uint64), np.cumsum(counts, dtype=np.uint64))
    )
    vertices = np.ascontiguousarray(np.concatenate(footprints, axis=0))
    return vertices, offsets


def _as_ring_cells(
    values: int | Sequence[int] | np.ndarray,
    name: str,
) -> np.ndarray:
    return _as_uint64_vector(values, name)


def _coverage(payload: tuple[np.ndarray, np.ndarray], resolution: int) -> Coverage:
    cells, offsets = payload
    return Coverage(
        cells=cells,
        offsets=offsets,
        resolution=resolution,
    )


def _cover_xyz(
    vertices: np.ndarray,
    offsets: np.ndarray,
    resolution: int,
    candidate_cells: Sequence[int] | np.ndarray | None,
    threads: int | None,
) -> Coverage:
    requested_threads = _as_threads(threads)
    candidates = (
        None
        if candidate_cells is None
        else _as_ring_cells(candidate_cells, "candidate_cells")
    )
    return _coverage(
        _cover(vertices, offsets, resolution, candidates, requested_threads),
        resolution,
    )


def cover_footprint(
    footprints_xyz: Sequence[Sequence[float]]
    | Sequence[Sequence[Sequence[float]]]
    | np.ndarray,
    resolution: int,
    *,
    candidate_cells: Sequence[int] | np.ndarray | None = None,
    threads: int | None = None,
) -> Coverage:
    resolved = _as_resolution(resolution)
    vertices, offsets = _as_footprints(footprints_xyz)
    return _cover_xyz(vertices, offsets, resolved, candidate_cells, threads)


def cover_strip(
    left_edge_xyz: Sequence[Sequence[float]] | np.ndarray,
    right_edge_xyz: Sequence[Sequence[float]] | np.ndarray,
    resolution: int,
    *,
    candidate_cells: Sequence[int] | np.ndarray | None = None,
    threads: int | None = None,
) -> Coverage:
    resolved = _as_resolution(resolution)
    left = _as_float_matrix(left_edge_xyz, 3, "left_edge_xyz")
    right = _as_float_matrix(right_edge_xyz, 3, "right_edge_xyz")
    if left.shape[0] != right.shape[0]:
        raise ValueError(
            "left_edge_xyz and right_edge_xyz must contain the same number of samples."
        )
    if left.shape[0] < 2:
        raise ValueError("cover_strip() requires at least two edge samples.")

    requested_threads = _as_threads(threads)
    candidates = (
        None
        if candidate_cells is None
        else _as_ring_cells(candidate_cells, "candidate_cells")
    )
    return _coverage(
        _cover_strip(left, right, resolved, candidates, requested_threads),
        resolved,
    )


def centers(
    cells: int | Sequence[int] | np.ndarray,
    resolution: int,
) -> np.ndarray:
    resolved = _as_resolution(resolution)
    ring = _as_ring_cells(cells, "cells")
    return _center(ring, resolved)


def boundaries(
    cells: int | Sequence[int] | np.ndarray,
    resolution: int,
) -> np.ndarray:
    resolved = _as_resolution(resolution)
    ring = _as_ring_cells(cells, "cells")
    return _boundary_many(ring, resolved)


__all__ = [
    "Coverage",
    "__version__",
    "boundaries",
    "centers",
    "cover_footprint",
    "cover_strip",
]
