"""Fast center-in-polygon coverage on the HEALPix RING grid."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, SupportsIndex, cast

import numpy as np
import numpy.typing as npt

from ._core import (
    _MAX_RESOLUTION,
    __version__,
    _boundary_many,
    _center,
    _cover,
    _cover_strip,
)

_FOOTPRINT_SHAPE_ERROR = (
    "footprints_xyz must have shape (vertices, 3), "
    "(footprints, vertices, 3), or be a sequence of (vertices, 3) arrays."
)


@dataclass(frozen=True, eq=False)
class Coverage:
    """Segmented HEALPix RING coverage using identity, not array-value, equality."""

    cells: npt.NDArray[np.uint64]
    offsets: npt.NDArray[np.uint64]
    resolution: int

    @property
    def counts(self) -> npt.NDArray[np.intp]:
        """Number of covered cells for each input item."""
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
        integer = operator.index(cast(SupportsIndex, value))
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
    if (
        isinstance(values, np.ndarray)
        and values.dtype == np.float64
        and values.flags.c_contiguous
    ):
        return values
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
    if array.ndim == 1 and array.size == 0:
        return np.empty((0, 3), dtype=np.float64), np.zeros(1, dtype=np.uint64)
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


def _require_dense_footprints(values: object) -> tuple[np.ndarray, np.ndarray]:
    dense = _dense_footprints(_as_float_array(values, "footprints_xyz"))
    if dense is None:
        raise ValueError(_FOOTPRINT_SHAPE_ERROR)
    return dense


def _ragged_footprints(values: Sequence[object]) -> tuple[np.ndarray, np.ndarray]:
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


def _as_footprints(
    values: object,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(values, np.ndarray):
        return _require_dense_footprints(values)
    if not isinstance(values, Sequence):
        return _require_dense_footprints(values)
    if len(values) == 0:
        return np.empty((0, 3), dtype=np.float64), np.zeros(1, dtype=np.uint64)
    if np.asarray(values[0]).ndim == 2:
        try:
            dense = np.asarray(values)
        except ValueError:
            return _ragged_footprints(values)
        if dense.dtype != object:
            return _require_dense_footprints(dense)
        return _ragged_footprints(values)
    return _require_dense_footprints(values)


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
        else _as_uint64_vector(candidate_cells, "candidate_cells")
    )
    return _coverage(
        _cover(vertices, offsets, resolution, candidates, requested_threads),
        resolution,
    )


def cover_footprint(
    footprints_xyz: Sequence[Sequence[float]]
    | Sequence[Sequence[Sequence[float]]]
    | npt.ArrayLike,
    resolution: int,
    *,
    candidate_cells: Sequence[int] | npt.NDArray[np.integer[Any]] | None = None,
    threads: int | None = None,
) -> Coverage:
    """Cover convex spherical footprints by HEALPix cell-center inclusion.

    Parameters
    ----------
    footprints_xyz
        One ``(vertices, 3)`` footprint, a dense batch, or a ragged sequence.
        Finite nonzero vectors are normalized; edges follow minor great-circle
        arcs.
    resolution
        HEALPix resolution from 0 through 29.
    candidate_cells
        Optional RING indices restricting which cell centers are tested.
    threads
        ``None`` selects the automatic policy, 1 is sequential, and larger
        values are reusable worker-pool maximums.

    Returns
    -------
    Coverage
        Flat RING indices and offsets delimiting each input footprint.

    Raises
    ------
    TypeError
        If inputs have incompatible numeric types.
    ValueError
        If shapes, indices, vectors, or polygon geometry are invalid.
    """
    resolved = _as_resolution(resolution)
    vertices, offsets = _as_footprints(footprints_xyz)
    return _cover_xyz(vertices, offsets, resolved, candidate_cells, threads)


def cover_strip(
    left_edge_xyz: Sequence[Sequence[float]] | npt.ArrayLike,
    right_edge_xyz: Sequence[Sequence[float]] | npt.ArrayLike,
    resolution: int,
    *,
    candidate_cells: Sequence[int] | npt.NDArray[np.integer[Any]] | None = None,
    threads: int | None = None,
) -> Coverage:
    """Cover the quadrilateral segments between two sampled spherical edges.

    Each output segment covers ``[left[i], right[i], right[i+1], left[i+1]]``.
    Repeated paired samples create a zero-area segment and are rejected.
    Inputs, resolution, candidates, threading, return value, and errors follow
    :func:`cover_footprint`.
    """
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
        else _as_uint64_vector(candidate_cells, "candidate_cells")
    )
    return _coverage(
        _cover_strip(left, right, resolved, candidates, requested_threads),
        resolved,
    )


def centers(
    cells: int | Sequence[int] | npt.NDArray[np.integer[Any]],
    resolution: int,
) -> npt.NDArray[np.float64]:
    """Return unit-vector centers for HEALPix RING indices.

    The result has shape ``(cells, 3)``. Invalid resolutions, non-integer
    inputs, negative values, and out-of-range indices are rejected.
    """
    resolved = _as_resolution(resolution)
    ring = _as_uint64_vector(cells, "cells")
    return _center(ring, resolved)


def boundaries(
    cells: int | Sequence[int] | npt.NDArray[np.integer[Any]],
    resolution: int,
) -> npt.NDArray[np.float64]:
    """Return four unit-vector corners for HEALPix RING indices.

    The result has shape ``(cells, 4, 3)`` in boundary traversal order.
    Validation follows :func:`centers`.
    """
    resolved = _as_resolution(resolution)
    ring = _as_uint64_vector(cells, "cells")
    return _boundary_many(ring, resolved)


__all__ = [
    "Coverage",
    "__version__",
    "boundaries",
    "centers",
    "cover_footprint",
    "cover_strip",
]
