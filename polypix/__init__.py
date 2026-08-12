"""Fast center-sampled coverage on the HEALPix RING grid."""

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
    _cell_at,
    _center,
    _corner_many,
    _count_caps_per_cell,
    _cover,
    _cover_cap,
    _cover_sweep,
    _summarize_occupancy,
    _validate_coverage,
)

_FOOTPRINT_SHAPE_ERROR = (
    "footprints_xyz must have shape (vertices, 3), "
    "(footprints, vertices, 3), or be a sequence of (vertices, 3) arrays."
)


@dataclass(frozen=True, eq=False, init=False, slots=True)
class Coverage:
    """Validated, read-only segmented HEALPix RING coverage.

    Public construction copies and validates imported arrays. Results returned
    by Polypix reuse their already-owned native buffers without another copy.
    Equality remains identity-based so comparing two large coverages never
    performs an implicit linear scan.
    """

    cells: npt.NDArray[np.uint64]
    offsets: npt.NDArray[np.uint64]
    resolution: int

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "Coverage cannot be constructed directly; use Coverage.from_arrays()."
        )

    @classmethod
    def from_arrays(
        cls,
        cells: Sequence[int] | npt.NDArray[np.integer[Any]],
        offsets: Sequence[int] | npt.NDArray[np.integer[Any]],
        resolution: int,
    ) -> Coverage:
        """Copy and validate imported segmented RING-cell arrays."""
        resolved = _as_resolution(resolution)
        owned_cells = _owned_uint64_vector(cells, "cells")
        owned_offsets = _owned_uint64_vector(offsets, "offsets")
        _validate_coverage(owned_cells, owned_offsets, resolved)
        result = object.__new__(cls)
        _freeze_array(owned_cells)
        _freeze_array(owned_offsets)
        object.__setattr__(result, "cells", owned_cells)
        object.__setattr__(result, "offsets", owned_offsets)
        object.__setattr__(result, "resolution", resolved)
        return result

    @classmethod
    def _from_native(
        cls,
        cells: npt.NDArray[np.uint64],
        offsets: npt.NDArray[np.uint64],
        resolution: int,
    ) -> Coverage:
        """Construct from trusted, newly owned native output buffers."""
        result = object.__new__(cls)
        _freeze_array(cells)
        _freeze_array(offsets)
        object.__setattr__(result, "cells", cells)
        object.__setattr__(result, "offsets", offsets)
        object.__setattr__(result, "resolution", resolution)
        return result

    def __len__(self) -> int:
        """Return the number of input items (segments)."""
        return self.segment_count

    def __getitem__(self, index: SupportsIndex) -> npt.NDArray[np.uint64]:
        """Return a read-only view of one segment's cell IDs."""
        if isinstance(index, (bool, np.bool_)):
            raise TypeError("Coverage indices must be integers, not bool.")
        item = operator.index(index)
        if item < 0:
            item += self.segment_count
        if item < 0 or item >= self.segment_count:
            raise IndexError("Coverage segment index out of range.")
        start = int(self.offsets[item])
        stop = int(self.offsets[item + 1])
        return self.cells[start:stop]

    @property
    def segment_count(self) -> int:
        """Number of segmented input items represented by this result."""
        return self.offsets.size - 1

    @property
    def counts(self) -> npt.NDArray[np.intp]:
        """Number of covered cells for each input item."""
        return np.diff(self.offsets).astype(np.intp, copy=False)


@dataclass(frozen=True, eq=False, init=False, slots=True)
class OccupancySummary:
    """Read-only sparse run and merged-gap statistics for coverage sources."""

    cells: npt.NDArray[np.uint64]
    run_counts: npt.NDArray[np.uint64]
    merged_gap_steps_sum: npt.NDArray[np.uint64]
    merged_gap_counts: npt.NDArray[np.uint64]
    resolution: int
    segment_count: int

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("OccupancySummary values are constructed by Polypix.")

    @classmethod
    def _from_native(
        cls,
        cells: npt.NDArray[np.uint64],
        run_counts: npt.NDArray[np.uint64],
        merged_gap_steps_sum: npt.NDArray[np.uint64],
        merged_gap_counts: npt.NDArray[np.uint64],
        resolution: int,
        segment_count: int,
    ) -> OccupancySummary:
        result = object.__new__(cls)
        for name, array in (
            ("cells", cells),
            ("run_counts", run_counts),
            ("merged_gap_steps_sum", merged_gap_steps_sum),
            ("merged_gap_counts", merged_gap_counts),
        ):
            _freeze_array(array)
            object.__setattr__(result, name, array)
        object.__setattr__(result, "resolution", resolution)
        object.__setattr__(result, "segment_count", segment_count)
        return result

    def __len__(self) -> int:
        """Return the number of observed cells in the sparse summary."""
        return self.cells.size

    @property
    def mean_merged_gap_steps(self) -> npt.NDArray[np.float64]:
        """Mean uncovered steps between merged occupancy windows per cell."""
        mean = np.full(self.cells.size, np.nan, dtype=np.float64)
        measured = self.merged_gap_counts > 0
        np.divide(
            self.merged_gap_steps_sum,
            self.merged_gap_counts,
            out=mean,
            where=measured,
        )
        return mean


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


def _owned_uint64_vector(
    values: Sequence[int] | npt.NDArray[np.integer[Any]],
    name: str,
) -> npt.NDArray[np.uint64]:
    """Return an owned one-dimensional uint64 copy for a public result object."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional integer array.")
    converted = _as_uint64_vector(array, name)
    return np.array(converted, dtype=np.uint64, order="C", copy=True)


def _freeze_array(array: np.ndarray) -> None:
    """Make an owned result buffer read-only through the public array view."""
    array.flags.writeable = False


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
    # ``ascontiguousarray`` promotes scalar inputs to shape ``(1,)``. Preserve
    # their rank so scalar cap radii remain distinguishable from length-one
    # arrays while still copying non-contiguous inputs exactly once.
    return np.asarray(array, dtype=np.float64, order="C")


def _as_float_matrix(values: object, width: int, name: str) -> np.ndarray:
    array = _as_float_array(values, name)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape (items, {width}).")
    return array


def _as_cap_centers(values: object) -> np.ndarray:
    centers = _as_float_array(values, "centers_xyz")
    if centers.ndim == 1 and centers.shape == (3,):
        return centers.reshape(1, 3)
    if centers.ndim == 2 and centers.shape[1] == 3:
        return centers
    raise ValueError("centers_xyz must have shape (3,) or (caps, 3).")


def _as_cap_radii(values: object, count: int) -> np.ndarray:
    radii = _as_float_array(values, "radii_rad")
    if radii.ndim == 0:
        return np.full(count, radii.item(), dtype=np.float64)
    if radii.ndim != 1 or radii.shape[0] != count:
        raise ValueError("radii_rad must be a scalar or contain one radius per center.")
    return radii


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
    return Coverage._from_native(
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
    MemoryError
        If the explicit segmented result cannot be materialized.
    """
    resolved = _as_resolution(resolution)
    vertices, offsets = _as_footprints(footprints_xyz)
    return _cover_xyz(vertices, offsets, resolved, candidate_cells, threads)


def cover_cap(
    centers_xyz: Sequence[float] | Sequence[Sequence[float]] | npt.ArrayLike,
    radii_rad: float | Sequence[float] | npt.ArrayLike,
    resolution: int,
    *,
    candidate_cells: Sequence[int] | npt.NDArray[np.integer[Any]] | None = None,
    threads: int | None = None,
) -> Coverage:
    """Cover exact spherical caps by HEALPix cell-center inclusion.

    ``centers_xyz`` accepts one ``(3,)`` vector or a ``(caps, 3)`` batch.
    ``radii_rad`` is either a scalar shared by every center or an exact
    ``(caps,)`` array with one radius per center; length-one arrays are not
    broadcast. Radii must be finite and lie between zero and pi radians. Input
    vectors are normalized internally. Other arguments and the segmented
    result follow :func:`cover_footprint`.
    """
    resolved = _as_resolution(resolution)
    centers = _as_cap_centers(centers_xyz)
    radii = _as_cap_radii(radii_rad, centers.shape[0])
    requested_threads = _as_threads(threads)
    candidates = (
        None
        if candidate_cells is None
        else _as_uint64_vector(candidate_cells, "candidate_cells")
    )
    return _coverage(
        _cover_cap(centers, radii, resolved, candidates, requested_threads),
        resolved,
    )


def count_caps_per_cell(
    centers_xyz: Sequence[float] | Sequence[Sequence[float]] | npt.ArrayLike,
    radii_rad: float | Sequence[float] | npt.ArrayLike,
    resolution: int,
    *,
    cells: int | Sequence[int] | npt.NDArray[np.integer[Any]] | None = None,
    threads: int | None = None,
) -> npt.NDArray[np.int64]:
    """Count how many exact spherical caps cover each HEALPix cell center.

    With ``cells=None``, the returned dense ``int64`` array is indexed by
    standard RING cell ID and has length ``12 * 4**resolution``. Supplying
    ``cells`` instead returns one count per requested ID in the original order,
    including duplicates, without allocating the full grid.

    Unlike materializing :func:`cover_cap` and calling ``numpy.bincount``, the
    dense operation accumulates contiguous RING spans directly and never
    allocates every cap-cell pair.
    """
    resolved = _as_resolution(resolution)
    centers = _as_cap_centers(centers_xyz)
    radii = _as_cap_radii(radii_rad, centers.shape[0])
    requested_cells = None if cells is None else _as_uint64_vector(cells, "cells")
    return _count_caps_per_cell(
        centers,
        radii,
        resolved,
        requested_cells,
        _as_threads(threads),
    )


def cover_sweep(
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
        raise ValueError("cover_sweep() requires at least two edge samples.")

    requested_threads = _as_threads(threads)
    candidates = (
        None
        if candidate_cells is None
        else _as_uint64_vector(candidate_cells, "candidate_cells")
    )
    return _coverage(
        _cover_sweep(left, right, resolved, candidates, requested_threads),
        resolved,
    )


def summarize_occupancy(
    sources: Coverage | Sequence[Coverage],
) -> OccupancySummary:
    """Summarize source-local runs and merged occupancy gaps.

    Each input ``Coverage`` represents one independent source whose segments
    are aligned, ordered occupancy intervals or bins. A run is maximal over
    consecutive segments for one source and cell. Merged gaps are uncovered
    bins between windows after unioning all sources; leading and trailing bins
    outside the observed windows are excluded. Hits in bins zero and two have
    a merged gap of one.

    The result contains only observed cells in ascending RING order. Gap values
    are expressed in ordinal steps, so callers retain ownership of cadence and
    physical time units.
    """
    normalized_sources: tuple[Coverage, ...]
    if isinstance(sources, Coverage):
        normalized_sources = (sources,)
    elif isinstance(sources, Sequence):
        normalized_sources = tuple(sources)
    else:
        raise TypeError("sources must be a Coverage or a sequence of Coverage values.")
    if not normalized_sources:
        raise ValueError("summarize_occupancy() requires at least one coverage source.")
    if not all(isinstance(source, Coverage) for source in normalized_sources):
        raise TypeError("sources must contain only Coverage values.")

    resolutions = tuple(
        _as_resolution(source.resolution) for source in normalized_sources
    )
    resolution = resolutions[0]
    if any(source_resolution != resolution for source_resolution in resolutions[1:]):
        raise ValueError("all coverage sources must use the same resolution.")

    cell_arrays = [
        _as_uint64_vector(source.cells, f"sources[{index}].cells")
        for index, source in enumerate(normalized_sources)
    ]
    offset_arrays = [
        _as_uint64_vector(source.offsets, f"sources[{index}].offsets")
        for index, source in enumerate(normalized_sources)
    ]
    segment_count = offset_arrays[0].size - 1
    if any(offsets.size - 1 != segment_count for offsets in offset_arrays[1:]):
        raise ValueError(
            "all coverage sources must contain the same number of segments."
        )
    (
        cells,
        run_counts,
        merged_gap_steps_sum,
        merged_gap_counts,
        native_segment_count,
    ) = _summarize_occupancy(cell_arrays, offset_arrays, resolution)
    return OccupancySummary._from_native(
        cells=cells,
        run_counts=run_counts,
        merged_gap_steps_sum=merged_gap_steps_sum,
        merged_gap_counts=merged_gap_counts,
        resolution=resolution,
        segment_count=native_segment_count,
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


def cell_at(
    vectors_xyz: Sequence[float] | Sequence[Sequence[float]] | npt.ArrayLike,
    resolution: int,
) -> npt.NDArray[np.uint64]:
    """Return the HEALPix RING cell containing each Cartesian direction.

    ``vectors_xyz`` accepts one ``(3,)`` vector or a ``(vectors, 3)`` batch.
    Finite nonzero vectors are normalized internally, and the result always
    has shape ``(vectors,)``. Directions exactly on a cell boundary follow the
    deterministic HEALPix partition used by the native RING kernel.
    """
    resolved = _as_resolution(resolution)
    vectors = _as_float_array(vectors_xyz, "vectors_xyz")
    if vectors.ndim == 1 and vectors.shape == (3,):
        vectors = vectors.reshape(1, 3)
    elif vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("vectors_xyz must have shape (3,) or (vectors, 3).")
    return _cell_at(vectors, resolved)


def corners(
    cells: int | Sequence[int] | npt.NDArray[np.integer[Any]],
    resolution: int,
) -> npt.NDArray[np.float64]:
    """Return four unit-vector corners for HEALPix RING indices.

    The result has shape ``(cells, 4, 3)`` in boundary traversal order.
    Validation follows :func:`centers`.
    """
    resolved = _as_resolution(resolution)
    ring = _as_uint64_vector(cells, "cells")
    return _corner_many(ring, resolved)


__all__ = [
    "Coverage",
    "OccupancySummary",
    "__version__",
    "cell_at",
    "corners",
    "centers",
    "count_caps_per_cell",
    "cover_cap",
    "cover_footprint",
    "cover_sweep",
    "summarize_occupancy",
]
