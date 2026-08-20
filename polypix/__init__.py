"""Fast center-sampled coverage on the HEALPix RING grid."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Never, SupportsIndex, cast, overload

import numpy as np
import numpy.typing as npt

from ._core import (
    _MAX_RESOLUTION,
    __version__,
    _cell_at,
    _center,
    _corner_many,
    _count_caps_per_cell,
    _count_coverage_per_cell,
    _cover,
    _cover_cap,
    _cover_sweep,
    _occupancy_runs,
    _occupancy_stats,
    _sum_coverage_per_cell,
    _validate_coverage,
)

_POLYGON_SHAPE_ERROR = (
    "polygons_xyz must have shape (vertices, 3), "
    "(polygons, vertices, 3), or be a sequence of (vertices, 3) arrays."
)


@dataclass(frozen=True, eq=False, init=False, slots=True)
class Coverage:
    """Validated, read-only segmented HEALPix RING coverage.

    Public construction copies and validates imported arrays. Results returned
    by Polypix reuse their already-owned native buffers without another copy.
    Equality remains identity-based so comparing two large coverages never
    performs an implicit linear scan.
    """

    cells: npt.NDArray[np.int64]
    offsets: npt.NDArray[np.int64]
    resolution: int

    def __init__(self, *_args: Never, **_kwargs: Never) -> None:
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
        # ``_validate_coverage`` range-checks every cell natively; offsets are
        # only bounds-checked there, so they keep the signed scan.
        owned_cells = _owned_int64_vector(cells, "cells", native_range_checked=True)
        owned_offsets = _owned_int64_vector(offsets, "offsets")
        _validate_coverage(
            owned_cells.view(np.uint64),
            owned_offsets.view(np.uint64),
            resolved,
        )
        return cls._from_owned(owned_cells, owned_offsets, resolved)

    @classmethod
    def _from_owned(
        cls,
        cells: npt.NDArray[np.int64],
        offsets: npt.NDArray[np.int64],
        resolution: int,
    ) -> Coverage:
        """Take ownership of arrays already known to satisfy the invariants."""
        result = object.__new__(cls)
        _freeze_array(cells)
        _freeze_array(offsets)
        object.__setattr__(result, "cells", cells)
        object.__setattr__(result, "offsets", offsets)
        object.__setattr__(result, "resolution", resolution)
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
        object.__setattr__(result, "cells", _signed_view(cells))
        object.__setattr__(result, "offsets", _signed_view(offsets))
        object.__setattr__(result, "resolution", resolution)
        return result

    def __len__(self) -> int:
        """Return the number of input items (segments)."""
        return self.segment_count

    def __getitem__(self, index: SupportsIndex) -> npt.NDArray[np.int64]:
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
    def segment_sizes(self) -> npt.NDArray[np.int64]:
        """Number of covered cells for each input item."""
        return np.diff(self.offsets)

    def segment_indices(self) -> npt.NDArray[np.int64]:
        """Return the segment index aligned with every flat cell hit."""
        return np.repeat(
            np.arange(self.segment_count, dtype=np.int64),
            self.segment_sizes,
        )

    def reduce(
        self,
        reducer: CoverageReducer,
    ) -> npt.NDArray[np.int64] | npt.NDArray[np.float64]:
        """Accumulate this coverage with a :class:`Count` or :class:`Sum`.

        The reducer vocabulary is the one the covering operations accept as
        ``into=``, so a stored coverage reduces exactly as a fused call
        would have.
        """
        if not isinstance(reducer, (Count, Sum)):
            raise TypeError("reducer must be a Count or Sum reducer.")
        return _reduce_coverage(self, reducer)

    def filter_hits(
        self,
        mask: Sequence[bool] | npt.NDArray[np.bool_],
    ) -> Coverage:
        """Return coverage containing only flat hits selected by a boolean mask."""
        selected = np.asarray(mask)
        if selected.ndim != 1:
            raise ValueError("mask must be a one-dimensional boolean array.")
        if selected.size != self.cells.size:
            raise ValueError("mask must contain one value per covered cell.")
        if not np.issubdtype(selected.dtype, np.bool_):
            # An empty Python sequence carries no dtype to honour, so accept it.
            # Anything else reaching here was explicitly typed as non-boolean.
            if selected.size or isinstance(mask, np.ndarray):
                raise TypeError("mask must contain boolean values.")
            selected = np.empty(0, dtype=np.bool_)

        cumulative = np.empty(selected.size + 1, dtype=np.int64)
        cumulative[0] = 0
        np.cumsum(selected, dtype=np.int64, out=cumulative[1:])
        # Dropping hits keeps cells in range, keeps offsets nondecreasing, and
        # keeps each segment's cells unique, so the result needs no rescan.
        return Coverage._from_owned(
            self.cells[selected],
            cumulative[self.offsets],
            self.resolution,
        )


@dataclass(frozen=True, eq=False, init=False, slots=True)
class OccupancyRuns:
    """Read-only cell-major ordinal occupancy runs.

    For cell ``i``, ``starts[offsets[i]:offsets[i + 1]]`` and the matching
    ``stops`` form maximal half-open segment intervals ``[start, stop)``.
    """

    cells: npt.NDArray[np.int64]
    offsets: npt.NDArray[np.int64]
    starts: npt.NDArray[np.int64]
    stops: npt.NDArray[np.int64]
    resolution: int
    segment_count: int
    minimum_sources: int
    source_count: int

    def __init__(self, *_args: Never, **_kwargs: Never) -> None:
        raise TypeError("OccupancyRuns values are constructed by Polypix.")

    @classmethod
    def _from_native(
        cls,
        cells: npt.NDArray[np.uint64],
        offsets: npt.NDArray[np.uint64],
        starts: npt.NDArray[np.uint64],
        stops: npt.NDArray[np.uint64],
        resolution: int,
        segment_count: int,
        minimum_sources: int,
        source_count: int,
    ) -> OccupancyRuns:
        result = object.__new__(cls)
        for name, array in (
            ("cells", cells),
            ("offsets", offsets),
            ("starts", starts),
            ("stops", stops),
        ):
            object.__setattr__(result, name, _signed_view(array))
        object.__setattr__(result, "resolution", resolution)
        object.__setattr__(result, "segment_count", segment_count)
        object.__setattr__(result, "minimum_sources", minimum_sources)
        object.__setattr__(result, "source_count", source_count)
        return result

    def __len__(self) -> int:
        """Return the number of cells having at least one qualifying run."""
        return self.cells.size

    @property
    def run_counts(self) -> npt.NDArray[np.int64]:
        """Number of qualifying runs for each cell."""
        return np.diff(self.offsets)


@dataclass(frozen=True, eq=False, init=False, slots=True)
class OccupancyStats:
    """Read-only per-cell occupancy statistics on one thresholded axis.

    Every field describes the same source-unioned occupancy of a cell after
    ``minimum_sources`` thresholding. ``first_start`` and ``last_stop`` bound
    the observed window so callers can apply their own leading, trailing, or
    cyclic gap policy; the gap fields cover complete internal gaps only.
    """

    cells: npt.NDArray[np.int64]
    run_counts: npt.NDArray[np.int64]
    internal_gap_steps_sum: npt.NDArray[np.int64]
    maximum_internal_gap_steps: npt.NDArray[np.int64]
    first_start: npt.NDArray[np.int64]
    last_stop: npt.NDArray[np.int64]
    resolution: int
    segment_count: int
    minimum_sources: int
    source_count: int

    def __init__(self, *_args: Never, **_kwargs: Never) -> None:
        raise TypeError("OccupancyStats values are constructed by Polypix.")

    @classmethod
    def _from_native(
        cls,
        arrays: tuple[npt.NDArray[np.uint64], ...],
        resolution: int,
        segment_count: int,
        minimum_sources: int,
        source_count: int,
    ) -> OccupancyStats:
        result = object.__new__(cls)
        names = (
            "cells",
            "run_counts",
            "internal_gap_steps_sum",
            "maximum_internal_gap_steps",
            "first_start",
            "last_stop",
        )
        for name, array in zip(names, arrays, strict=True):
            object.__setattr__(result, name, _signed_view(array))
        object.__setattr__(result, "resolution", resolution)
        object.__setattr__(result, "segment_count", segment_count)
        object.__setattr__(result, "minimum_sources", minimum_sources)
        object.__setattr__(result, "source_count", source_count)
        return result

    def __len__(self) -> int:
        """Return the number of cells having at least one qualifying run."""
        return self.cells.size

    @property
    def internal_gap_counts(self) -> npt.NDArray[np.int64]:
        """Number of complete internal gaps for each cell."""
        return self.run_counts - 1


def _as_integer(value: object, name: str, expected: str) -> int:
    """Reject bool, then coerce through ``__index__``, naming the argument."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be {expected}, not bool.")
    try:
        return operator.index(cast(SupportsIndex, value))
    except TypeError as exc:
        raise TypeError(f"{name} must be {expected}.") from exc


def _as_resolution(value: int) -> int:
    resolution = _as_integer(value, "resolution", "an integer")
    if resolution < 0 or resolution > _MAX_RESOLUTION:
        raise ValueError(f"resolution must be between 0 and {_MAX_RESOLUTION}.")
    return resolution


def cell_count(resolution: int) -> int:
    """Return the number of fixed-resolution HEALPix cells."""
    resolved = _as_resolution(resolution)
    return 12 * (1 << (2 * resolved))


def _as_candidates(
    candidate_cells: int | Sequence[int] | npt.NDArray[np.integer[Any]] | None,
) -> npt.NDArray[np.uint64] | None:
    if candidate_cells is None:
        return None
    return _as_uint64_vector(
        candidate_cells, "candidate_cells", native_range_checked=True
    )


def _as_threads(value: int | None) -> int | None:
    if value is None:
        return None
    threads = _as_integer(value, "threads", "a positive integer")
    if threads < 1:
        raise ValueError("threads must be a positive integer.")
    return threads


def _as_uint64_scalar(value: object, name: str) -> int:
    integer = _as_integer(value, name, "an integer")
    if integer < 0:
        raise ValueError(f"{name} must contain non-negative integers.")
    if integer > np.iinfo(np.uint64).max:
        raise OverflowError(f"{name} value is out of range for uint64.")
    return integer


def _as_uint64_vector(
    values: int | Sequence[int] | np.ndarray,
    name: str,
    *,
    native_range_checked: bool = False,
) -> np.ndarray:
    """Reinterpret an integer array as the ``uint64`` the kernel expects.

    Set ``native_range_checked`` only where a native cell-range validation is
    guaranteed to follow. A negative signed index reinterprets as a ``u64`` at
    or above ``1 << 63``, which no resolution can contain, so that pass rejects
    it and names it as negative. Skipping the scan here keeps importing a
    Polypix result array free of any pass the unsigned kernel does not need.
    Offset arrays never set it: they are bounds-checked rather than
    range-checked, and are small enough that the scan does not matter.
    """
    array = np.asarray(values)
    if array.ndim == 0:
        return np.asarray([_as_uint64_scalar(array.item(), name)], dtype=np.uint64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a scalar or one-dimensional integer array.")
    if np.issubdtype(array.dtype, np.bool_):
        raise TypeError(f"{name} must contain integers, not bool.")
    integer_dtype = np.issubdtype(array.dtype, np.unsignedinteger) or np.issubdtype(
        array.dtype, np.signedinteger
    )
    if array.size == 0:
        # An empty ordinary sequence has no values from which NumPy can infer
        # its intended integer dtype. Explicitly typed arrays do, so enforce
        # their dtype exactly as for a nonempty input.
        untyped_sequence = isinstance(values, Sequence) and not isinstance(
            values, np.ndarray
        )
        if not integer_dtype and not untyped_sequence:
            raise TypeError(f"{name} must contain integers.")
        return np.empty(0, dtype=np.uint64)
    if np.issubdtype(array.dtype, np.unsignedinteger):
        return _aligned(np.ascontiguousarray(array.astype(np.uint64, copy=False)))
    if np.issubdtype(array.dtype, np.signedinteger):
        # A reduction reads the array once and allocates nothing, where
        # ``any(array < 0)`` also materializes a full boolean temporary.
        if not native_range_checked and array.min() < 0:
            raise ValueError(f"{name} must contain non-negative integers.")
        if (
            array.dtype == np.dtype(np.int64)
            and array.flags.c_contiguous
            and array.flags.aligned
        ):
            return array.view(np.uint64)
        return _aligned(np.ascontiguousarray(array.astype(np.uint64, copy=False)))
    if array.dtype == np.dtype("O"):
        integers = [_as_uint64_scalar(value, name) for value in array.tolist()]
        return np.ascontiguousarray(np.asarray(integers, dtype=np.uint64))
    raise TypeError(f"{name} must contain integers.")


def _owned_int64_vector(
    values: Sequence[int] | npt.NDArray[np.integer[Any]],
    name: str,
    *,
    native_range_checked: bool = False,
) -> npt.NDArray[np.int64]:
    """Return an owned one-dimensional int64 copy for a public result object."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional integer array.")
    converted = _as_uint64_vector(
        values, name, native_range_checked=native_range_checked
    )
    # A signed input that survived the non-negative check above is already
    # inside int64, so only unsigned and object inputs can exceed it. Skipping
    # the pass keeps importing a Polypix-produced array to a single copy.
    if converted.size and not np.issubdtype(array.dtype, np.signedinteger):
        if converted.max() > np.iinfo(np.int64).max:
            raise OverflowError(f"{name} value is out of range for int64.")
    return np.array(converted, dtype=np.int64, order="C", copy=True)


def _signed_view(values: npt.NDArray[np.uint64]) -> npt.NDArray[np.int64]:
    """Expose trusted non-negative native indices as a zero-copy signed view."""
    return values.view(np.int64)


def _trusted_uint64(values: npt.NDArray[np.int64]) -> npt.NDArray[np.uint64]:
    """Reinterpret a ``Coverage``-owned index array without rescanning it.

    ``Coverage`` validates its arrays once and exposes them read-only, so the
    range and contiguity checks in :func:`_as_uint64_vector` cannot fail here.
    """
    return values.view(np.uint64)


def _aligned(array: np.ndarray) -> np.ndarray:
    """Return an array the kernel can borrow as a slice.

    A contiguous array of the right dtype may still sit on an unaligned
    address, which happens whenever one is viewed out of a packed byte buffer.
    ``asarray`` and ``ascontiguousarray`` both return such an array unchanged,
    because contiguity and dtype already match, so the copy has to be forced.
    Native code cannot borrow it: a slice over a misaligned pointer is
    undefined behaviour, and rust-numpy refuses to build one.
    """
    if array.flags.aligned:
        return array
    return np.array(array, dtype=array.dtype, order="C", copy=True)


def _freeze_array(array: np.ndarray) -> None:
    """Make an owned result buffer read-only through the public array view."""
    array.flags.writeable = False


def _as_float_array(values: object, name: str) -> np.ndarray:
    if (
        isinstance(values, np.ndarray)
        and values.dtype == np.float64
        and values.flags.c_contiguous
        and values.flags.aligned
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
    return _aligned(np.asarray(array, dtype=np.float64, order="C"))


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
        radius = float(radii.item())
        if not np.isfinite(radius) or radius < 0.0 or radius > np.pi:
            raise ValueError("radii_rad must be finite and between zero and pi.")
        return np.full(count, radius, dtype=np.float64)
    if radii.ndim != 1 or radii.shape[0] != count:
        raise ValueError("radii_rad must be a scalar or contain one radius per center.")
    return radii


def _dense_polygons(array: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if array.ndim == 1 and array.size == 0:
        return np.empty((0, 3), dtype=np.float64), np.zeros(1, dtype=np.uint64)
    if array.ndim == 2 and array.shape[1] == 3:
        return array, np.asarray([0, array.shape[0]], dtype=np.uint64)
    if array.ndim == 3 and array.shape[2] == 3:
        polygon_count, vertex_count, _ = array.shape
        vertices = np.ascontiguousarray(array.reshape(polygon_count * vertex_count, 3))
        if vertex_count == 0:
            offsets = np.zeros(polygon_count + 1, dtype=np.uint64)
        else:
            offsets = np.arange(0, vertices.shape[0] + 1, vertex_count, dtype=np.uint64)
        return vertices, offsets
    return None


def _require_dense_polygons(values: object) -> tuple[np.ndarray, np.ndarray]:
    dense = _dense_polygons(_as_float_array(values, "polygons_xyz"))
    if dense is None:
        raise ValueError(_POLYGON_SHAPE_ERROR)
    return dense


def _ragged_polygons(values: Sequence[object]) -> tuple[np.ndarray, np.ndarray]:
    polygons = [
        _as_float_matrix(polygon, 3, f"polygons_xyz[{index}]")
        for index, polygon in enumerate(values)
    ]
    counts = np.asarray([polygon.shape[0] for polygon in polygons], dtype=np.uint64)
    offsets = np.concatenate(
        (np.zeros(1, dtype=np.uint64), np.cumsum(counts, dtype=np.uint64))
    )
    vertices = np.ascontiguousarray(np.concatenate(polygons, axis=0))
    return vertices, offsets


def _as_polygons(
    values: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Accept one polygon, a uniform batch, or a batch of differing lengths."""
    if not isinstance(values, Sequence) or isinstance(values, np.ndarray):
        return _require_dense_polygons(values)
    if len(values) == 0:
        return np.empty((0, 3), dtype=np.float64), np.zeros(1, dtype=np.uint64)
    if np.ndim(values[0]) != 2:
        return _require_dense_polygons(values)
    shapes = [np.shape(polygon) for polygon in values]
    if (
        all(len(shape) == 2 for shape in shapes)
        and len({shape[0] for shape in shapes}) == 1
    ):
        return _require_dense_polygons(values)
    return _ragged_polygons(values)


def _as_packed_polygons(
    values: object,
    vertex_offsets: Sequence[int] | npt.NDArray[np.integer[Any]],
) -> tuple[np.ndarray, np.ndarray]:
    vertices = _as_float_matrix(values, 3, "polygons_xyz")
    raw_offsets = np.asarray(vertex_offsets)
    if raw_offsets.ndim != 1:
        raise ValueError("vertex_offsets must be a one-dimensional integer array.")
    offsets = _as_uint64_vector(vertex_offsets, "vertex_offsets")
    if offsets.size == 0:
        raise ValueError("vertex_offsets must contain at least the initial zero.")
    if offsets[0] != 0:
        raise ValueError("vertex_offsets must start at zero.")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("vertex_offsets must be monotonically non-decreasing.")
    if offsets[-1] != vertices.shape[0]:
        raise ValueError(
            "vertex_offsets must end at the number of packed polygon vertices."
        )
    return vertices, offsets


@dataclass(frozen=True, eq=False, slots=True)
class Count:
    """Count how many segments contain each cell.

    With ``cells=None`` the result is a dense array indexed by RING cell ID.
    Supplying ``cells`` returns one count per requested ID, in query order and
    including duplicates, without a grid-sized result. Small grids still use a
    dense scratch array internally; large ones accumulate through a hash table.
    """

    cells: int | Sequence[int] | npt.NDArray[np.integer[Any]] | None = None


@dataclass(frozen=True, eq=False, slots=True)
class Sum:
    """Accumulate one finite value per segment into the cells it covers.

    ``values`` is a scalar shared by every segment or one value per segment.
    ``cells`` selects the output as for :class:`Count`.
    """

    values: float | Sequence[float] | npt.ArrayLike
    cells: int | Sequence[int] | npt.NDArray[np.integer[Any]] | None = None


@dataclass(frozen=True, eq=False, slots=True)
class Stats:
    """Accumulate per-cell occupancy statistics without building the runs."""


CoverageReducer = Count | Sum


@overload
def cover_convex_polygon(
    polygons_xyz: Sequence[Sequence[float]]
    | Sequence[Sequence[Sequence[float]]]
    | npt.ArrayLike,
    resolution: int,
    *,
    vertex_offsets: Sequence[int] | npt.NDArray[np.integer[Any]] | None = None,
    candidate_cells: int | Sequence[int] | npt.NDArray[np.integer[Any]] | None = None,
    threads: int | None = None,
    into: None = None,
) -> Coverage: ...


@overload
def cover_convex_polygon(
    polygons_xyz: Sequence[Sequence[float]]
    | Sequence[Sequence[Sequence[float]]]
    | npt.ArrayLike,
    resolution: int,
    *,
    vertex_offsets: Sequence[int] | npt.NDArray[np.integer[Any]] | None = None,
    candidate_cells: int | Sequence[int] | npt.NDArray[np.integer[Any]] | None = None,
    threads: int | None = None,
    into: Count,
) -> npt.NDArray[np.int64]: ...


@overload
def cover_convex_polygon(
    polygons_xyz: Sequence[Sequence[float]]
    | Sequence[Sequence[Sequence[float]]]
    | npt.ArrayLike,
    resolution: int,
    *,
    vertex_offsets: Sequence[int] | npt.NDArray[np.integer[Any]] | None = None,
    candidate_cells: int | Sequence[int] | npt.NDArray[np.integer[Any]] | None = None,
    threads: int | None = None,
    into: Sum,
) -> npt.NDArray[np.float64]: ...


def cover_convex_polygon(
    polygons_xyz: Sequence[Sequence[float]]
    | Sequence[Sequence[Sequence[float]]]
    | npt.ArrayLike,
    resolution: int,
    *,
    vertex_offsets: Sequence[int] | npt.NDArray[np.integer[Any]] | None = None,
    candidate_cells: int | Sequence[int] | npt.NDArray[np.integer[Any]] | None = None,
    threads: int | None = None,
    into: CoverageReducer | None = None,
) -> Coverage | npt.NDArray[np.int64] | npt.NDArray[np.float64]:
    """Cover convex spherical polygons by HEALPix cell-center inclusion.

    Parameters
    ----------
    polygons_xyz
        One ``(vertices, 3)`` polygon, a dense batch, or a ragged sequence.
        Finite nonzero vectors are normalized; edges follow minor great-circle
        arcs.
    resolution
        HEALPix resolution from 0 through 29.
    vertex_offsets
        Optional boundaries for a packed ``(vertices, 3)`` ragged batch.
    candidate_cells
        Optional RING indices restricting which cell centers are tested.
    into
        Optional :class:`Count` or :class:`Sum` reducer. ``None`` returns the
        segmented ``Coverage``; a reducer returns its accumulated array and
        lets Polypix skip building the cell lists where it can.
    threads
        ``None`` selects the automatic policy, 1 is sequential, and larger
        values are reusable worker-pool maximums.

    Returns
    -------
    Coverage or ndarray
        Flat RING indices and offsets delimiting each input footprint, or the
        reduced array when ``into`` is given.

    Raises
    ------
    TypeError
        If inputs have incompatible numeric types.
    ValueError
        If shapes, indices, vectors, or polygon geometry are invalid.
    MemoryError
        If the explicit segmented result does not fit in memory.
    """
    resolved = _as_resolution(resolution)
    vertices, offsets = (
        _as_polygons(polygons_xyz)
        if vertex_offsets is None
        else _as_packed_polygons(polygons_xyz, vertex_offsets)
    )
    reducer = _as_coverage_reducer(into)
    candidates = _as_candidates(candidate_cells)
    if candidates is None:
        candidates = _selected_candidates(reducer, resolved)
    coverage = Coverage._from_native(
        *_cover(
            vertices,
            offsets,
            resolved,
            candidates,
            _as_threads(threads),
        ),
        resolved,
    )
    return coverage if reducer is None else _reduce_coverage(coverage, reducer)


@overload
def cover_cap(
    centers_xyz: Sequence[float] | Sequence[Sequence[float]] | npt.ArrayLike,
    radii_rad: float | Sequence[float] | npt.ArrayLike,
    resolution: int,
    *,
    candidate_cells: int | Sequence[int] | npt.NDArray[np.integer[Any]] | None = None,
    threads: int | None = None,
    into: None = None,
) -> Coverage: ...


@overload
def cover_cap(
    centers_xyz: Sequence[float] | Sequence[Sequence[float]] | npt.ArrayLike,
    radii_rad: float | Sequence[float] | npt.ArrayLike,
    resolution: int,
    *,
    candidate_cells: int | Sequence[int] | npt.NDArray[np.integer[Any]] | None = None,
    threads: int | None = None,
    into: Count,
) -> npt.NDArray[np.int64]: ...


@overload
def cover_cap(
    centers_xyz: Sequence[float] | Sequence[Sequence[float]] | npt.ArrayLike,
    radii_rad: float | Sequence[float] | npt.ArrayLike,
    resolution: int,
    *,
    candidate_cells: int | Sequence[int] | npt.NDArray[np.integer[Any]] | None = None,
    threads: int | None = None,
    into: Sum,
) -> npt.NDArray[np.float64]: ...


def cover_cap(
    centers_xyz: Sequence[float] | Sequence[Sequence[float]] | npt.ArrayLike,
    radii_rad: float | Sequence[float] | npt.ArrayLike,
    resolution: int,
    *,
    candidate_cells: int | Sequence[int] | npt.NDArray[np.integer[Any]] | None = None,
    threads: int | None = None,
    into: CoverageReducer | None = None,
) -> Coverage | npt.NDArray[np.int64] | npt.NDArray[np.float64]:
    """Cover exact spherical caps by HEALPix cell-center inclusion.

    ``centers_xyz`` accepts one ``(3,)`` vector or a ``(caps, 3)`` batch.
    ``radii_rad`` is either a scalar shared by every center or an exact
    ``(caps,)`` array with one radius per center; length-one arrays are not
    broadcast. Radii must be finite and lie between zero and pi radians. Input
    vectors are normalized internally. Other arguments and the segmented
    result follow :func:`cover_convex_polygon`.

    ``into=Count()`` counts caps per cell directly, without building cap
    coverage first. For selected cells the kernel picks whichever of the two is
    cheaper and falls back to covering once and counting when that wins.
    """
    resolved = _as_resolution(resolution)
    reducer = _as_coverage_reducer(into)
    centers = _as_cap_centers(centers_xyz)
    radii = _as_cap_radii(radii_rad, centers.shape[0])
    requested_threads = _as_threads(threads)
    if isinstance(reducer, Count) and candidate_cells is None:
        selected = (
            None
            if reducer.cells is None
            else _as_uint64_vector(reducer.cells, "cells", native_range_checked=True)
        )
        counts = _count_caps_per_cell(
            centers, radii, resolved, selected, requested_threads
        )
        if counts is not None:
            return counts
    candidates = _as_candidates(candidate_cells)
    if candidates is None:
        candidates = _selected_candidates(reducer, resolved)
    coverage = Coverage._from_native(
        *_cover_cap(centers, radii, resolved, candidates, requested_threads),
        resolved,
    )
    return coverage if reducer is None else _reduce_coverage(coverage, reducer)


def _coverage_cells(
    values: int | Sequence[int] | npt.NDArray[np.integer[Any]],
    resolution: int,
) -> npt.NDArray[np.uint64]:
    """Validate a positional reduction query, naming the public argument.

    The native reducers range-check this array too, but they know it as
    ``requested_cells``, so the message is produced here instead. One maximum
    classifies both failures: a reinterpreted negative index lands at or above
    ``1 << 63``, above every resolution's cell count.
    """
    cells = _as_uint64_vector(values, "cells", native_range_checked=True)
    if cells.size:
        largest = int(cells.max())
        if largest >= cell_count(resolution):
            if largest >= 1 << 63:
                raise ValueError("cells must contain non-negative integers.")
            raise ValueError(
                f"cells must contain valid RING indices at resolution {resolution}."
            )
    return cells


# Reducing over a selection asks about those cells and no others, so the
# selection is itself a candidate set: the kernel can test each selected cell
# against each footprint instead of scanning every ring the footprints cross
# and discarding nearly everything it finds. Testing costs one cell decode plus
# one predicate per selected cell per footprint, so it wins only while the
# selection stays small. Measured against covering then gathering, it is worth
# between 1.4x and 380x below the two bounds here, and loses by up to 50x well
# above them; at the bounds themselves the worst case measured was a 9% loss on
# a one-millisecond call. The proportional bound keeps the trade comparable
# across resolutions, and the absolute bound stops a resolution-11 grid from
# admitting fifty thousand cells merely because they are a small share of it.
_SELECTED_CANDIDATE_GRID_DIVISOR = 1000
_SELECTED_CANDIDATE_MAXIMUM = 4096


def _selected_candidates(
    reducer: CoverageReducer | None,
    resolution: int,
) -> npt.NDArray[np.uint64] | None:
    """Return a reducer's cell selection as a candidate set, when that is cheaper.

    Returns ``None`` when there is no selection or it is large enough that
    scanning and gathering wins, leaving the caller's normal path in place.
    Restricting the scan cannot change a result: a cell outside the selection
    never contributes to a selected cell's count, and never to its sum either,
    so :class:`Sum` keeps both its value and its addition order.
    """
    if reducer is None or reducer.cells is None:
        return None
    cells = _coverage_cells(reducer.cells, resolution)
    limit = min(
        cell_count(resolution) // _SELECTED_CANDIDATE_GRID_DIVISOR,
        _SELECTED_CANDIDATE_MAXIMUM,
    )
    return cells if cells.size <= limit else None


def _as_segment_values(
    values: object,
    segment_count: int,
) -> npt.NDArray[np.float64]:
    array = _as_float_array(values, "values")
    if array.ndim == 0:
        value = float(array.item())
        if not np.isfinite(value):
            raise ValueError("values must contain only finite values.")
        return np.full(segment_count, value, dtype=np.float64)
    if array.ndim != 1 or array.size != segment_count:
        raise ValueError(
            "values must be a scalar or contain one value per coverage segment."
        )
    return cast(npt.NDArray[np.float64], array)


def _reduce_coverage(
    coverage: Coverage,
    reducer: CoverageReducer,
) -> npt.NDArray[np.int64] | npt.NDArray[np.float64]:
    """Apply a coverage reducer to cell lists that are already built."""
    requested = (
        None
        if reducer.cells is None
        else _coverage_cells(reducer.cells, coverage.resolution)
    )
    if isinstance(reducer, Count):
        if requested is not None and requested.size == 0:
            return np.empty(0, dtype=np.int64)
        return _count_coverage_per_cell(
            _trusted_uint64(coverage.cells),
            coverage.resolution,
            requested,
        )
    segment_values = _as_segment_values(reducer.values, coverage.segment_count)
    if requested is not None and requested.size == 0:
        if np.any(~np.isfinite(segment_values)):
            raise ValueError("values must contain only finite values.")
        return np.empty(0, dtype=np.float64)
    return _sum_coverage_per_cell(
        _trusted_uint64(coverage.cells),
        _trusted_uint64(coverage.offsets),
        segment_values,
        coverage.resolution,
        requested,
    )


def _as_coverage_reducer(into: object) -> CoverageReducer | None:
    if into is None or isinstance(into, (Count, Sum)):
        return into
    raise TypeError("into must be a Count or Sum reducer, or None.")


@overload
def cover_sweep(
    left_edge_xyz: Sequence[Sequence[float]] | npt.ArrayLike,
    right_edge_xyz: Sequence[Sequence[float]] | npt.ArrayLike,
    resolution: int,
    *,
    candidate_cells: int | Sequence[int] | npt.NDArray[np.integer[Any]] | None = None,
    threads: int | None = None,
    into: None = None,
) -> Coverage: ...


@overload
def cover_sweep(
    left_edge_xyz: Sequence[Sequence[float]] | npt.ArrayLike,
    right_edge_xyz: Sequence[Sequence[float]] | npt.ArrayLike,
    resolution: int,
    *,
    candidate_cells: int | Sequence[int] | npt.NDArray[np.integer[Any]] | None = None,
    threads: int | None = None,
    into: Count,
) -> npt.NDArray[np.int64]: ...


@overload
def cover_sweep(
    left_edge_xyz: Sequence[Sequence[float]] | npt.ArrayLike,
    right_edge_xyz: Sequence[Sequence[float]] | npt.ArrayLike,
    resolution: int,
    *,
    candidate_cells: int | Sequence[int] | npt.NDArray[np.integer[Any]] | None = None,
    threads: int | None = None,
    into: Sum,
) -> npt.NDArray[np.float64]: ...


def cover_sweep(
    left_edge_xyz: Sequence[Sequence[float]] | npt.ArrayLike,
    right_edge_xyz: Sequence[Sequence[float]] | npt.ArrayLike,
    resolution: int,
    *,
    candidate_cells: int | Sequence[int] | npt.NDArray[np.integer[Any]] | None = None,
    threads: int | None = None,
    into: CoverageReducer | None = None,
) -> Coverage | npt.NDArray[np.int64] | npt.NDArray[np.float64]:
    """Cover the quadrilateral segments between two sampled spherical edges.

    Each output segment covers ``[left[i], right[i], right[i+1], left[i+1]]``.
    Repeated paired samples create a zero-area segment and are rejected.
    Inputs, resolution, candidates, threading, ``into``, return value, and
    errors follow :func:`cover_convex_polygon`.
    """
    resolved = _as_resolution(resolution)
    reducer = _as_coverage_reducer(into)
    left = _as_float_matrix(left_edge_xyz, 3, "left_edge_xyz")
    right = _as_float_matrix(right_edge_xyz, 3, "right_edge_xyz")
    if left.shape[0] != right.shape[0]:
        raise ValueError(
            "left_edge_xyz and right_edge_xyz must contain the same number of samples."
        )
    requested_threads = _as_threads(threads)
    candidates = _as_candidates(candidate_cells)
    if candidates is None:
        candidates = _selected_candidates(reducer, resolved)
    coverage = Coverage._from_native(
        *_cover_sweep(left, right, resolved, candidates, requested_threads),
        resolved,
    )
    return coverage if reducer is None else _reduce_coverage(coverage, reducer)


def _as_minimum_sources(value: int) -> int:
    minimum_sources = _as_integer(value, "minimum_sources", "a positive integer")
    if minimum_sources < 1:
        raise ValueError("minimum_sources must be a positive integer.")
    return minimum_sources


def _prepared_sources(
    sources: Coverage | Sequence[Coverage],
    minimum_sources: int,
    operation: str,
) -> tuple[
    list[npt.NDArray[np.uint64]],
    list[npt.NDArray[np.uint64]],
    int,
    int,
    int,
    int,
]:
    """Validate aligned coverage sources and view them for the native call."""
    threshold = _as_minimum_sources(minimum_sources)
    normalized: tuple[Coverage, ...]
    if isinstance(sources, Coverage):
        normalized = (sources,)
    elif isinstance(sources, Sequence):
        normalized = tuple(sources)
    else:
        raise TypeError("sources must be a Coverage or a sequence of Coverage values.")
    if not normalized:
        raise ValueError(f"{operation}() requires at least one coverage source.")
    if not all(isinstance(source, Coverage) for source in normalized):
        raise TypeError("sources must contain only Coverage values.")

    resolution = normalized[0].resolution
    if any(source.resolution != resolution for source in normalized[1:]):
        raise ValueError("all coverage sources must use the same resolution.")
    segment_count = normalized[0].segment_count
    if any(source.segment_count != segment_count for source in normalized[1:]):
        raise ValueError(
            "all coverage sources must contain the same number of segments."
        )

    return (
        [_trusted_uint64(source.cells) for source in normalized],
        [_trusted_uint64(source.offsets) for source in normalized],
        resolution,
        segment_count,
        threshold,
        len(normalized),
    )


@overload
def occupancy(
    sources: Coverage | Sequence[Coverage],
    *,
    minimum_sources: int = 1,
    into: None = None,
) -> OccupancyRuns: ...


@overload
def occupancy(
    sources: Coverage | Sequence[Coverage],
    *,
    minimum_sources: int = 1,
    into: Stats,
) -> OccupancyStats: ...


def occupancy(
    sources: Coverage | Sequence[Coverage],
    *,
    minimum_sources: int = 1,
    into: Stats | None = None,
) -> OccupancyRuns | OccupancyStats:
    """Read aligned coverage segments as ordered occupancy bins.

    Each sequence entry is counted as one source. A cell is occupied in a
    segment when at least ``minimum_sources`` entries cover it. Callers must
    supply unique sources with identical, temporally adjacent bin boundaries
    when those semantics matter. The result stays ordinal; callers decide how
    boundaries map to physical time and how leading, trailing, or cyclic gaps
    should be treated.

    By default every maximal half-open run is kept, which costs memory in
    proportion to the run count; that approaches the hit count when cells are
    occupied briefly and repeatedly. ``into=Stats()`` instead accumulates
    per-cell counts and complete internal gaps in one pass, without building
    the runs at all. Keep the default when the boundaries themselves are the
    answer: percentiles, minimum-duration filtering, short-gap merging, or
    arbitrary per-run timestamps.
    """
    if into is not None and not isinstance(into, Stats):
        raise TypeError("into must be a Stats reducer, or None.")
    cells, offsets, resolution, segment_count, threshold, count = _prepared_sources(
        sources, minimum_sources, "occupancy"
    )
    native_threshold = min(threshold, count + 1)
    if isinstance(into, Stats):
        return OccupancyStats._from_native(
            _occupancy_stats(cells, offsets, resolution, native_threshold),
            resolution=resolution,
            segment_count=segment_count,
            minimum_sources=threshold,
            source_count=count,
        )
    result = _occupancy_runs(cells, offsets, resolution, native_threshold)
    return OccupancyRuns._from_native(
        cells=result[0],
        offsets=result[1],
        starts=result[2],
        stops=result[3],
        resolution=resolution,
        segment_count=segment_count,
        minimum_sources=threshold,
        source_count=count,
    )


def cell_centers(
    cells: int | Sequence[int] | npt.NDArray[np.integer[Any]],
    resolution: int,
) -> npt.NDArray[np.float64]:
    """Return unit-vector centers for HEALPix RING indices.

    The result has shape ``(cells, 3)``. Invalid resolutions, non-integer
    inputs, negative values, and out-of-range indices are rejected.
    """
    resolved = _as_resolution(resolution)
    ring = _as_uint64_vector(cells, "cells", native_range_checked=True)
    return _center(ring, resolved)


def cell_at(
    vectors_xyz: Sequence[float] | Sequence[Sequence[float]] | npt.ArrayLike,
    resolution: int,
) -> npt.NDArray[np.int64]:
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
    return _signed_view(_cell_at(vectors, resolved))


def cell_corners(
    cells: int | Sequence[int] | npt.NDArray[np.integer[Any]],
    resolution: int,
) -> npt.NDArray[np.float64]:
    """Return four unit-vector corners for HEALPix RING indices.

    The result has shape ``(cells, 4, 3)`` in boundary traversal order.
    Validation follows :func:`cell_centers`.
    """
    resolved = _as_resolution(resolution)
    ring = _as_uint64_vector(cells, "cells", native_range_checked=True)
    return _corner_many(ring, resolved)


__all__ = [
    "Count",
    "Coverage",
    "OccupancyRuns",
    "OccupancyStats",
    "Stats",
    "Sum",
    "__version__",
    "cell_at",
    "cell_centers",
    "cell_corners",
    "cell_count",
    "cover_cap",
    "cover_convex_polygon",
    "cover_sweep",
    "occupancy",
]
