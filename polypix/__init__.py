"""Fast region coverage on the HEALPix RING grid."""

from __future__ import annotations

import operator
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Never, Protocol, SupportsIndex, cast, overload

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
    _cover_prepared_regions,
    _cover_sweep,
    _neighbors,
    _prepare_polygon,
    _revisit_stats,
    _sum_coverage_per_cell,
    _validate_coverage,
)

_GEOMETRY_SHAPE_ERROR = (
    "geometry must have shape (vertices, 3), "
    "(polygons, vertices, 3), or be a sequence of (vertices, 3) arrays."
)
_MISSING = object()

# Spellings for the argument shapes that repeat across this module. They are
# not part of the public API - ``__all__`` is - and exist so that one signature
# reads as one line. They document intent; ``npt.ArrayLike`` already admits
# everything else, and the real shape and dtype checks happen on the way in.
CellsLike = int | Sequence[int] | npt.NDArray[np.integer[Any]]
# Offsets and imported cell arrays are always sequences; a bare scalar is not
# a meaningful segmentation, so they do not admit one.
OffsetsLike = Sequence[int] | npt.NDArray[np.integer[Any]]
VectorsLike = Sequence[float] | Sequence[Sequence[float]] | npt.ArrayLike
PolygonsLike = (
    Sequence[Sequence[float]] | Sequence[Sequence[Sequence[float]]] | npt.ArrayLike
)
EdgesLike = Sequence[Sequence[float]] | npt.ArrayLike
ValuesLike = float | Sequence[float] | npt.ArrayLike
CoverageMode = Literal["center", "overlap"]


class _GeoInterface(Protocol):
    @property
    def __geo_interface__(self) -> Mapping[str, object]: ...


_GeoLike = Mapping[str, object] | _GeoInterface


@dataclass(frozen=True, eq=False, init=False, slots=True)
class Polygon:
    """One spherical polygon outer boundary and any holes, in Cartesian XYZ.

    Coordinates are copied into read-only ``float64`` arrays and the geometry
    is validated immediately. Ring orientation does not matter.

    Parameters
    ----------
    outer : array_like
        The ``(vertices, 3)`` outer boundary.
    *holes : array_like
        Zero or more ``(vertices, 3)`` hole boundaries.

    Notes
    -----
    Coverage uses the geometry copied and validated at construction. The
    coordinate arrays are read-only to catch accidental writes; deliberately
    changing their NumPy flags and contents is unsupported and does not alter
    the prepared geometry.
    """

    outer: npt.NDArray[np.float64]
    holes: tuple[npt.NDArray[np.float64], ...]
    _prepared: object = field(repr=False)

    def __init__(self, outer: object, *holes: object) -> None:
        owned_outer = np.array(
            _as_float_matrix(outer, 3, "outer"), dtype=np.float64, order="C", copy=True
        )
        owned_holes = tuple(
            np.array(
                _as_float_matrix(hole, 3, f"holes[{index}]"),
                dtype=np.float64,
                order="C",
                copy=True,
            )
            for index, hole in enumerate(holes)
        )
        rings = (owned_outer, *owned_holes)
        offsets = np.concatenate(
            (
                np.zeros(1, dtype=np.uint64),
                np.cumsum([len(ring) for ring in rings], dtype=np.uint64),
            )
        )
        vertices = np.concatenate(rings) if rings else np.empty((0, 3), np.float64)
        prepared = _prepare_polygon(vertices, offsets)
        for ring in rings:
            _freeze_array(ring)
        object.__setattr__(self, "outer", owned_outer)
        object.__setattr__(self, "holes", owned_holes)
        object.__setattr__(self, "_prepared", prepared)

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return Polygon, (self.outer, *self.holes)


@dataclass(frozen=True, eq=False, init=False, slots=True)
class MultiPolygon:
    """A union of zero or more :class:`Polygon` components.

    The union is one input region and therefore one coverage segment. An empty
    ``MultiPolygon`` is a valid empty region.
    """

    polygons: tuple[Polygon, ...]

    def __init__(self, *polygons: Polygon) -> None:
        if not all(isinstance(polygon, Polygon) for polygon in polygons):
            raise TypeError("MultiPolygon components must be Polygon objects.")
        object.__setattr__(self, "polygons", polygons)

    def __len__(self) -> int:
        return len(self.polygons)

    def __iter__(self) -> Iterator[Polygon]:
        return iter(self.polygons)


RegionLike = (
    PolygonsLike
    | Polygon
    | MultiPolygon
    | _GeoLike
    | Sequence[Polygon | MultiPolygon | _GeoLike]
)


@dataclass(frozen=True, eq=False, init=False, slots=True)
class Coverage:
    """Segmented HEALPix RING coverage, validated and read-only.

    One segment is one input item: one polygon, cap, sweep interval, or cell.
    Every cell is stored in the flat ``cells`` array, and ``offsets`` records
    where each segment begins and ends. So ``len(coverage)`` counts input
    items and not cells, and ``coverage[i]`` is a zero-copy view of the
    cells found for item ``i``.

    The covering functions and :func:`cell_neighbors` return this type.
    Calling ``Coverage(...)`` directly raises :exc:`TypeError`; use
    :meth:`from_arrays` to bring segmented arrays back in from storage.

    Attributes
    ----------
    cells : ndarray
        Flat ``int64`` array of standard HEALPix RING indices, in input
        order.
    offsets : ndarray
        ``int64`` segment boundaries, of length ``len(coverage) + 1``.
    resolution : int
        The HEALPix resolution that every cell in the result belongs to.

    Notes
    -----
    Holding a Coverage is proof that its invariants hold: every cell is in
    range for the resolution, the offsets are nondecreasing and closed, and
    no cell repeats within a segment. That is why :func:`revisit` does not
    rescan what it is handed. The same cell may of course turn up in many
    different segments.

    Two derived arrays are easy to want, and we deliberately do not return
    them, since NumPy computes them no more slowly than we could::

        sizes = np.diff(coverage.offsets)
        segment_of_each_hit = np.repeat(np.arange(len(coverage)), sizes)

    Cell order within a segment is deterministic for a given build and
    platform, but it is not part of the interface, and we never sort a result
    for presentation alone. Imported segments keep the order they arrived in.

    Read-only means the arrays come back with ``WRITEABLE=False``. That
    catches accidental writes; it is not a promise of deep immutability
    against deliberate flag manipulation. Imported arrays are copied, so
    later edits to your own input cannot reach into a Coverage.

    Equality is identity-based, so comparing two large coverages never
    starts an implicit linear scan. Compare ``cells``, ``offsets``, and
    ``resolution`` yourself when you want value equality.
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
        cells: OffsetsLike,
        offsets: OffsetsLike,
        resolution: int,
    ) -> Coverage:
        """Copy and validate segmented RING-cell arrays from elsewhere.

        This is the way back in from storage. Coverage is usually the
        expensive product of a campaign, so the realistic shape of the work
        is to compute it once and query it many times afterwards. A reloaded
        coverage is an ordinary coverage: :func:`revisit` and
        :meth:`reduce` apply unchanged.

        Parameters
        ----------
        cells : array_like of int
            RING indices at ``resolution``, concatenated in segment order.
        offsets : array_like of int
            Segment boundaries. The sequence starts at 0, never decreases,
            and ends at the number of cells.
        resolution : int
            HEALPix resolution from 0 through 29.

        Returns
        -------
        Coverage
            A coverage owning validated copies of both arrays.

        Raises
        ------
        TypeError
            If either input is not an integer array.
        ValueError
            If a cell falls outside the grid, the offsets are not a closed
            nondecreasing sequence, or a cell repeats inside one segment.
        OverflowError
            If a value cannot be represented by the signed ``int64`` result
            arrays.

        Notes
        -----
        Validation costs one pass over the cells and segments, and an
        unsorted segment may need temporary storage. It happens once, here
        at the boundary, so nothing downstream has to re-check a hit.

        Examples
        --------
        >>> import polypix as px
        >>> coverage = px.Coverage.from_arrays(
        ...     cells=[2, 7, 9], offsets=[0, 2, 3], resolution=1
        ... )
        >>> len(coverage)
        2
        >>> coverage[1]
        array([9])
        """
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
        return self.offsets.size - 1

    def __getitem__(self, index: SupportsIndex) -> npt.NDArray[np.int64]:
        """Return a read-only view of one segment's cell IDs."""
        if isinstance(index, (bool, np.bool_)):
            raise TypeError("Coverage indices must be integers, not bool.")
        item = operator.index(index)
        count = len(self)
        if item < 0:
            item += count
        if item < 0 or item >= count:
            raise IndexError("Coverage segment index out of range.")
        start = int(self.offsets[item])
        stop = int(self.offsets[item + 1])
        return self.cells[start:stop]

    def reduce(
        self,
        reducer: CoverageReducer,
        *,
        cells: CellsLike | None = None,
    ) -> npt.NDArray[np.int64] | npt.NDArray[np.float64]:
        """Accumulate this coverage with a :class:`Count` or :class:`Sum`.

        Use this when several reductions share one expensive covering pass,
        or when the coverage came back from storage. The answer is the one a
        fused ``reduce=`` call on the covering function would have given.

        Parameters
        ----------
        reducer : Count or Sum
            The accumulation to perform.
        cells : int or array_like of int, optional
            RING indices to report. This plays the part ``candidate_cells``
            plays on a covering call: leave it out for a dense array indexed
            by cell ID, or name cells to get one value per requested ID.

        Returns
        -------
        ndarray
            With ``cells=None``, a dense array of length
            ``cell_count(resolution)`` indexed by RING cell ID. Otherwise
            one value per requested ID, in query order and including
            duplicates. :class:`Count` returns ``int64`` and :class:`Sum`
            returns ``float64``.

        Raises
        ------
        TypeError
            If ``reducer`` is not a :class:`Count` or :class:`Sum`, or
            ``cells`` is not an integer array.
        ValueError
            If a requested cell falls outside the grid, or a
            :class:`Sum` carries the wrong number of values.

        Notes
        -----
        Naming cells is the reason to reach for this method: it answers a
        question about a few thousand cells without ever allocating the
        grid. Small grids still accumulate through a dense scratch array
        internally, and large ones through a hash table, so memory follows
        what you asked for instead of the resolution.

        Without ``cells``, the array-level equivalent is a one-liner and
        usually faster than coming back through Polypix, so prefer it::

            np.bincount(coverage.cells, minlength=px.cell_count(resolution))

        The same call with ``weights=np.repeat(values, sizes)`` covers
        :class:`Sum`. That equivalence stops holding as soon as you name
        cells, because :func:`numpy.bincount` has to build the whole grid
        before it can index a few cells out of it. At resolution 13 that is
        six gibibytes to answer a question about one city.
        """
        if not isinstance(reducer, (Count, Sum)):
            raise TypeError("reducer must be a Count or Sum reducer.")
        requested = (
            None if cells is None else _coverage_cells(cells, self.resolution, "cells")
        )
        return _reduce_coverage(self, reducer, requested)


@dataclass(frozen=True, eq=False, init=False, slots=True)
class RevisitStats:
    """Per-cell revisit statistics on one thresholded axis.

    :func:`revisit` produces these; direct construction raises
    :exc:`TypeError`. Every field describes the same thresholded,
    source-unioned coverage of a cell, so they can be combined freely.
    Gaps and window bounds are measured in segments, not seconds, because a
    Coverage carries no clock.

    Attributes
    ----------
    cells : ndarray
        Ascending ``int64`` RING indices of the cells that were covered at
        least once.
    run_counts : ndarray
        ``int64`` number of separate visits per cell, at least one.
    internal_gap_steps_sum : ndarray
        ``int64`` total length of the gaps between visits, in segments.
        Zero for a cell visited once.
    maximum_internal_gap_steps : ndarray
        ``int64`` longest single gap between two visits, in segments. Zero
        for a cell visited once.
    first_start : ndarray
        ``int64`` segment index at which the cell was first covered.
    last_stop : ndarray
        ``int64`` segment index just after the cell was last covered.

    Notes
    -----
    All six arrays need the same single pass over the segment axis, and
    none of them can be recovered from the others afterwards.
    ``maximum_internal_gap_steps`` is the clearest case: an individual gap
    exists only in the moment one visit closes and the next opens, so
    computing it later would mean materializing every visit.

    Only complete gaps between two visits are counted. What happens at the
    ends of the timeline is a policy choice, so we report the window
    instead of choosing for you: the trailing gap is
    ``segment_count - last_stop`` against the segment count you passed in,
    and the number of internal gaps per cell is ``run_counts - 1``.

    Anything this result does not carry is either one NumPy expression away
    or already in your hands, since the resolution, segment count,
    threshold, and source count all came from the arguments you supplied.
    The arrays follow the same read-only and identity-equality rules as
    :class:`Coverage`, and ``len(stats)`` is the number of cells reported.
    """

    cells: npt.NDArray[np.int64]
    run_counts: npt.NDArray[np.int64]
    internal_gap_steps_sum: npt.NDArray[np.int64]
    maximum_internal_gap_steps: npt.NDArray[np.int64]
    first_start: npt.NDArray[np.int64]
    last_stop: npt.NDArray[np.int64]

    def __init__(self, *_args: Never, **_kwargs: Never) -> None:
        raise TypeError("RevisitStats values are constructed by Polypix.")

    @classmethod
    def _from_native(
        cls,
        arrays: tuple[npt.NDArray[np.uint64], ...],
    ) -> RevisitStats:
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
        return result

    def __len__(self) -> int:
        """Return the number of cells having at least one qualifying run."""
        return self.cells.size


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
    """Return the number of HEALPix cells at one resolution.

    The count is ``12 * 4 ** resolution``, since the grid starts from 12
    cells and splits each one into four at every step up. Use this when
    allocating or checking the length of a dense map.

    Parameters
    ----------
    resolution : int
        HEALPix resolution, 0 through 29.

    Returns
    -------
    int
        The number of cells in the grid.

    Raises
    ------
    TypeError
        If ``resolution`` is not an integer.
    ValueError
        If ``resolution`` falls outside 0 through 29.

    Examples
    --------
    >>> import polypix as px
    >>> px.cell_count(4)
    3072
    """
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
    return min(threads, int(np.iinfo(np.uintp).max))


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
    # ``dtype.kind`` classifies the buffer with one attribute read, where each
    # ``np.issubdtype`` call costs a type resolution. That matters here: on a
    # Polypix result array the rest of this function only reinterprets a view,
    # so the classification would otherwise dominate the conversion.
    kind = array.dtype.kind
    if kind == "b":
        raise TypeError(f"{name} must contain integers, not bool.")
    if array.size == 0:
        # An empty ordinary sequence has no values from which NumPy can infer
        # its intended integer dtype. Explicitly typed arrays do, so enforce
        # their dtype exactly as for a nonempty input.
        untyped_sequence = isinstance(values, Sequence) and not isinstance(
            values, np.ndarray
        )
        if kind not in ("i", "u") and not untyped_sequence:
            raise TypeError(f"{name} must contain integers.")
        return np.empty(0, dtype=np.uint64)
    if kind == "u":
        return _aligned(np.ascontiguousarray(array.astype(np.uint64, copy=False)))
    if kind == "i":
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
    if kind == "O":
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
    dense = _dense_polygons(_as_float_array(values, "geometry"))
    if dense is None:
        raise ValueError(_GEOMETRY_SHAPE_ERROR)
    return dense


def _name_ragged_failure(values: Sequence[object]) -> None:
    """Reraise a ragged batch failure against the entry that caused it."""
    for index, polygon in enumerate(values):
        _as_float_matrix(polygon, 3, f"geometry[{index}]")


def _ragged_polygons(
    values: Sequence[object],
    shapes: Sequence[tuple[int, ...]],
) -> tuple[np.ndarray, np.ndarray]:
    """Pack a sequence of differing-length polygons into vertices and offsets.

    There is no per-entry work left here: the shapes were already read to
    choose this path, and the offsets follow from them. Converting, the dtype
    check, and the width check all happen once over the concatenated buffer.
    That is safe because the entries have to agree for ``concatenate`` to
    succeed at all - a wrong rank or width raises there, and a wrong dtype
    survives to the whole-buffer check - so a shape that misreports its own
    vertex count is one that fails anyway.

    Anything that does fail reruns the per-entry conversion to name the
    offending index exactly, which costs nothing on the path that succeeds.
    Converting entry by entry up front instead cost about five microseconds
    each, which on large batches exceeded the covering work.
    """
    try:
        vertices = _as_float_matrix(
            np.concatenate(values, axis=0),  # type: ignore[arg-type]
            3,
            "geometry",
        )
    except (TypeError, ValueError):
        _name_ragged_failure(values)
        raise
    counts = np.fromiter(
        (shape[0] for shape in shapes), dtype=np.uint64, count=len(shapes)
    )
    if int(counts.sum()) != vertices.shape[0]:
        # No entry can hide vertices from ``concatenate``, so reaching this
        # means one reported a length its own data does not have.
        _name_ragged_failure(values)
        raise ValueError(_GEOMETRY_SHAPE_ERROR)
    offsets = np.concatenate(
        (np.zeros(1, dtype=np.uint64), np.cumsum(counts, dtype=np.uint64))
    )
    return vertices, offsets


def _as_polygons(
    values: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Accept one polygon, a uniform batch, or a batch of differing lengths."""
    if isinstance(values, Iterable) and not isinstance(
        values, (Sequence, np.ndarray, str, bytes, Mapping)
    ):
        raise TypeError(
            "non-sequence iterables are not accepted; pass list(geometry) for a "
            "geometry batch."
        )
    if not isinstance(values, Sequence) or isinstance(values, np.ndarray):
        array = np.asarray(values)
        if array.ndim == 1 and array.size == 0:
            return np.empty((0, 3), dtype=np.float64), np.zeros(1, dtype=np.uint64)
        # An ndarray means numeric coordinates here. Giving object arrays a
        # second meaning as geometry batches would make dispatch dtype-dependent.
        if array.ndim > 0 and array.dtype == np.dtype("O"):
            raise TypeError(
                "object-dtype arrays are not accepted; pass list(geometry)."
            )
        return _require_dense_polygons(array)
    if len(values) == 0:
        return np.empty((0, 3), dtype=np.float64), np.zeros(1, dtype=np.uint64)
    if np.ndim(values[0]) != 2:
        return _require_dense_polygons(values)
    # Identical shapes pack into one dense array. Anything else - differing
    # vertex counts, and differing widths, which would otherwise reach
    # ``np.asarray`` as a ragged nested sequence and leak its message - takes
    # the per-entry path, where a failure can be named. Comparing against the
    # first shape short-circuits on a ragged batch, where hashing every shape
    # into a set would not.
    shapes = [np.shape(polygon) for polygon in values]
    first = shapes[0]
    if len(first) == 2 and all(shape == first for shape in shapes):
        return _require_dense_polygons(values)
    return _ragged_polygons(values, shapes)


def _geo_mapping(value: object, name: str) -> Mapping[str, object] | None:
    mapping = getattr(value, "__geo_interface__", _MISSING)
    if mapping is _MISSING:
        return cast(Mapping[str, object], value) if isinstance(value, Mapping) else None
    if not isinstance(mapping, Mapping):
        raise TypeError(f"{name}.__geo_interface__ must be a mapping.")
    return mapping


def _geo_sequence(value: object, name: str) -> list[object]:
    if isinstance(value, (str, bytes, Mapping)):
        raise TypeError(f"{name} must be a sequence.")
    try:
        return list(cast(Iterable[object], value))
    except TypeError:
        raise TypeError(f"{name} must be a sequence.") from None


def _geo_ring_to_xyz(value: object, name: str) -> np.ndarray:
    shape_error = f"{name} must have shape (positions, 2) or (positions, 3)."
    try:
        coordinates = _as_float_array(value, name)
    except ValueError:
        raise ValueError(shape_error) from None
    if coordinates.ndim != 2 or coordinates.shape[1] not in (2, 3):
        raise ValueError(shape_error)
    if not np.isfinite(coordinates).all():
        raise ValueError(f"{name} must contain only finite coordinates.")
    longitude = coordinates[:, 0]
    latitude = coordinates[:, 1]
    if np.any((longitude < -180.0) | (longitude > 180.0)):
        raise ValueError(f"{name} longitude must be between -180 and 180 degrees.")
    if np.any((latitude < -90.0) | (latitude > 90.0)):
        raise ValueError(f"{name} latitude must be between -90 and 90 degrees.")
    longitude_rad = np.radians(longitude)
    latitude_rad = np.radians(latitude)
    radial = np.cos(latitude_rad)
    return np.column_stack(
        (
            radial * np.cos(longitude_rad),
            radial * np.sin(longitude_rad),
            np.sin(latitude_rad),
        )
    )


def _geo_polygon(value: object, name: str) -> Polygon | None:
    rings = _geo_sequence(value, name)
    if not rings:
        return None
    converted = [
        _geo_ring_to_xyz(ring, f"{name}[{index}]") for index, ring in enumerate(rings)
    ]
    try:
        return Polygon(converted[0], *converted[1:])
    except ValueError as error:
        raise ValueError(f"{name}: {error}") from None


def _geo_region(mapping: Mapping[str, object], name: str) -> Polygon | MultiPolygon:
    geometry_type = mapping.get("type")
    if not isinstance(geometry_type, str):
        raise TypeError(f"{name}['type'] must be a string.")
    if geometry_type == "Feature":
        if "geometry" not in mapping:
            raise ValueError(f"{name} is missing 'geometry'.")
        geometry = mapping["geometry"]
        if geometry is None:
            return MultiPolygon()
        if not isinstance(geometry, Mapping):
            raise TypeError(f"{name}['geometry'] must be a mapping or None.")
        nested = cast(Mapping[str, object], geometry)
        if nested.get("type") == "Feature":
            raise ValueError(f"{name} cannot contain another Feature.")
        return _geo_region(nested, f"{name}['geometry']")
    if geometry_type == "FeatureCollection":
        raise ValueError(
            f"{name} has unsupported geometry type 'FeatureCollection'; "
            "pass a sequence of geometries for a batch."
        )
    if geometry_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(
            f"{name} has unsupported geometry type {geometry_type!r}; "
            "expected Polygon, MultiPolygon, or Feature."
        )
    if "coordinates" not in mapping:
        raise ValueError(f"{name} is missing 'coordinates'.")
    coordinates = mapping["coordinates"]
    if geometry_type == "Polygon":
        return _geo_polygon(coordinates, f"{name}['coordinates']") or MultiPolygon()
    polygons = []
    for index, value in enumerate(_geo_sequence(coordinates, f"{name}['coordinates']")):
        polygon = _geo_polygon(value, f"{name}['coordinates'][{index}]")
        if polygon is not None:
            polygons.append(polygon)
    return MultiPolygon(*polygons)


def _as_prepared_regions(
    values: object,
) -> list[list[object]] | None:
    mapping = _geo_mapping(values, "geometry")
    if mapping is not None:
        regions = [_geo_region(mapping, "geometry")]
    elif isinstance(values, (Polygon, MultiPolygon)):
        regions = [values]
    elif isinstance(values, Sequence) and values:
        native_regions = []
        mappings = []
        unknown_index = None
        for index, value in enumerate(values):
            if isinstance(value, (Polygon, MultiPolygon)):
                native_regions.append(value)
            # Raw coordinate containers dominate large ragged batches. Avoid
            # paying for Mapping and geo-interface probes on every entry.
            elif type(value) in (np.ndarray, list, tuple):
                if unknown_index is None:
                    unknown_index = index
            elif (mapping := _geo_mapping(value, f"geometry[{index}]")) is not None:
                mappings.append((index, mapping))
            elif unknown_index is None:
                unknown_index = index
        if not native_regions and not mappings:
            return None
        if unknown_index is not None:
            raise TypeError(
                f"geometry[{unknown_index}] cannot be used in a structured geometry "
                "batch; pass only Polygon/MultiPolygon objects, only geo-interface "
                "objects, or only polygon arrays."
            )
        if native_regions and mappings:
            raise TypeError(
                "Do not mix geo-interface objects with Polygon objects in one batch."
            )
        regions = native_regions or [
            _geo_region(mapping, f"geometry[{index}]") for index, mapping in mappings
        ]
    else:
        return None

    polygons = [
        (region,) if isinstance(region, Polygon) else region.polygons
        for region in regions
    ]
    return [[polygon._prepared for polygon in region] for region in polygons]


@dataclass(frozen=True, eq=False, slots=True)
class Count:
    """Count how many segments cover each cell.

    Pass an instance as ``reduce=Count()`` to a covering function, or to
    :meth:`Coverage.reduce`. The reducer names the accumulation and nothing
    else: which cells are reported is decided by the operation being
    reduced, through ``candidate_cells`` when covering or ``cells`` when
    reducing a stored coverage.

    Examples
    --------
    >>> import numpy as np
    >>> import polypix as px
    >>> counts = px.cover_cap(
    ...     np.array([[0.0, 0.0, 1.0]]), 0.2, resolution=4, reduce=px.Count()
    ... )
    >>> int(counts.sum())
    24
    """


@dataclass(frozen=True, eq=False, slots=True)
class Sum:
    """Add one value per segment into every cell that segment covers.

    Reach for this when a segment contributes an exposure, a duration, a
    probability, or a capacity instead of a single count.

    Parameters
    ----------
    values : float or array_like of float
        One finite value shared by every segment, or one value per segment.

    Notes
    -----
    Floating-point addition runs in segment and hit order, so a repeated
    call on the same input returns the same sums. Cells are selected as
    described for :class:`Count`.

    Examples
    --------
    >>> import numpy as np
    >>> import polypix as px
    >>> weighted = px.cover_cap(
    ...     np.array([[0.0, 0.0, 1.0]]),
    ...     0.4,
    ...     resolution=2,
    ...     reduce=px.Sum([2.5]),
    ... )
    >>> float(weighted.sum())
    10.0
    """

    values: ValuesLike


CoverageReducer = Count | Sum


@overload
def cover_polygon(
    geometry: RegionLike,
    resolution: int,
    *,
    mode: CoverageMode = "center",
    candidate_cells: CellsLike | None = None,
    threads: int | None = None,
    reduce: None = None,
) -> Coverage: ...


@overload
def cover_polygon(
    geometry: RegionLike,
    resolution: int,
    *,
    mode: CoverageMode = "center",
    candidate_cells: CellsLike | None = None,
    threads: int | None = None,
    reduce: Count,
) -> npt.NDArray[np.int64]: ...


@overload
def cover_polygon(
    geometry: RegionLike,
    resolution: int,
    *,
    mode: CoverageMode = "center",
    candidate_cells: CellsLike | None = None,
    threads: int | None = None,
    reduce: Sum,
) -> npt.NDArray[np.float64]: ...


def cover_polygon(
    geometry: RegionLike,
    resolution: int,
    *,
    mode: CoverageMode = "center",
    candidate_cells: CellsLike | None = None,
    threads: int | None = None,
    reduce: CoverageReducer | None = None,
) -> Coverage | npt.NDArray[np.int64] | npt.NDArray[np.float64]:
    """Find the cells selected by each polygonal region.

    This is the shape of an imaging scene, a detector frame projected on the
    ground, or an area of interest. Simple array inputs may be convex or
    concave. Use :class:`Polygon` for holes and :class:`MultiPolygon` for a
    multipart region. Adjacent vertices use the shorter great-circle arc.

    Parameters
    ----------
    geometry : array_like, Polygon, MultiPolygon, mapping, or sequence
        One ``(vertices, 3)`` polygon, a dense ``(polygons, vertices, 3)``
        batch, or a sequence of ``(vertices, 3)`` arrays when the polygons
        have different vertex counts. A ``Polygon`` or ``MultiPolygon`` is
        one result segment; a sequence of those objects is a batch. An object
        implementing ``__geo_interface__``, or its mapping directly, may hold a
        ``Polygon``, ``MultiPolygon``, or one polygonal ``Feature`` in longitude
        and latitude degrees. Pass ``list(series)`` for a GeoPandas GeoSeries;
        its own geo-interface mapping is a FeatureCollection rather than one
        polygonal region. Vectors are Cartesian directions in one frame of your
        choosing, finite and nonzero; we normalize the magnitudes internally.
        The accepted geometry is described under :ref:`geometry-contract`.
    resolution : int
        HEALPix resolution, 0 through 29. Returned cells satisfy
        ``0 <= cell < cell_count(resolution)``.
    mode : {"center", "overlap"}, optional
        ``"center"`` selects a cell when its center is inside the region.
        ``"overlap"`` selects it when any part of its area intersects the
        region, including boundary tangency.
    candidate_cells : array_like of int, optional
        RING indices at ``resolution`` limiting which cells are tested.
        Duplicates and order are ignored. An empty selection
        returns empty segments without dropping any input item.
    threads : int, optional
        ``None`` picks the automatic policy, ``1`` runs sequentially, and a
        larger value sets the maximum size of the reusable worker pool,
        capped by the host.
    reduce : Count or Sum, optional
        Leave this out to get the segmented :class:`Coverage`. Pass a
        reducer to get its accumulated array instead, which also lets us
        skip building the cell lists where we can. See
        :ref:`choosing-a-reducer`.

    Returns
    -------
    Coverage or ndarray
        Without ``reduce``, one segment per input region. With a reducer,
        the array described under :class:`Count` and :class:`Sum`.

    Raises
    ------
    TypeError
        If the inputs have incompatible numeric types.
    ValueError
        If a shape, resolution, vector, candidate index, or the polygon
        geometry itself is invalid.
    MemoryError
        If the segmented result does not fit in memory. Consider batching
        the input, or a reducer if counts are what you are after.

    Notes
    -----
    The default ``mode="center"`` selects a cell when its center lies inside
    the polygon or exactly on its boundary. ``mode="overlap"`` instead uses
    the true curved HEALPix cell boundary and returns every cell the region
    touches. Adjacent regions can therefore share boundary cells.

    Geo-interface coordinates must be longitude and latitude in decimal
    degrees, interpreted directly as angles on a unit sphere, with optional
    altitude ignored. The datum and frame belong to the caller. Longitude is
    limited to ``[-180, 180]`` and latitude to ``[-90, 90]``. Polypix does not
    inspect or transform a CRS. Mapping properties, IDs, bounding boxes, and
    foreign members are ignored; non-polygonal geometries and collections are
    rejected.
    Supplied vertices retain Polypix's shorter great-circle edges rather than
    GeoJSON's planar longitude/latitude interpolation.

    An empty sequence, a one-dimensional empty array, and a dense
    ``(0, vertices, 3)`` array all describe a batch of zero polygons. A
    ``(0, 3)`` array is unambiguously one polygon with no vertices, and is
    rejected.

    An aligned, C-contiguous ``float64`` dense batch is the cheapest thing
    to hand over. Other real numeric arrays are converted once, and a
    ragged sequence is concatenated before the native call, which costs
    about what concatenating it yourself would. Strictly increasing
    candidate arrays are borrowed as they are; anything else is sorted and
    deduplicated internally.

    Contiguous arrays are borrowed for the duration of the call. Since the
    native kernel releases the GIL, do not mutate an input or candidate
    array from another thread before the call returns. Threading never
    changes membership, segment order, or cell order on one build and
    platform.

    Validation compares vertex pairs and tests every edge against every
    vertex, so its cost grows with the square of the vertex count. A
    densely sampled boundary is better handed to :func:`cover_sweep` in
    short segments than passed as one polygon with hundreds of vertices.

    Examples
    --------
    >>> import numpy as np
    >>> import polypix as px
    >>> scene = np.array(
    ...     [
    ...         [0.99, -0.10, -0.10],
    ...         [0.99, 0.10, -0.10],
    ...         [0.99, 0.10, 0.10],
    ...         [0.99, -0.10, 0.10],
    ...     ]
    ... )
    >>> coverage = px.cover_polygon(scene, resolution=6)
    >>> len(coverage)
    1
    """
    resolved = _as_resolution(resolution)
    overlap = _as_coverage_mode(mode)
    prepared_regions = _as_prepared_regions(geometry)
    reducer = _as_coverage_reducer(reduce)
    candidates, requested = _reduction_plan(candidate_cells, reducer, resolved)
    thread_count = _as_threads(threads)
    if prepared_regions is not None:
        native = _cover_prepared_regions(
            prepared_regions,
            resolved,
            candidates,
            reducer is None,
            thread_count,
            overlap,
        )
    else:
        vertices, ring_offsets = _as_polygons(geometry)
        native = _cover(
            vertices,
            ring_offsets,
            resolved,
            candidates,
            reducer is None,
            thread_count,
            overlap,
        )
    coverage = Coverage._from_native(*native, resolved)
    if reducer is None:
        return coverage
    return _reduce_coverage(coverage, reducer, requested)


@overload
def cover_cap(
    centers_xyz: VectorsLike,
    radii_rad: ValuesLike,
    resolution: int,
    *,
    mode: CoverageMode = "center",
    candidate_cells: CellsLike | None = None,
    threads: int | None = None,
    reduce: None = None,
) -> Coverage: ...


@overload
def cover_cap(
    centers_xyz: VectorsLike,
    radii_rad: ValuesLike,
    resolution: int,
    *,
    mode: CoverageMode = "center",
    candidate_cells: CellsLike | None = None,
    threads: int | None = None,
    reduce: Count,
) -> npt.NDArray[np.int64]: ...


@overload
def cover_cap(
    centers_xyz: VectorsLike,
    radii_rad: ValuesLike,
    resolution: int,
    *,
    mode: CoverageMode = "center",
    candidate_cells: CellsLike | None = None,
    threads: int | None = None,
    reduce: Sum,
) -> npt.NDArray[np.float64]: ...


def cover_cap(
    centers_xyz: VectorsLike,
    radii_rad: ValuesLike,
    resolution: int,
    *,
    mode: CoverageMode = "center",
    candidate_cells: CellsLike | None = None,
    threads: int | None = None,
    reduce: CoverageReducer | None = None,
) -> Coverage | npt.NDArray[np.int64] | npt.NDArray[np.float64]:
    """Find the cells selected by each spherical cap.

    A cap is a center direction plus an angular radius. It is the shape of a
    ground station's view of anything above an elevation mask, of a
    satellite's own service circle, and of any circular instantaneous field
    of view. Use this rather than approximating a circle with a many-sided
    polygon: the region here is the exact cap.

    Parameters
    ----------
    centers_xyz : array_like
        One ``(3,)`` Cartesian direction or a ``(caps, 3)`` batch, in one
        frame of your choosing. Vectors are finite and nonzero, and we
        normalize the magnitudes internally.
    radii_rad : float or array_like
        One finite angular radius in radians shared by every center, or
        exactly ``(caps,)`` radii, one per center. A length-one array is not
        broadcast. Radii lie in the closed interval ``[0, pi]``, where 0 is
        a point cap and pi is the whole sphere.
    resolution : int
        HEALPix resolution, 0 through 29.
    mode : {"center", "overlap"}, optional
        ``"center"`` selects a cell when its center is inside the cap.
        ``"overlap"`` selects it when any part of its area intersects the
        cap, including boundary tangency.
    candidate_cells : array_like of int, optional
        RING indices at ``resolution`` limiting which cells are tested.
        Duplicates and order are ignored.
    threads : int, optional
        ``None`` picks the automatic policy, ``1`` runs sequentially, and a
        larger value sets the maximum size of the reusable worker pool.
    reduce : Count or Sum, optional
        Leave this out to get the segmented :class:`Coverage`. Pass a
        reducer to get its accumulated array instead.

    Returns
    -------
    Coverage or ndarray
        Without ``reduce``, one segment per cap; a single ``(3,)`` center
        still comes back as one segment. With a reducer, its accumulated
        array.

    Raises
    ------
    TypeError
        If the inputs have incompatible numeric types.
    ValueError
        If a shape, vector, radius, resolution, or candidate index is
        invalid.
    MemoryError
        If the segmented result does not fit in memory. Use
        ``reduce=Count()`` if counts are the intended result.

    Notes
    -----
    The default ``mode="center"`` selects a cell when its center lies inside
    the cap or on its boundary. ``mode="overlap"`` instead returns every cell
    whose true curved area touches the cap, so even a point cap returns the
    cell containing that point.

    Counting center-selected caps is the one place where fusing the reduction
    into the geometry kernel currently wins by a wide margin, because the cap kernel
    accumulates private RING spans and never allocates the cap-cell pairs
    at all. It does this for a dense grid and for a selection alike, and
    falls back to covering once and counting when it judges that cheaper.
    Either way the answer is the same. Overlap mode uses the ordinary
    coverage-then-reduce path.

    An empty ``(0, 3)`` batch of centers accepts a scalar or empty radius
    array and returns a coverage with offsets ``[0]``, which keeps empty
    chunks composable. As with the other borrowed inputs, do not mutate a
    contiguous center, radius, or candidate array from another thread
    before the call returns.

    Examples
    --------
    >>> import numpy as np
    >>> import polypix as px
    >>> centers = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    >>> coverage = px.cover_cap(centers, np.radians([6.0, 4.0]), resolution=4)
    >>> np.diff(coverage.offsets)
    array([8, 4])
    """
    resolved = _as_resolution(resolution)
    overlap = _as_coverage_mode(mode)
    reducer = _as_coverage_reducer(reduce)
    centers = _as_cap_centers(centers_xyz)
    radii = _as_cap_radii(radii_rad, centers.shape[0])
    requested_threads = _as_threads(threads)
    candidates, requested = _reduction_plan(candidate_cells, reducer, resolved)
    if isinstance(reducer, Count) and not overlap:
        # Counting caps consumes analytic RING spans and never builds the cell
        # lists, so try it for the dense grid and for a selection alike. The
        # kernel declines by returning None when covering and counting wins.
        counts = _count_caps_per_cell(
            centers, radii, resolved, requested, requested_threads
        )
        if counts is not None:
            return counts
    coverage = Coverage._from_native(
        *_cover_cap(
            centers,
            radii,
            resolved,
            candidates,
            reducer is None,
            requested_threads,
            overlap,
        ),
        resolved,
    )
    if reducer is None:
        return coverage
    return _reduce_coverage(coverage, reducer, requested)


def _coverage_cells(
    values: CellsLike,
    resolution: int,
    name: str,
) -> npt.NDArray[np.uint64]:
    """Validate a positional reduction query, naming the public argument.

    The native reducers range-check this array too, but they know it as
    ``requested_cells``, so the message is produced here instead. One maximum
    classifies both failures: a reinterpreted negative index lands at or above
    ``1 << 63``, above every resolution's cell count.
    """
    cells = _as_uint64_vector(values, name, native_range_checked=True)
    if cells.size:
        largest = int(cells.max())
        if largest >= cell_count(resolution):
            if largest >= 1 << 63:
                raise ValueError(f"{name} must contain non-negative integers.")
            raise ValueError(
                f"{name} must contain valid RING indices at resolution {resolution}."
            )
    return cells


def _reduction_plan(
    candidate_cells: CellsLike | None,
    reducer: CoverageReducer | None,
    resolution: int,
) -> tuple[npt.NDArray[np.uint64] | None, npt.NDArray[np.uint64] | None]:
    """Validate ``candidate_cells`` and say whether it also restricts the scan.

    Without a reducer the selection *is* the result, so the kernel must restrict
    the scan to it. With one it fixes the output's index space instead, and
    restricting the scan becomes a free choice: a cell outside the selection
    contributes to no selected cell's count, and to no selected cell's sum
    either, so :class:`Sum` keeps both its value and its addition order. The
    selection is then passed as a hint, and the kernel takes it only while
    testing it is cheaper than scanning -- a comparison that needs the scan
    cost, which only the kernel can estimate.
    """
    if candidate_cells is None:
        return None, None
    if reducer is None:
        return _as_candidates(candidate_cells), None
    requested = _coverage_cells(candidate_cells, resolution, "candidate_cells")
    return requested, requested


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
    requested: npt.NDArray[np.uint64] | None,
) -> npt.NDArray[np.int64] | npt.NDArray[np.float64]:
    """Apply a coverage reducer to cell lists that are already built."""
    if isinstance(reducer, Count):
        if requested is not None and requested.size == 0:
            return np.empty(0, dtype=np.int64)
        return _count_coverage_per_cell(
            _trusted_uint64(coverage.cells),
            coverage.resolution,
            requested,
        )
    segment_values = _as_segment_values(reducer.values, len(coverage))
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


def _as_coverage_reducer(reducer: object) -> CoverageReducer | None:
    if reducer is None or isinstance(reducer, (Count, Sum)):
        return reducer
    raise TypeError("reduce must be a Count or Sum reducer, or None.")


def _as_coverage_mode(mode: object) -> bool:
    if mode == "center":
        return False
    if mode == "overlap":
        return True
    raise ValueError("mode must be 'center' or 'overlap'.")


@overload
def cover_sweep(
    left_edge_xyz: EdgesLike,
    right_edge_xyz: EdgesLike,
    resolution: int,
    *,
    mode: CoverageMode = "center",
    candidate_cells: CellsLike | None = None,
    threads: int | None = None,
    reduce: None = None,
) -> Coverage: ...


@overload
def cover_sweep(
    left_edge_xyz: EdgesLike,
    right_edge_xyz: EdgesLike,
    resolution: int,
    *,
    mode: CoverageMode = "center",
    candidate_cells: CellsLike | None = None,
    threads: int | None = None,
    reduce: Count,
) -> npt.NDArray[np.int64]: ...


@overload
def cover_sweep(
    left_edge_xyz: EdgesLike,
    right_edge_xyz: EdgesLike,
    resolution: int,
    *,
    mode: CoverageMode = "center",
    candidate_cells: CellsLike | None = None,
    threads: int | None = None,
    reduce: Sum,
) -> npt.NDArray[np.float64]: ...


def cover_sweep(
    left_edge_xyz: EdgesLike,
    right_edge_xyz: EdgesLike,
    resolution: int,
    *,
    mode: CoverageMode = "center",
    candidate_cells: CellsLike | None = None,
    threads: int | None = None,
    reduce: CoverageReducer | None = None,
) -> Coverage | npt.NDArray[np.int64] | npt.NDArray[np.float64]:
    """Cover the quadrilaterals between two sampled edges of a swath.

    A moving sensor sweeps out a swath. Sample its left and right edges at
    whatever cadence your propagator gives you, and every consecutive pair
    of samples becomes one quadrilateral,
    ``[left[i], right[i], right[i + 1], left[i + 1]]``, covered on its own.
    This paired-edge sweep is not the constant-colatitude operation
    traditionally called a HEALPix "strip".

    Parameters
    ----------
    left_edge_xyz, right_edge_xyz : array_like
        ``(samples, 3)`` arrays of Cartesian directions of equal length, in
        one frame of your choosing.
    resolution : int
        HEALPix resolution, 0 through 29.
    mode : {"center", "overlap"}, optional
        ``"center"`` selects a cell when its center is inside a segment.
        ``"overlap"`` selects it when any part of its area intersects the
        segment, including boundary tangency.
    candidate_cells : array_like of int, optional
        RING indices at ``resolution`` limiting which cells are tested.
        Duplicates and order are ignored.
    threads : int, optional
        ``None`` picks the automatic policy, ``1`` runs sequentially, and a
        larger value sets the maximum size of the reusable worker pool.
    reduce : Count or Sum, optional
        Leave this out to get the segmented :class:`Coverage`. Pass a
        reducer to get its accumulated array instead.

    Returns
    -------
    Coverage or ndarray
        Without ``reduce``, ``max(samples - 1, 0)`` segments. With a
        reducer, its accumulated array.

    Raises
    ------
    TypeError
        If the inputs have incompatible numeric types.
    ValueError
        If the two edges have different lengths, or a segment is invalid or
        encloses no area.
    MemoryError
        If the segmented result does not fit in memory.

    Notes
    -----
    Consecutive samples are joined by the shorter great-circle arc, which
    makes sampling density part of the input contract. Steps approaching
    180 degrees bow noticeably, steps past 180 degrees give you the
    opposite arc, and exactly ambiguous steps are rejected. We cannot tell
    a deliberate minor arc from an undersampled trajectory, so sample
    densely enough that each arc is the boundary you meant.

    Segments are independent, and we neither merge nor deduplicate them.
    A global ``np.unique(coverage.cells)`` forms the sorted union, at the
    cost of the segmentation and some sorting time and memory.

    The default ``mode="center"`` samples one point per HEALPix cell.
    ``mode="overlap"`` uses the true curved cell boundary and includes every
    cell a segment touches. A tiny boundary touch still contributes a whole
    hit to :class:`Count` and a whole value to :class:`Sum`.

    Zero or one paired sample describes no intervals and returns empty
    coverage with offsets ``[0]``. Repeating both edge samples at the same
    step encloses no area and is rejected; represent a stationary interval
    upstream instead, since deleting a sample shifts your time bins.
    Repeating a sample on one edge only is accepted, and gives a triangle
    pinched at that edge.

    Examples
    --------
    >>> import numpy as np
    >>> import polypix as px
    >>> track = np.radians(np.linspace(-13.0, 13.0, 7))
    >>> def edge(offset):
    ...     lat = np.radians(offset)
    ...     return np.stack(
    ...         [
    ...             np.cos(lat) * np.cos(track),
    ...             np.cos(lat) * np.sin(track),
    ...             np.full_like(track, np.sin(lat)),
    ...         ],
    ...         axis=-1,
    ...     )
    >>> coverage = px.cover_sweep(edge(3.2), edge(-3.2), resolution=4)
    >>> len(coverage)
    6
    """
    resolved = _as_resolution(resolution)
    overlap = _as_coverage_mode(mode)
    reducer = _as_coverage_reducer(reduce)
    left = _as_float_matrix(left_edge_xyz, 3, "left_edge_xyz")
    right = _as_float_matrix(right_edge_xyz, 3, "right_edge_xyz")
    if left.shape[0] != right.shape[0]:
        raise ValueError(
            "left_edge_xyz and right_edge_xyz must contain the same number of samples."
        )
    requested_threads = _as_threads(threads)
    candidates, requested = _reduction_plan(candidate_cells, reducer, resolved)
    coverage = Coverage._from_native(
        *_cover_sweep(
            left,
            right,
            resolved,
            candidates,
            reducer is None,
            requested_threads,
            overlap,
        ),
        resolved,
    )
    if reducer is None:
        return coverage
    return _reduce_coverage(coverage, reducer, requested)


def _as_minimum_sources(value: int) -> int:
    minimum_sources = _as_integer(value, "minimum_sources", "a positive integer")
    if minimum_sources < 1:
        raise ValueError("minimum_sources must be a positive integer.")
    return minimum_sources


def _prepared_timelines(
    timelines: Coverage | Sequence[Coverage],
    minimum_sources: int,
    operation: str,
) -> tuple[
    list[npt.NDArray[np.uint64]],
    list[npt.NDArray[np.uint64]],
    int,
    int,
    int,
]:
    """Validate aligned coverage timelines and view them for the native call."""
    threshold = _as_minimum_sources(minimum_sources)
    normalized: tuple[Coverage, ...]
    if isinstance(timelines, Coverage):
        normalized = (timelines,)
    elif isinstance(timelines, Sequence):
        normalized = tuple(timelines)
    else:
        raise TypeError(
            "timelines must be a Coverage or a sequence of Coverage values."
        )
    if not normalized:
        raise ValueError(f"{operation}() requires at least one timeline.")
    if not all(isinstance(source, Coverage) for source in normalized):
        raise TypeError("timelines must contain only Coverage values.")

    resolution = normalized[0].resolution
    if any(source.resolution != resolution for source in normalized[1:]):
        raise ValueError("all timelines must use the same resolution.")
    segment_count = len(normalized[0])
    if any(len(source) != segment_count for source in normalized[1:]):
        raise ValueError("all timelines must contain the same number of segments.")

    return (
        [_trusted_uint64(source.cells) for source in normalized],
        [_trusted_uint64(source.offsets) for source in normalized],
        resolution,
        threshold,
        len(normalized),
    )


def revisit(
    timelines: Coverage | Sequence[Coverage],
    *,
    minimum_sources: int = 1,
) -> RevisitStats:
    """Summarize how often each cell is covered across ordered bins.

    Read the segments of a coverage as a timeline instead of a batch. A
    cell is covered in a bin when at least ``minimum_sources`` of the
    timelines contain it there, and consecutive covered bins form one
    visit. What comes back is the per-cell visit count, the gaps between
    visits, and the bounds of the observed window.

    Parameters
    ----------
    timelines : Coverage or sequence of Coverage
        One coverage per source, each of whose segments are consecutive,
        temporally adjacent bins in ascending order. Every entry must share
        a resolution and a segment count.
    minimum_sources : int, default 1
        How many sources must cover a cell in the same bin for it to count.
        A threshold above the number of sources returns an empty result.

    Returns
    -------
    RevisitStats
        Ascending qualifying cells, with one statistic per cell.

    Raises
    ------
    TypeError
        If ``timelines`` holds something other than coverages.
    ValueError
        If the sequence is empty, the threshold is not positive, or the
        sources disagree on resolution or segment count.

    Notes
    -----
    Two things we cannot check for you, because a :class:`Coverage` carries
    no clock. Matching segment indices must describe identical bin
    boundaries, and consecutive bins must really be adjacent in time. Split
    the analysis at a discontinuity, or insert a deliberately empty
    separator bin, so that a visit cannot bridge one.

    Bins are ordinal, so map ``first_start`` and ``last_stop`` through your
    own array of time edges afterwards, and choose there whether revisit
    means end-to-start, start-to-start, or something cyclic.

    Counts and gaps accumulate in a single pass, and the result is
    allocated by cell rather than by visit. That matters because the visits
    themselves are not guaranteed to be smaller than the input: a cell hit
    in alternating bins produces one visit per hit. The result is sparse
    and sorted by cell ID, so a high resolution does not force a dense
    global allocation.

    Sequence positions are counted independently, so source uniqueness is
    yours to guarantee, and thresholding several sources deliberately drops
    source identity. Call this once per source when you need to know which
    observer saw what. For a sampled sweep, these are occupied bins rather
    than exact access events, and the boundary times are uncertain at the
    sampling cadence.

    Examples
    --------
    Four bins, in which one cell is covered, missed, then covered twice:

    >>> import polypix as px
    >>> timeline = px.Coverage.from_arrays(
    ...     cells=[5, 5, 5], offsets=[0, 1, 1, 2, 3], resolution=0
    ... )
    >>> stats = px.revisit(timeline)
    >>> stats.cells, stats.run_counts, stats.maximum_internal_gap_steps
    (array([5]), array([2]), array([1]))
    >>> stats.first_start, stats.last_stop
    (array([0]), array([4]))
    """
    cells, offsets, resolution, threshold, count = _prepared_timelines(
        timelines, minimum_sources, "revisit"
    )
    return RevisitStats._from_native(
        _revisit_stats(cells, offsets, resolution, min(threshold, count + 1))
    )


def cell_centers(
    cells: CellsLike,
    resolution: int,
) -> npt.NDArray[np.float64]:
    """Return the center direction of each HEALPix cell.

    Parameters
    ----------
    cells : int or array_like of int
        RING indices at ``resolution``.
    resolution : int
        HEALPix resolution, 0 through 29.

    Returns
    -------
    ndarray
        Shape ``(cells, 3)``, dtype ``float64``, unit vectors. A scalar
        cell returns ``(1, 3)`` and an empty input returns ``(0, 3)``.

    Raises
    ------
    TypeError
        If ``cells`` is not an integer array.
    ValueError
        If the resolution is invalid, or an index is negative or off the
        grid.

    Notes
    -----
    These are the cells' own centers, not the directions you started from.
    Only a cell center round-trips exactly through :func:`cell_at`. Large
    arrays are parallelized inside the native kernel, which is why there is
    no threading argument here.

    Examples
    --------
    >>> import polypix as px
    >>> px.cell_centers(0, resolution=0).round(3)
    array([[0.527, 0.527, 0.667]])
    """
    resolved = _as_resolution(resolution)
    ring = _as_uint64_vector(cells, "cells", native_range_checked=True)
    return _center(ring, resolved)


def cell_at(
    vectors_xyz: VectorsLike,
    resolution: int,
) -> npt.NDArray[np.int64]:
    """Return the HEALPix cell that each direction falls in.

    Use this when your input is points rather than regions: a ground track,
    a set of targets, individual pointings.

    Parameters
    ----------
    vectors_xyz : array_like
        One ``(3,)`` Cartesian direction or a ``(vectors, 3)`` batch.
        Vectors are finite and nonzero; magnitudes are ignored.
    resolution : int
        HEALPix resolution, 0 through 29.

    Returns
    -------
    ndarray
        Shape ``(vectors,)``, dtype ``int64``. A single ``(3,)`` vector
        returns shape ``(1,)`` and an empty ``(0, 3)`` batch returns shape
        ``(0,)``.

    Raises
    ------
    TypeError
        If the input is not a real numeric array.
    ValueError
        If the shape, a vector, or the resolution is invalid.

    Notes
    -----
    Every finite nonzero direction lands in exactly one cell, so this
    quantizes rather than approximating: it does not turn center-sampled
    region coverage into a conservative spatial index.

    A direction sitting numerically on a cell edge or vertex is a
    floating-point tie. The answer is repeatable for the same input, build,
    and platform, but we do not promise which of the adjacent cells owns an
    exact boundary direction across platforms. Resolve the tie upstream if
    your application needs a portable policy.

    Large batches are parallelized inside the native kernel, and
    contiguous inputs are borrowed while the GIL is released, so do not
    mutate them concurrently.

    Examples
    --------
    >>> import numpy as np
    >>> import polypix as px
    >>> cells = px.cell_at(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), 4)
    >>> px.cell_at(px.cell_centers(cells, 4), 4)
    array([1504,    0])
    """
    resolved = _as_resolution(resolution)
    vectors = _as_float_array(vectors_xyz, "vectors_xyz")
    if vectors.ndim == 1 and vectors.shape == (3,):
        vectors = vectors.reshape(1, 3)
    elif vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("vectors_xyz must have shape (3,) or (vectors, 3).")
    return _signed_view(_cell_at(vectors, resolved))


def cell_corners(
    cells: CellsLike,
    resolution: int,
) -> npt.NDArray[np.float64]:
    """Return the four corner directions of each HEALPix cell.

    Parameters
    ----------
    cells : int or array_like of int
        RING indices at ``resolution``.
    resolution : int
        HEALPix resolution, 0 through 29.

    Returns
    -------
    ndarray
        Shape ``(cells, 4, 3)``, dtype ``float64``, unit vectors in
        boundary traversal order. A scalar cell keeps the leading axis and
        returns ``(1, 4, 3)``. The first corner is not repeated at the end.

    Raises
    ------
    TypeError
        If ``cells`` is not an integer array.
    ValueError
        If the resolution is invalid, or an index is negative or off the
        grid.

    Notes
    -----
    HEALPix cell edges are curved, and we do not sample them between the
    corners. Four corners are therefore not a boundary: do not round-trip
    them as an exact great-circle polygon for the cell.
    """
    resolved = _as_resolution(resolution)
    ring = _as_uint64_vector(cells, "cells", native_range_checked=True)
    return _corner_many(ring, resolved)


def cell_neighbors(
    cells: CellsLike,
    resolution: int,
) -> Coverage:
    """Return the topological neighbors of each HEALPix cell.

    Cells touching at an edge or corner are neighbors. The input cell itself
    is excluded. Most cells have eight neighbors. The 24 cells meeting at
    eight exceptional grid vertices have seven, and cells at resolution 0
    have six.

    Parameters
    ----------
    cells : int or array_like of int
        RING indices at ``resolution``.
    resolution : int
        HEALPix resolution, 0 through 29.

    Returns
    -------
    Coverage
        One unordered segment per input cell, preserving input alignment.
        A scalar input produces one segment.

    Raises
    ------
    TypeError
        If ``cells`` is not an integer array.
    ValueError
        If the resolution is invalid, or an index is negative or off the
        grid.

    Examples
    --------
    >>> import polypix as px
    >>> neighbors = px.cell_neighbors(4, resolution=0)
    >>> sorted(neighbors[0].tolist())
    [0, 3, 5, 7, 8, 11]
    """
    resolved = _as_resolution(resolution)
    ring = _as_uint64_vector(cells, "cells", native_range_checked=True)
    return Coverage._from_native(*_neighbors(ring, resolved), resolved)


__all__ = [
    "Count",
    "Coverage",
    "MultiPolygon",
    "Polygon",
    "RevisitStats",
    "Sum",
    "__version__",
    "cell_at",
    "cell_centers",
    "cell_corners",
    "cell_count",
    "cell_neighbors",
    "cover_cap",
    "cover_polygon",
    "cover_sweep",
    "revisit",
]
