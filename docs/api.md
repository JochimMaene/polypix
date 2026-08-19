# API reference

```python
import polypix as px
```

For task-oriented examples see [Getting started](guide.md); for allocation
and threading trade-offs see [Performance and memory](performance.md); for
HEALPix and NumPy handoff conventions see
[Handing cells to other libraries](concepts.md#handing-cells-to-other-libraries).

| Object | Purpose |
| --- | --- |
| [`cover_convex_polygon`](#cover_convex_polygon) | Cover convex spherical polygons |
| [`cover_cap`](#cover_cap) | Cover exact spherical caps |
| [`cover_sweep`](#cover_sweep) | Cover the quadrilaterals between two sampled edges |
| [`count_coverage_per_cell`](#count_coverage_per_cell) | Count segmented memberships per cell |
| [`sum_coverage_per_cell`](#sum_coverage_per_cell) | Accumulate one value per segment into covered cells |
| [`count_caps_per_cell`](#count_caps_per_cell) | Fused cap-count accelerator |
| [`occupancy_runs`](#occupancy_runs) | Extract complete ordinal occupied-bin runs |
| [`cell_at`](#cell_at) | Direction vectors to RING cell IDs |
| [`cell_centers`](#cell_centers) | Cell center vectors |
| [`cell_corners`](#cell_corners) | Cell corner vectors |
| [`cell_count`](#cell_count) | Number of cells at a resolution |
| [`Coverage`](#coverage) | Segmented result of the coverage calls |
| [`OccupancyRuns`](#occupancyruns) | Sparse cell-major ordinal windows |

## cover_convex_polygon

```python
cover_convex_polygon(
    polygons_xyz,
    resolution,
    *,
    vertex_offsets=None,
    candidate_cells=None,
    threads=None,
)
```

Cover convex spherical polygons by HEALPix cell-center inclusion.

**Parameters**

`polygons_xyz`
: *array_like or sequence of array_like*. One `(vertices, 3)` polygon, a
  dense `(polygons, vertices, 3)` batch, or a sequence of `(vertices, 3)`
  arrays for a ragged batch. Vectors are finite, nonzero, and expressed in
  one caller-defined Cartesian frame; magnitudes are normalized internally.

`resolution`
: *int*. HEALPix resolution, 0 through 29. Returned cells satisfy `0 <= cell
  < 12 * 4 ** resolution`.

`vertex_offsets`
: *array_like of int, optional*. Boundaries for a packed ragged batch in one
  `(total_vertices, 3)` array. Starts at zero and ends at `total_vertices`.
  Do not combine it with dense or sequence-style polygon grouping.

`candidate_cells`
: *array_like of int, optional*. RING indices at `resolution` restricting
  which centers are tested. Set semantics; duplicates are ignored. An empty
  set returns empty segments without dropping input items.

`threads`
: *int, optional*. `None` selects the automatic policy, `1` is sequential,
  and larger values set the reusable worker-pool maximum, capped by the
  host.

**Returns**

`Coverage`
: Flat RING indices with offsets delimiting one segment per input polygon.

**Raises**

`TypeError`
: Inputs have incompatible numeric types.

`ValueError`
: Invalid shapes, resolution, vectors, or polygon geometry.

`MemoryError`
: The explicit segmented result cannot be materialized.

**Notes**

An empty sequence, a one-dimensional empty array, or a dense
`(0, vertices, 3)` array is a batch of zero polygons. A `(0, 3)` array is
unambiguously one polygon with zero vertices and is rejected.

Contiguous NumPy arrays are borrowed for the duration of the call. Because the
native kernel releases the GIL, do not mutate an input or candidate array from
another thread before the call returns.

A C-contiguous `float64` dense batch is the lowest-overhead input form.
Non-contiguous or other real numeric arrays are converted once; ragged
sequences are validated and concatenated before the native call.

Strictly increasing candidate arrays use a borrowed fast path; other inputs are
sorted and deduplicated internally.

Threading does not change membership, segment order, or cell order on the same
build and platform.

See [Geometry contract](#geometry-contract) for the accepted polygons and
[Restricting coverage to known cells](concepts.md#restricting-coverage-to-known-cells)
for the performance trade-offs.

## cover_cap

```python
cover_cap(centers_xyz, radii_rad, resolution, *, candidate_cells=None, threads=None)
```

Cover exact spherical caps by HEALPix cell-center inclusion.

**Parameters**

`centers_xyz`
: *array_like*. One `(3,)` Cartesian direction vector or a `(caps, 3)` batch
  in one caller-defined frame. Finite nonzero vectors are normalized
  internally.

`radii_rad`
: *float or array_like*. One finite angular radius shared by every center
  when passed as a scalar, or exact shape `(caps,)` with one radius per
  center. A length-one array is not broadcast. Radii are in radians and must
  lie in the closed interval `[0, pi]`.

`resolution`, `candidate_cells`, `threads`
: As in [`cover_convex_polygon`](#cover_convex_polygon).

**Returns**

`Coverage`
: One segment per input cap. A single `(3,)` center retains one segment.

**Raises**

`TypeError`
: Inputs have incompatible numeric types.

`ValueError`
: Shapes, vectors, radii, resolution, or candidates are invalid.

`MemoryError`
: The explicit segmented result cannot be materialized; use
  `count_caps_per_cell()` if counts are the intended result.

**Notes**

This covers the exact spherical region, sampled at cell centers. It does not
test whether the area of a HEALPix cell intersects a cap. Radius zero and pi
mean a point cap and the complete sphere respectively.

An empty `(0, 3)` center batch accepts a scalar or empty radius array and
returns a `Coverage` with offsets `[0]`. As with other borrowed inputs, do not
mutate compatible contiguous center, radius, or candidate arrays from another
thread before the call returns.

Use this function instead of approximating a circular field of view with a
many-sided polygon. When only the number of caps covering each cell matters,
[`count_caps_per_cell`](#count_caps_per_cell) avoids materializing repeated
cap-cell IDs.

## count_coverage_per_cell

```python
count_coverage_per_cell(coverage, *, cells=None)
```

Count how many segments of an existing `Coverage` contain each cell. Because a
validated segment contains each cell at most once, the result counts segment
membership rather than duplicate IDs.

`coverage`
: *Coverage*. Segmented membership from any geometry or an imported result.

`cells`
: *int or array_like of int, optional*. Positional RING IDs to query. Order and
  duplicates are preserved. With `None`, return the complete dense grid.

**Returns**

`ndarray`
: `int64`, with length `cell_count(coverage.resolution)` in dense mode or the
  requested-cell count in positional mode.

The native reducer scans the flat membership array directly. Selected-cell mode
indexes the requested IDs and does not allocate a dense global grid, making it
safe for sparse high-resolution analysis.

## sum_coverage_per_cell

```python
sum_coverage_per_cell(coverage, values, *, cells=None)
```

Add one constant value for every segment to every cell in that segment. This is
the generic exposure, dwell, probability, and capacity primitive.

`coverage`
: *Coverage*. Segmented membership from any geometry.

`values`
: *float or array_like*. One finite scalar shared by every segment, or one
  finite value per segment.

`cells`
: As in [`count_coverage_per_cell`](#count_coverage_per_cell).

**Returns**

`ndarray`
: `float64`, dense or positional according to `cells`.

The native implementation reads the segment offsets directly and never expands
`values` into one temporary value per hit.

## count_caps_per_cell

```python
count_caps_per_cell(centers_xyz, radii_rad, resolution, *, cells=None, threads=None)
```

Count the input caps containing each HEALPix cell center. Center and radius
inputs follow [`cover_cap`](#cover_cap).

**Parameters**

`centers_xyz`, `radii_rad`, `resolution`, `threads`
: As in [`cover_cap`](#cover_cap).

`cells`
: *int or array_like of int, optional*. Positional RING IDs to query. A
  scalar returns shape `(1,)`; an empty input returns shape `(0,)`. Order
  and duplicates are preserved.

**Returns**

`ndarray`
: C-contiguous `int64` counts. With `cells=None`, its length is `12 *
  4**resolution`; otherwise it matches the positional query length.

**Raises**

`TypeError`
: Inputs have incompatible numeric types.

`ValueError`
: Shapes, vectors, radii, resolution, or cell IDs are invalid.

`MemoryError`
: A requested dense result cannot be materialized; pass `cells=` to use
  positional query mode instead.

With `cells=None`, the result is a dense `int64` array of length
`12 * 4**resolution`, indexed by RING cell ID. Internally, contiguous RING spans
are accumulated directly, so cap-cell pairs are never materialized.

With `cells` supplied, the result has one value per requested RING ID in the
original order. Unlike `candidate_cells`, this is positional query input:
duplicates are retained and produce repeated equal values. Query mode is the
memory-safe choice at high resolutions, but its work grows with both the cap
count and requested-cell count; it is intended for genuinely sparse queries.
A dense resolution-12 result alone is about 1.5 GiB.

The predicate is evaluated at each requested HEALPix center. If the IDs came
from `cell_at(site_directions, resolution)`, this does not test the original
site directions.

An empty cap batch returns zeros in either mode. Compatible contiguous inputs
are borrowed while the native kernel runs without the GIL; do not mutate them
concurrently.

The operation is exactly equivalent to the following, without its potentially
large intermediate `Coverage`:

```python
import numpy as np

coverage = px.cover_cap(centers_xyz, radii_rad, resolution)
expected = px.count_coverage_per_cell(coverage)
```

## cover_sweep

```python
cover_sweep(left_edge_xyz, right_edge_xyz, resolution, *, candidate_cells=None, threads=None)
```

Cover the quadrilateral segments between two sampled spherical edges by
HEALPix cell-center inclusion.

This paired-edge sweep is not the constant-colatitude operation traditionally
called a HEALPix "strip".

**Parameters**

`left_edge_xyz`, `right_edge_xyz`
: *array_like*. `(samples, 3)` vector arrays of equal length.

`resolution`, `candidate_cells`, `threads`
: As in [`cover_convex_polygon`](#cover_convex_polygon).

**Returns**

`Coverage`
: For `N` paired samples, `N - 1` segments, where segment `i` covers the
  quadrilateral `[left[i], right[i], right[i+1], left[i+1]]`.

**Raises**

`TypeError`
: Inputs have incompatible numeric types.

`ValueError`
: Mismatched edge lengths or an invalid or zero-area segment.

`MemoryError`
: The explicit segmented result cannot be materialized.

**Notes**

Segments are independent. Polypix does not merge or deduplicate them. A global
`np.unique(coverage.cells)` forms a sorted union but destroys interval
segmentation and requires additional sorting memory and time.

Zero or one paired sample describes zero intervals and returns empty coverage
with offsets `[0]`. This makes empty chunks composable in vectorized workflows.

Repeating both edge samples at the same step gives a zero-area segment and is
rejected. A zero-motion interval must be represented upstream; deleting a
sample can change time-bin alignment. Repeating a sample on one edge only is
accepted and gives a triangle pinched at that edge.

Consecutive samples are joined by minor great-circle arcs, so sampling density
is part of the input contract. See
[Batches and segments](concepts.md#batches-and-segments).

## occupancy_runs

```python
occupancy_runs(sources, *, minimum_sources=1)
```

Extract complete, maximal ordinal occupied-bin runs from one `Coverage`, or from
aligned coverage belonging to multiple sources. A cell qualifies in a segment
when at least `minimum_sources` sequence entries contain it.

All sources must share a resolution and segment count. The caller must also
ensure that matching segment indices have identical bin boundaries and that
consecutive indices are temporally adjacent. Polypix cannot infer either fact
from `Coverage`; split the analysis at a discontinuity or insert an explicitly
empty separator bin. The result retains half-open
`[start, stop)` windows rather than choosing a particular revisit statistic.
Callers can therefore calculate end-to-start or start-to-start gaps, maximum or
percentile gaps, finite-horizon edge gaps, or cyclic gaps without information
having already been discarded.

**Parameters**

`sources`
: *Coverage or nonempty sequence of Coverage*. One segmented result per
  source entry, all with the same resolution and segment count. Sequence
  positions are counted independently, so source uniqueness is a caller
  responsibility.

`minimum_sources`
: *int, default 1*. Positive number of simultaneous source entries required
  for a cell to qualify. A threshold larger than the source count returns an
  empty result.

**Returns**

`OccupancyRuns`
: Sparse ascending cells with chronological half-open runs grouped by cell.

**Raises**

`TypeError`
: `sources` contains another type or an invalid resolution.

`ValueError`
: The sequence is empty, the threshold is not positive, or source grids or
  segment counts differ.

**Notes**

Coverage arrays are read-only and borrowed while the native reducer runs
without the GIL. To retain observer attribution, call `occupancy_runs()` once
per source; the multi-source form intentionally returns thresholded occupancy.
For sampled sweeps these are occupied-bin runs, not exact continuous access
events; boundary times are uncertain at the input sampling scale.

## cell_at

```python
cell_at(vectors_xyz, resolution)
```

Return the HEALPix RING cell containing each Cartesian direction.

**Parameters**

`vectors_xyz`
: *array_like*. One `(3,)` vector or a `(vectors, 3)` batch. Vectors must be
  finite and nonzero; magnitudes are ignored.

`resolution`
: *int*. HEALPix resolution, 0 through 29.

**Returns**

`ndarray`
: Shape `(vectors,)`, dtype `int64`. A single `(3,)` vector returns shape
  `(1,)`; an empty `(0, 3)` batch returns shape `(0,)`.

**Raises**

`TypeError`
: Incompatible numeric input.

`ValueError`
: Invalid shape, vector, or resolution.

The operation quantizes each direction to one cell. The exact center round trip
is:

```python
round_trip_cells = px.cell_at(px.cell_centers(cells, 8), 8)
```

`cell_centers(cell_at(directions))` returns representative cell centers rather than
the original directions. Every finite nonzero input is assigned to one cell,
but a direction numerically on or extremely near a mathematical cell edge or
vertex is subject to floating-point tie behavior. The result is repeatable for
the same input, build, and platform; the API does not promise which adjacent
cell owns an exact boundary direction across platforms. Applications that need
a portable tie policy should resolve it upstream. This maps points to cells; it
does not make center-selected region coverage a conservative spatial index.
Large batches are parallelized inside the native kernel. Compatible contiguous
inputs are borrowed while the GIL is released; do not mutate them concurrently.

## cell_centers

```python
cell_centers(cells, resolution)
```

Return unit-vector centers for HEALPix RING indices.

**Parameters**

`cells`
: *int or array_like of int*. RING indices at `resolution`.

`resolution`
: *int*. HEALPix resolution, 0 through 29.

**Returns**

`ndarray`
: Shape `(cells, 3)`, dtype `float64`. A scalar cell returns `(1, 3)`; empty
  input returns `(0, 3)`.

**Raises**

`TypeError`
: Non-integer input.

`ValueError`
: Invalid resolution, negative values, or out-of-range indices.

**Notes**

Large arrays are parallelized inside the native kernel. This adds no threading
argument to the supporting utilities.

## cell_corners

```python
cell_corners(cells, resolution)
```

Return the four unit-vector corners of each HEALPix cell, in boundary traversal
order.

The curved cell edges are not sampled between these corners. Do not treat the
four returned points as an exact great-circle polygon for the cell.

**Parameters**: as in [`cell_centers`](#cell_centers).

**Returns**

`ndarray`
: Shape `(cells, 4, 3)`, dtype `float64`. A scalar cell retains the leading
  axis and returns `(1, 4, 3)`. The first corner is not repeated.

**Raises**: as in [`cell_centers`](#cell_centers).

## cell_count

```python
cell_count(resolution)
```

Return `12 * 4**resolution` after applying the same integer and range
validation as every grid operation.

## Coverage

```python
Coverage.from_arrays(cells, offsets, resolution)
```

Coverage calls return this type directly. `from_arrays()` is the validating
interchange constructor for imported segmented arrays: it copies its inputs,
checks offsets, cell ranges, and within-segment uniqueness, then makes both
arrays read-only. Direct `Coverage(...)` construction is intentionally
disabled. Validation is linear in cells plus segments and may use temporary
storage for an unsorted segment. Native coverage calls wrap their newly owned
arrays without copying.

```python
coverage = px.Coverage.from_arrays(
    cells=[2, 7, 9],
    offsets=[0, 2, 3],
    resolution=1,
)
```

**Attributes**

`cells`
: *ndarray*. Flat one-dimensional `int64` array of standard HEALPix RING
  indices.

`offsets`
: *ndarray*. `int64` segment boundaries, length `segment_count + 1`.

`resolution`
: *int*. The resolution shared by every returned cell.

`segment_count`
: *int*. Number of input items, equal to `len(offsets) - 1` and
  `len(coverage)`.

`segment_sizes`
: *ndarray*. Newly allocated `int64` array equal to `np.diff(offsets)`.

`segment_indices()`
: Return one `int64` segment index aligned with every flat cell hit.

`filter_hits(mask)`
: Return new coverage with a boolean mask applied to flat hits while retaining
  the original number and ordering of segments.

Indexing returns a zero-copy, read-only view of one segment and supports
negative integer indices:

```python
cells_for_item = coverage[i]
```

**Notes**

Segments preserve input order and contain no duplicate cells. Cell order within
a segment is deterministic for a given build and platform but is not part of the
API; Polypix never sorts for presentation alone.

Duplicate cells across different segments are valid. Duplicate cells within
one segment are not. Imported segments retain their supplied cell order.

`Coverage` uses identity equality. Compare `cells`, `offsets`, and
`resolution` explicitly when value equality is needed. That avoids an
implicit linear scan of arrays that may be very large.

Read-only means Polypix returns these NumPy arrays with `WRITEABLE=False` and
does not support mutating them. It prevents accidental writes; it is not a
security boundary or a promise of deep immutability against deliberate NumPy
flag manipulation. Imported arrays are still copied, so later changes to the
caller's inputs cannot change a `Coverage`.

## OccupancyRuns

`OccupancyRuns` values are produced only by `occupancy_runs()`; manual
construction is intentionally disabled.

**Attributes**

`cells`
: *ndarray*. Ascending qualifying RING IDs, dtype `int64`.

`offsets`
: *ndarray*. `int64` boundaries grouping flattened runs by cell.

`starts`, `stops`
: *ndarray*. Matching `int64` half-open segment boundaries. Runs for cell `i`
  are in `starts[offsets[i]:offsets[i + 1]]` and the corresponding `stops`.

`run_counts`
: *ndarray*. Derived `int64` run count per cell, equal to `np.diff(offsets)`.

`resolution`, `segment_count`
: *int*. Common grid resolution and number of ordered input segments.

`minimum_sources`, `source_count`
: *int*. Threshold used to produce the runs and number of source entries that
  were supplied.

`OccupancyRuns` uses identity equality for the same reason as `Coverage`. Its
arrays follow the same read-only contract, and `len(runs)` is the number of
represented cells.

## Geometry contract

A polygon is a convex spherical polygon contained in an open hemisphere.
Adjacent vertices are joined by the unique shorter great-circle arc, so
longitudes `-179°` and `179°` are two degrees apart, and a hemisphere or larger
region cannot be represented.

Accepted:

- either vertex orientation;
- one repeated closing vertex;
- redundant vertices on the same great-circle edge, within floating-point
  precision;
- a cell center lying exactly on an edge, which is covered.

Rejected:

- fewer than three unique vertices;
- duplicate, antipodal, or non-finite vertices;
- degenerate edges, self-intersections, and non-convex geometry;
- exact-hemisphere boundaries and other detectably ambiguous geometry.

Polypix rejects detectable ambiguity but cannot infer that otherwise valid
vertices were meant to describe the other side of the sphere.

Two numerical limits apply. Footprints below roughly `1e-8` radians in angular
extent are unsupported and may be rejected as degenerate; the crossover depends
on vertex layout and conditioning. Concavity at the same scale can be
numerically indistinguishable from a collinear edge. Center inclusion uses a
nominal `1e-14` dot-product tolerance. That is a predicate threshold, not a
bound on absolute error, since uncertainty also depends on edge length and the
equivalent center-evaluation path. Only centers numerically indistinguishable from a
boundary are strategy-sensitive.

Caps do not have polygon conditioning or convexity limits. Their boundary uses
an angular tolerance of `1e-14` radians and a stable chord-distance predicate;
only cell centers indistinguishable from that boundary are numerically
sensitive.

Validation compares vertex pairs and tests each edge against every vertex, so
its cost is quadratic in vertex count. Split densely sampled boundaries into
short `cover_sweep()` segments instead of passing one polygon with many
vertices.
