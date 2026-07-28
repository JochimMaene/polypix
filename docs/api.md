# API reference

```python
import polypix as px
```

| Object | Purpose |
| --- | --- |
| [`cover_footprint`](#cover_footprint) | Cover convex spherical polygons |
| [`cover_strip`](#cover_strip) | Cover the quadrilaterals between two sampled edges |
| [`centers`](#centers) | Cell center vectors |
| [`boundaries`](#boundaries) | Cell corner vectors |
| [`Coverage`](#coverage) | Result of both coverage calls |

## cover_footprint

```python
cover_footprint(footprints_xyz, resolution, *, candidate_cells=None, threads=None)
```

Cover convex spherical footprints by HEALPix cell-center inclusion.

**Parameters**

- **`footprints_xyz`** : *array_like or sequence of array_like* —
  one `(vertices, 3)` footprint, a dense `(footprints, vertices, 3)` batch, or a
  sequence of `(vertices, 3)` arrays for a ragged batch. Vectors are finite,
  nonzero, and body-centered; magnitudes are normalized internally.
- **`resolution`** : *int* — HEALPix resolution, 0 through 29. Returned cells
  satisfy `0 <= cell < 12 * 4 ** resolution`.
- **`candidate_cells`** : *array_like of int, optional* — RING indices at
  `resolution` restricting which centers are tested. Set semantics; duplicates
  are ignored. An empty set returns empty segments without dropping input items.
- **`threads`** : *int, optional* — `None` selects the automatic policy, `1` is
  sequential, and larger values set the reusable worker-pool maximum, capped by
  the host.

**Returns**

- **`Coverage`** — flat RING indices with offsets delimiting one segment per
  input footprint.

**Raises**

- **`TypeError`** — inputs have incompatible numeric types.
- **`ValueError`** — invalid shapes, resolution, vectors, or polygon geometry.

**Notes**

An empty sequence, a one-dimensional empty array, or a dense
`(0, vertices, 3)` array is a batch of zero footprints. A `(0, 3)` array is
unambiguously one footprint with zero vertices and is rejected.

Contiguous NumPy arrays are borrowed for the duration of the call. Because the
native kernel releases the GIL, do not mutate an input or candidate array from
another thread before the call returns.

Strictly increasing candidate arrays use a borrowed fast path; other inputs are
sorted and deduplicated internally.

Threading does not change membership, segment order, or cell order on the same
build and platform.

See [Geometry contract](#geometry-contract) for the accepted polygons and
[Candidate cells](concepts.md#candidate-cells) for the performance trade-offs.

## cover_strip

```python
cover_strip(left_edge_xyz, right_edge_xyz, resolution, *, candidate_cells=None, threads=None)
```

Cover the quadrilateral segments between two sampled spherical edges.

**Parameters**

- **`left_edge_xyz`**, **`right_edge_xyz`** : *array_like* — `(samples, 3)`
  vector arrays of equal length, with at least two samples.
- **`resolution`**, **`candidate_cells`**, **`threads`** — as in
  [`cover_footprint`](#cover_footprint).

**Returns**

- **`Coverage`** — for `N` paired samples, `N - 1` segments, where segment `i`
  covers the quadrilateral `[left[i], right[i], right[i+1], left[i+1]]`.

**Raises**

- **`TypeError`** — inputs have incompatible numeric types.
- **`ValueError`** — mismatched edge lengths, fewer than two samples, or an
  invalid or zero-area segment.

**Notes**

Segments are independent. Polypix does not merge or deduplicate them; form a
union explicitly with NumPy when one is needed.

Repeating both edge samples at the same step gives a zero-area segment and is
rejected. Remove stationary duplicate samples upstream. Repeating a sample on
one edge only is accepted and gives a triangle pinched at that edge.

Consecutive samples are joined by minor great-circle arcs, so sampling density
is part of the input contract. See
[Batches and segments](concepts.md#batches-and-segments).

## centers

```python
centers(cells, resolution)
```

Return unit-vector centers for HEALPix RING indices.

**Parameters**

- **`cells`** : *int or array_like of int* — RING indices at `resolution`.
- **`resolution`** : *int* — HEALPix resolution, 0 through 29.

**Returns**

- **`ndarray`** — shape `(cells, 3)`, dtype `float64`. A scalar cell returns
  `(1, 3)`; empty input returns `(0, 3)`.

**Raises**

- **`TypeError`** — non-integer input.
- **`ValueError`** — invalid resolution, negative values, or out-of-range
  indices.

**Notes**

Large arrays are parallelized inside the native kernel. This adds no threading
argument to the supporting utilities.

## boundaries

```python
boundaries(cells, resolution)
```

Return the four unit-vector corners of each HEALPix cell, in boundary traversal
order.

**Parameters** — as in [`centers`](#centers).

**Returns**

- **`ndarray`** — shape `(cells, 4, 3)`, dtype `float64`. A scalar cell retains
  the leading axis and returns `(1, 4, 3)`. The first corner is not repeated.

**Raises** — as in [`centers`](#centers).

## Coverage

```python
Coverage(cells, offsets, resolution)
```

**Attributes**

- **`cells`** : *ndarray* — flat one-dimensional `uint64` array of standard
  HEALPix RING indices.
- **`offsets`** : *ndarray* — `uint64` segment boundaries, length
  `item_count + 1`.
- **`resolution`** : *int* — the resolution shared by every returned cell.
- **`counts`** : *ndarray* — derived `intp` array equal to `np.diff(offsets)`.

Cells for input item `i`:

```python
coverage.cells[coverage.offsets[i] : coverage.offsets[i + 1]]
```

**Notes**

Segments preserve input order and contain no duplicate cells. Cell order within
a segment is deterministic for a given build and platform but is not part of the
API; Polypix never sorts for presentation alone.

`Coverage` uses identity equality. Compare `cells`, `offsets`, and `resolution`
explicitly when value equality is needed — this avoids an implicit linear scan of
arrays that may be very large.

## Geometry contract

A footprint is a convex spherical polygon contained in an open hemisphere.
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
nominal `1e-14` dot-product tolerance — a predicate threshold, not a bound on
absolute error, since uncertainty also depends on edge length and the equivalent
center-evaluation path. Only centers numerically indistinguishable from a
boundary are strategy-sensitive.

Validation compares vertex pairs and tests each edge against every vertex, so
its cost is quadratic in vertex count. Split densely sampled boundaries into
short `cover_strip()` segments instead of passing one polygon with many
vertices.
