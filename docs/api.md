# API Reference

```python
import polypix as px
```

The complete public API is `Coverage`, `cover_footprint`, `cover_strip`,
`centers`, and `boundaries`.

## Coverage

Every coverage call returns:

```python
px.Coverage(cells, offsets, resolution)
```

- `cells`: flat one-dimensional `uint64` array of standard HEALPix RING
  indices.
- `offsets`: `uint64` segment boundaries, with length `item_count + 1`.
- `resolution`: the common HEALPix resolution for every returned cell.
- `counts`: derived `intp` array equal to `np.diff(offsets)`.

Cells for input item `i` are:

```python
coverage.cells[coverage.offsets[i] : coverage.offsets[i + 1]]
```

Segments contain no duplicate cells and preserve input item order. Their
internal deterministic traversal order is intentionally not an API promise.
`Coverage` instances use identity equality: compare `cells`, `offsets`, and
`resolution` explicitly when value equality is needed. This avoids an implicit
linear scan of potentially very large arrays.

## cover_footprint

```python
px.cover_footprint(
    footprints_xyz,
    resolution,
    *,
    candidate_cells=None,
    threads=None,
)
```

Cover one convex spherical footprint or a batch using cell-center inclusion.

`footprints_xyz` accepts:

- one `(vertices, 3)` numeric array-like;
- a dense `(footprints, vertices, 3)` numeric array-like;
- a sequence of `(vertices, 3)` arrays for a ragged batch.

An empty sequence, a one-dimensional empty array, or a dense
`(0, vertices, 3)` array represents a batch with zero footprints. A
`(0, 3)` array instead represents one footprint with zero vertices and is
rejected.

Vectors are finite, nonzero, body-centered `(x, y, z)` coordinates. Polypix
normalizes their magnitudes. It assigns no datum, ellipsoid, or CRS meaning to
them.

`resolution` is an integer from 0 through 29. The returned values satisfy:

```text
0 <= cell < 12 * 4 ** resolution
```

`candidate_cells` optionally restricts the result to a one-dimensional set of
standard RING indices at the requested resolution. Duplicate candidates are
ignored. An empty candidate set returns empty segments without dropping input
items. Strictly increasing candidate arrays use a borrowed fast path; other
inputs are sorted and deduplicated internally. Center inclusion uses a nominal
`1e-14` dot-product boundary tolerance.
This is a predicate threshold, not a strict absolute-error guarantee:
floating-point uncertainty also depends on edge length and on the equivalent
center-evaluation path. Only centers numerically indistinguishable from a
boundary can be strategy-sensitive.

`threads=None` uses the automatic native policy. `threads=1` disables internal
parallelism; a larger positive integer sets the reusable worker-pool maximum.
The pool is capped by the host's available parallelism. Calls below the
measured parallel crossover remain sequential and do not initialize a pool,
even when a larger explicit maximum is supplied. Threading does not change
membership, segment order, or cell order on the same build and platform.

Example:

```python
import math

import numpy as np
import polypix as px


def lonlat_to_xyz(lon_deg, lat_deg):
    lon = math.radians(lon_deg)
    lat = math.radians(lat_deg)
    cos_lat = math.cos(lat)
    return cos_lat * math.cos(lon), cos_lat * math.sin(lon), math.sin(lat)


footprint = np.asarray(
    [
        lonlat_to_xyz(-5.0, -5.0),
        lonlat_to_xyz(12.0, -4.0),
        lonlat_to_xyz(10.0, 9.0),
        lonlat_to_xyz(-6.0, 7.0),
    ],
    dtype=np.float64,
)

coverage = px.cover_footprint(footprint, resolution=8)
```

## cover_strip

```python
px.cover_strip(
    left_edge_xyz,
    right_edge_xyz,
    resolution,
    *,
    candidate_cells=None,
    threads=None,
)
```

`left_edge_xyz` and `right_edge_xyz` are `(samples, 3)` vector arrays with the
same length and at least two samples. Each consecutive pair forms one convex
quadrilateral:

```text
[left[i], right[i], right[i + 1], left[i + 1]]
```

For `N` paired samples, the result therefore contains `N - 1` segments.
Polypix does not merge those segments. `candidate_cells` and `threads` have the
same meaning as in `cover_footprint()`.

Consecutive paired samples must describe a nonzero-area segment. Repeating both
edge samples at the same step is rejected; remove stationary duplicate samples
upstream before calling `cover_strip()`. Repeating a sample on only one edge is
accepted and produces a triangular segment pinched at that edge.

## centers

```python
px.centers(cells, resolution)
```

Returns normalized body-centered center vectors with shape `(n, 3)` and dtype
`float64`. A scalar cell is treated as a one-cell input and returns shape
`(1, 3)`. Empty input returns shape `(0, 3)`. Large arrays are parallelized
automatically inside the native kernel; this does not add threading controls to
the supporting utility.

## boundaries

```python
px.boundaries(cells, resolution)
```

Returns the four normalized HEALPix corner vectors per cell with shape
`(n, 4, 3)` and dtype `float64`. A scalar cell retains its leading cell axis
and returns shape `(1, 4, 3)`. The first corner is not repeated.
Large arrays are parallelized automatically inside the native kernel.

## Geometry Contract

Footprints must be convex spherical polygons contained in an open hemisphere.
Adjacent vertices are joined by the unique shorter great-circle arc. This
minor-arc interpretation determines the region across longitude wraparound and
means that a hemisphere or larger region cannot be represented. Polypix
rejects detectable ambiguity, including antipodal edges and exact-hemisphere
boundaries, but cannot infer that otherwise valid vertices were intended to
describe the other side of the sphere. Either orientation and one repeated
closing vertex are accepted. A cell center on an edge is included.

Polypix rejects footprints with fewer than three unique vertices, duplicate or
antipodal vertices, degenerate edges, non-finite coordinates, self
intersections, or non-convex geometry. Redundant vertices on the same
great-circle edge are accepted within floating-point precision. Validation has
a numerical scale floor: footprints with angular extent below roughly
`1e-8` radians are unsupported and may be rejected as degenerate. The precise
crossover depends on vertex layout and conditioning. Concavity below the same
floating-point validation scale may be numerically indistinguishable from a
collinear edge.

For `cover_strip()`, consecutive samples on each edge are joined by the same
minor arcs. Callers must sample physical swaths densely enough that these arcs
are the intended boundary. Near-180-degree steps can bow toward a pole, and a
step beyond 180 degrees selects the opposite shorter arc. Repeating both paired
samples creates a zero-area segment and is rejected. If only one edge repeats,
the segment is accepted as a triangle pinched at that edge.
