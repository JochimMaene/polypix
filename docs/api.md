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
items.

`threads=None` uses the automatic native policy. `threads=1` disables internal
parallelism; a larger positive integer requests that many workers. Threading
does not change membership, segment order, or cell order.

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

## centers

```python
px.centers(cells, resolution)
```

Returns normalized body-centered center vectors with shape `(n, 3)` and dtype
`float64`. A scalar cell is treated as a one-cell input and returns shape
`(1, 3)`. Empty input returns shape `(0, 3)`.

## boundaries

```python
px.boundaries(cells, resolution)
```

Returns the four normalized HEALPix corner vectors per cell with shape
`(n, 4, 3)` and dtype `float64`. A scalar cell retains its leading cell axis
and returns shape `(1, 4, 3)`. The first corner is not repeated.

## Geometry Contract

Footprints must be convex spherical polygons contained within an unambiguous
hemisphere. Edges follow shorter great-circle arcs. Either orientation and one
repeated closing vertex are accepted. A cell center on an edge is included.

Polypix rejects footprints with fewer than three unique vertices, duplicate or
antipodal vertices, degenerate edges, non-finite coordinates, self
intersections, or non-convex geometry.
