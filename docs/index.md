# Polypix

Polypix computes HEALPix cells whose centers fall inside convex spherical
polygons. It is a small Python package for workloads that already have clean
spherical footprints and need fast, NumPy-friendly coverage results.

Use Polypix when you want a deterministic center-sampled cover of a convex
region on the sphere. Do not use it when you need planar polygon semantics,
holes, non-convex polygons, or every HEALPix cell that touches a polygon
boundary.

## Install

```bash
python -m pip install polypix
```

Published wheels are available for Python 3.12 and newer on Linux x86_64 and
macOS 11 or newer on Intel and Apple Silicon.

## Quick Start

```python
import numpy as np
import polypix as px

polygon = np.array(
    [
        [-5.0, -5.0],
        [12.0, -4.0],
        [10.0, 9.0],
        [-6.0, 7.0],
    ],
    dtype=np.float64,
)

geometry = px.Polygon.from_lonlat(polygon)
cell_ids = px.cover(geometry, resolution=8)
centers = px.center(cell_ids)
boundaries = px.boundary(cell_ids[:3])
```

`cell_ids` is a one-dimensional `uint64` array. Each value is a packed Polypix
cell token that stores the HEALPix resolution and NESTED pixel index. Treat
these values as opaque IDs; use `center()` or `boundary()` when you need
longitude/latitude geometry.

## Batch Coverage

For many polygons, use `MultiPolygon`. The result is a flat cell array plus one
cell count per input polygon.

```python
polygon_a = np.array(
    [
        [-5.0, -5.0],
        [12.0, -4.0],
        [10.0, 9.0],
        [-6.0, 7.0],
    ],
    dtype=np.float64,
)
polygon_b = np.array(
    [
        [20.0, -10.0],
        [33.0, -10.0],
        [33.0, 0.0],
        [20.0, 0.0],
    ],
    dtype=np.float64,
)

batch = px.MultiPolygon.from_lonlat([polygon_a, polygon_b])
cell_ids, counts = px.cover(batch, resolution=8)
cells_by_polygon = np.split(cell_ids, np.cumsum(counts[:-1]))
```

## Coordinate Systems

Polypix accepts two explicit coordinate systems:

- longitude/latitude vertices in degrees, through `from_lonlat(...)`;
- normalized unit vectors as `(x, y, z)`, through `from_xyz(...)`.

Polygon edges are interpreted as great-circle segments. Vertex orientation does
not matter; Polypix normalizes orientation internally. A repeated final vertex
is accepted as a closed-ring marker.

## Coverage Rule

Polypix uses center-in-polygon coverage: a HEALPix cell is included only if its
center lies inside the spherical polygon. Boundary-touching cells whose centers
fall outside the polygon are not included.

This rule is compact and deterministic, but it is not a conservative overlap
cover.

## More Information

- [Install](install.md) describes supported wheels and source builds.
- [Concepts](concepts.md) explains resolutions, packed cell IDs, and geometry assumptions.
- [API](api.md) documents the public Python interface.
- [Development](development.md) covers local development, releases, and documentation publishing.
