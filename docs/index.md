# Polypix

Polypix computes HEALPix cells whose centers fall inside convex footprints on
the sphere. It is a small Python package for coverage simulations and indexing
pipelines that already have clean spherical footprints and need fast,
NumPy-friendly results.

Typical inputs are sensor footprints, beam contours, access regions, and swath
edges from satellite, aerial, astronomy, or other spherical-domain simulations.
Use Polypix when you want deterministic center-sampled coverage for convex
regions. Do not use it when you need planar geometry semantics, holes,
non-convex footprints, every HEALPix cell that touches a footprint boundary, or
footprint generation from orbit, attitude, sensor, or beam models.

## Install

```bash
python -m pip install polypix
```

Published wheels are available for Python 3.12 and newer on Linux x86_64 and
macOS 11 or newer on Intel and Apple Silicon.

## Quick Start

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
centers = px.centers(coverage.cell_ids)
boundaries = px.boundaries(coverage.cell_ids[:3])
```

`coverage.cell_ids` is a one-dimensional `uint64` array. Each value is a packed
Polypix cell token that stores the HEALPix resolution and NESTED pixel index.
Treat these values as opaque IDs; use `centers()` or `boundaries()` when you
need longitude/latitude geometry.

## Batch Coverage

For many footprints, pass a dense array with shape `(footprints, vertices, 3)`.
The result stores one flat cell array plus output offsets.

```python
footprint_a = np.asarray(
    [
        lonlat_to_xyz(-5.0, -5.0),
        lonlat_to_xyz(12.0, -4.0),
        lonlat_to_xyz(10.0, 9.0),
        lonlat_to_xyz(-6.0, 7.0),
    ],
    dtype=np.float64,
)
footprint_b = np.asarray(
    [
        lonlat_to_xyz(20.0, -10.0),
        lonlat_to_xyz(33.0, -10.0),
        lonlat_to_xyz(33.0, 0.0),
        lonlat_to_xyz(20.0, 0.0),
    ],
    dtype=np.float64,
)

coverage = px.cover_footprint(np.stack([footprint_a, footprint_b]), resolution=8)
cells_by_footprint = [
    coverage.cell_ids[start:stop]
    for start, stop in zip(coverage.offsets[:-1], coverage.offsets[1:])
]
```

## Coordinate Systems

Polypix accepts normalized unit vectors as `(x, y, z)`.

Footprint edges are interpreted as great-circle segments. Vertex orientation
does not matter; Polypix normalizes orientation internally. A repeated final
vertex is accepted as a closed-ring marker.

## Swath Coverage

For strip-like coverage, pass the sampled left and right footprint edges
directly:

```python
coverage = px.cover_swath(left_edge_xyz, right_edge_xyz, resolution=8)
```

Both edge arrays must have shape `(samples, 3)`. Polypix covers each
consecutive interval as one quadrilateral.

## Coverage Rule

Polypix uses center-in-footprint coverage: a HEALPix cell is included only if
its center lies inside the spherical footprint. Boundary-touching cells whose
centers fall outside the footprint are not included.

This rule is compact and deterministic, but it is not a conservative overlap
cover.

## More Information

- [Install](install.md) describes supported wheels and source builds.
- [Concepts](concepts.md) explains resolutions, packed cell IDs, and geometry assumptions.
- [API](api.md) documents the public Python interface.
- [Development](development.md) covers local development, releases, and documentation publishing.
