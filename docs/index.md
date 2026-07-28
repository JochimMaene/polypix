# Polypix

Polypix converts large batches of convex spherical footprints into the HEALPix
RING cells whose centers lie inside them. It is a deliberately small,
NumPy-first engine for coverage simulations and spatial indexing.

Use it when footprint geometry already exists and throughput matters. Satellite
and sensor coverage is the leading use case, but the vectors can belong to any
spherical coordinate frame. Polypix does not model orbits, attitude, sensors,
time, ellipsoids, datums, or coordinate reference systems.

## See It Work

Two complete constellation studies run during every documentation build, so
their maps and timings always match the code that produced them.

| Example | Workload | Polypix time |
| --- | --- | ---: |
| [Communications availability](examples/communication-constellation.md) | 30,500 service footprints, 500 satellites | ~0.3 s |
| [Earth-observation revisit](examples/earth-observation-constellation.md) | 144,000 swept swath intervals, 10 satellites | ~0.2 s |

Each returns roughly eight million footprint-cell pairs. Both pages show the
live measurements from the current build.

## Install

```bash
python -m pip install polypix
```

Wheels target CPython 3.12 and newer on Linux x86-64 and ARM64, macOS Intel and
Apple Silicon, and Windows x86-64. NumPy is the only runtime dependency.

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
center_vectors = px.centers(coverage.cells, coverage.resolution)
corner_vectors = px.boundaries(coverage.cells[:3], coverage.resolution)
```

`coverage.cells` contains standard fixed-resolution HEALPix RING indices.
`coverage.offsets` divides that flat `uint64` array into one segment per input
footprint.

## Batches

A dense batch has shape `(footprints, vertices, 3)`. A sequence of
`(vertices, 3)` arrays supports footprints with different vertex counts:

```python
coverage = px.cover_footprint(
    [triangle_xyz, quadrilateral_xyz, pentagon_xyz],
    resolution=9,
)

cells_by_footprint = [
    coverage.cells[start:stop]
    for start, stop in zip(coverage.offsets[:-1], coverage.offsets[1:])
]
```

Inputs are finite, nonzero body-centered vectors. Polypix normalizes their
magnitudes and accepts either polygon orientation.

## Strips

For a sampled strip, pass its paired edges:

```python
coverage = px.cover_strip(left_edge_xyz, right_edge_xyz, resolution=8)
```

Each consecutive sample pair forms one convex quadrilateral and one result
segment. Polypix does not implicitly union the intervals.

## Coverage Rule

A cell is included when its center lies inside the convex spherical footprint
or on its boundary. This is not conservative cell intersection or fractional
area coverage.

## More Information

- [Project Goal](project-goal.md) defines product scope and feature admission.
- [Install](install.md) covers wheels and source builds.
- [Communications constellation](examples/communication-constellation.md)
  covers batched instantaneous service footprints.
- [Earth-observation constellation](examples/earth-observation-constellation.md)
  covers swept strips, distinct observations, and revisit time.
- [Concepts](concepts.md) explains IDs, geometry, batches, and threading.
- [API](api.md) documents the complete Python interface.
- [Development](development.md) covers local development and releases.
