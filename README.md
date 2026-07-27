# Polypix

Fast HEALPix coverage for convex footprints on the sphere.

[![PyPI](https://img.shields.io/pypi/v/polypix.svg)](https://pypi.org/project/polypix/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB.svg?logo=python&logoColor=white)](https://pypi.org/project/polypix/)
[![License](https://img.shields.io/pypi/l/polypix.svg)](LICENSE)
[![Tests](https://github.com/JochimMaene/polypix/actions/workflows/run-tests.yml/badge.svg)](https://github.com/JochimMaene/polypix/actions/workflows/run-tests.yml)
[![Docs](https://github.com/JochimMaene/polypix/actions/workflows/docs.yml/badge.svg)](https://github.com/JochimMaene/polypix/actions/workflows/docs.yml)
[![Benchmarks](https://github.com/JochimMaene/polypix/actions/workflows/codspeed.yml/badge.svg)](https://github.com/JochimMaene/polypix/actions/workflows/codspeed.yml)

[Documentation](https://jochimmaene.github.io/polypix/) |
[PyPI](https://pypi.org/project/polypix/) |
[Repository](https://github.com/JochimMaene/polypix) |
[Issues](https://github.com/JochimMaene/polypix/issues)

Polypix returns the HEALPix cells whose centers fall inside convex footprints on
the unit sphere. It is built for coverage simulations and indexing pipelines
where footprints are already valid spherical geometry and throughput matters.

Typical inputs are sensor footprints, beam contours, access regions, and swath
edges from satellite, aerial, astronomy, or other spherical-domain simulations.
Use Polypix when you want deterministic center-sampled coverage for convex
regions. It is not a fit for holes, non-convex footprints, planar geometry
semantics, conservative overlap coverage, or generating footprints from orbit,
attitude, sensor, or beam models.

## Install

```bash
python -m pip install polypix
```

Published wheels support CPython 3.12 and newer on Linux x86-64 and ARM64,
macOS 11 or newer on Intel and Apple Silicon, and Windows x86-64. NumPy is the
only runtime dependency.

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

`coverage.cells` contains standard fixed-resolution HEALPix RING indices as
`uint64`; `coverage.offsets` divides the flat array into one segment per input
footprint. Geometry helpers return body-centered unit vectors, never
longitude/latitude or datum-specific coordinates.

## Supported Inputs

Polypix supports:

- convex spherical footprints with great-circle edges,
- finite body-centered vectors as `(x, y, z)`, normalized by Polypix,
- dense footprint batches as arrays with shape `(footprints, vertices, 3)`,
- ragged batches as sequences of `(vertices, 3)` arrays,
- strips from sampled left and right edge vectors.

Vertex orientation does not matter; Polypix normalizes it internally. A
repeated final vertex is accepted as a closed-ring marker.

## Coverage Rule

Polypix uses center-in-footprint coverage: a HEALPix cell is included when its
center lies inside the footprint or on its boundary. Boundary-touching cells
whose centers fall outside the footprint are excluded.

## Documentation

The public documentation is published at
<https://jochimmaene.github.io/polypix/>:

- [Install guide](https://jochimmaene.github.io/polypix/install/)
- [Project goal](https://jochimmaene.github.io/polypix/project-goal/)
- [Concepts](https://jochimmaene.github.io/polypix/concepts/)
- [API reference](https://jochimmaene.github.io/polypix/api/)
- [Development guide](https://jochimmaene.github.io/polypix/development/)

Contributor workflows, release notes, and local docs authoring live in the
development guide instead of this user-facing overview.

## License

Polypix is distributed under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). See
`THIRD_PARTY_NOTICES.md` for dependency and embedded-code notices.
