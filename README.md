# Polypix

Fast HEALPix center-in-polygon coverage for convex spherical polygons.

[![PyPI](https://img.shields.io/pypi/v/polypix.svg)](https://pypi.org/project/polypix/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB.svg?logo=python&logoColor=white)](https://pypi.org/project/polypix/)
[![License](https://img.shields.io/pypi/l/polypix.svg)](LICENSE)
[![Tests](https://github.com/JochimMaene/polypix/actions/workflows/run-tests.yml/badge.svg)](https://github.com/JochimMaene/polypix/actions/workflows/run-tests.yml)
[![Docs](https://github.com/JochimMaene/polypix/actions/workflows/docs.yml/badge.svg)](https://github.com/JochimMaene/polypix/actions/workflows/docs.yml)

[Documentation](https://jochimmaene.github.io/polypix/) |
[PyPI](https://pypi.org/project/polypix/) |
[Repository](https://github.com/JochimMaene/polypix) |
[Issues](https://github.com/JochimMaene/polypix/issues)

Polypix returns the HEALPix cells whose centers fall inside a convex spherical
polygon. It is built for NumPy-friendly, throughput-oriented workloads where
the input footprint is already valid on the unit sphere.

Use Polypix when you want a deterministic center-sampled cover of a convex
region. It is not a fit for holes, non-convex polygons, planar geometry
semantics, or conservative overlap coverage.

## Install

```bash
python -m pip install polypix
```

Published wheels support Python 3.12 and newer on Linux x86_64 and macOS 11 or
newer on Intel and Apple Silicon.

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

cell_ids = px.cover(px.Polygon.from_lonlat(polygon), resolution=8)
centers = px.center(cell_ids)
boundaries = px.boundary(cell_ids[:3])
```

The returned cell IDs are packed `uint64` tokens that include both the HEALPix
resolution and the NESTED pixel index. Treat them as opaque IDs; use
`center()` or `boundary()` when you need longitude/latitude geometry.

## Supported Inputs

Polypix supports:

- convex spherical polygons with great-circle edges,
- longitude/latitude vertices in degrees,
- unit-vector vertices as `(x, y, z)`,
- dense and ragged polygon batches.

Vertex orientation does not matter; Polypix normalizes it internally. A
repeated final vertex is accepted as a closed-ring marker.

## Coverage Rule

Polypix uses center-in-polygon coverage: a HEALPix cell is included only when
its center lies inside the polygon. Boundary-touching cells whose centers fall
outside the polygon are excluded.

Windows wheels are not published because `healpix_cxx` is not currently
available as a conda-forge `win-64` package.

## Documentation

The public documentation is published at
<https://jochimmaene.github.io/polypix/>:

- [Install guide](https://jochimmaene.github.io/polypix/install/)
- [Concepts](https://jochimmaene.github.io/polypix/concepts/)
- [API reference](https://jochimmaene.github.io/polypix/api/)
- [Development guide](https://jochimmaene.github.io/polypix/development/)

Contributor workflows, release notes, and local docs authoring live in the
development guide instead of this user-facing overview.

## License

Polypix is distributed under GPL-3.0-or-later. This reflects the binary
distribution relationship with HEALPix C++, which is GPL-2.0-or-later. See
`THIRD_PARTY_NOTICES.md` for native dependency notices.
