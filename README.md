# Polypix

Fast HEALPix coverage for convex footprints on the sphere.

[![PyPI](https://img.shields.io/pypi/v/polypix.svg)](https://pypi.org/project/polypix/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB.svg?logo=python&logoColor=white)](https://pypi.org/project/polypix/)
[![License](https://img.shields.io/pypi/l/polypix.svg)](LICENSE)
[![Tests](https://github.com/JochimMaene/polypix/actions/workflows/run-tests.yml/badge.svg)](https://github.com/JochimMaene/polypix/actions/workflows/run-tests.yml)
[![Docs](https://github.com/JochimMaene/polypix/actions/workflows/docs.yml/badge.svg)](https://github.com/JochimMaene/polypix/actions/workflows/docs.yml)
[![Benchmarks](https://github.com/JochimMaene/polypix/actions/workflows/codspeed.yml/badge.svg)](https://github.com/JochimMaene/polypix/actions/workflows/codspeed.yml)

[Documentation](https://jochimmaene.github.io/polypix/) |
[Changelog](CHANGELOG.md) |
[PyPI](https://pypi.org/project/polypix/) |
[Repository](https://github.com/JochimMaene/polypix) |
[Issues](https://github.com/JochimMaene/polypix/issues)

Polypix returns the HEALPix cells whose centers fall inside convex footprints on
the unit sphere, for coverage simulations and indexing pipelines where the
footprints are already valid spherical geometry and throughput matters. Typical
inputs are sensor footprints, beam contours, access regions, and swath edges.

It is not a fit for holes, non-convex footprints, planar geometry semantics,
conservative overlap coverage, or generating footprints from orbit, attitude,
sensor, or beam models.

## Install

```bash
python -m pip install polypix
```

Wheels cover CPython 3.12+ on Linux x86-64 and ARM64, macOS 11+ on Intel and
Apple Silicon, and Windows x86-64. NumPy is the only runtime dependency.

## Quick start

```python
import numpy as np
import polypix as px

lon, lat = np.radians([[-5.0, 12.0, 10.0, -6.0], [-5.0, -4.0, 9.0, 7.0]])
footprint = np.stack(
    [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
    axis=-1,
)

coverage = px.cover_footprint(footprint, resolution=8)
centers = px.centers(coverage.cells, coverage.resolution)
```

`coverage.cells` holds standard HEALPix RING indices as `uint64`;
`coverage.offsets` splits that flat array into one segment per input footprint.
Geometry helpers return body-centered unit vectors, never longitude/latitude or
datum-specific coordinates.

## Inputs

- convex spherical footprints with great-circle edges;
- finite body-centered `(x, y, z)` vectors, normalized by Polypix;
- dense batches of shape `(footprints, vertices, 3)`;
- ragged batches as sequences of `(vertices, 3)` arrays;
- strips from sampled left and right edge vectors.

Vertex orientation does not matter. A repeated final vertex is accepted as a
closed-ring marker.

## Coverage rule

A cell is included when its center lies inside the footprint or on its boundary.
Boundary-touching cells whose centers fall outside are excluded.

## Examples

Two constellation studies run during every documentation build, so their maps
and timings match the code that produced them:

- [Communications availability](https://jochimmaene.github.io/polypix/examples/communication-constellation/)
  — 30,500 service footprints from 500 satellites.
- [Earth-observation revisit](https://jochimmaene.github.io/polypix/examples/earth-observation-constellation/)
  — 144,000 swept swath intervals from 10 satellites.

## Documentation

Published at <https://jochimmaene.github.io/polypix/>:

- [Install](https://jochimmaene.github.io/polypix/install/)
- [Concepts](https://jochimmaene.github.io/polypix/concepts/)
- [API reference](https://jochimmaene.github.io/polypix/api/)
- [Project goal](https://jochimmaene.github.io/polypix/project-goal/)
- [Development](https://jochimmaene.github.io/polypix/development/)

## License

Apache License 2.0. See `THIRD_PARTY_NOTICES.md` for dependency and
embedded-code notices.
