# Polypix

Fast batch HEALPix rasterization and occupancy reduction for spherical regions.

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

Polypix maps directions to HEALPix RING cells and rasterizes exact spherical
caps, convex footprints, and paired-edge sweeps by cell-center inclusion. It
also provides focused reductions where materializing and reducing every
repeated cell ID would dominate the geometry itself. Typical inputs are survey
tiles, sensor footprints, beam contours, access regions, and swept edges.

It is not a fit for holes, non-convex footprints, planar geometry semantics,
conservative cell-intersection coverage, or generating geometry from orbit,
attitude, sensor, or beam models.

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
round_trip_cells = px.cell_at(centers, coverage.resolution)
```

`coverage.cells` holds standard HEALPix RING indices as `uint64`;
`coverage.offsets` splits that flat array into one segment per input footprint.
Geometry helpers return unit direction vectors in the caller's Cartesian
frame, never longitude/latitude or datum-specific coordinates.

Circular regions should use `cover_cap()` directly. If only the number of caps
containing each cell matters, `count_caps_per_cell()` accumulates exact RING
spans without constructing the much larger segmented cell list. Ordered sweep
results can be reduced into source runs and merged occupancy gaps with
`summarize_occupancy()`.

## Inputs

- convex spherical footprints with great-circle edges;
- finite Cartesian direction vectors `(x, y, z)` in one caller-defined frame,
  normalized by Polypix;
- dense batches of shape `(footprints, vertices, 3)`;
- ragged batches as sequences of `(vertices, 3)` arrays;
- sweeps from sampled left and right edge vectors;
- exact spherical caps from center vectors and angular radii.

Vertex orientation does not matter. A repeated final vertex is accepted as a
closed-ring marker.

## Coverage rule

A cell is included when its center lies inside the cap or footprint, or on its
boundary. Boundary-touching cells whose centers fall outside are excluded.

## Examples

Two constellation studies run during every documentation build, so their maps
and timings match the code that produced them:

- [Starlink snapshot visibility](https://jochimmaene.github.io/polypix/examples/communication-constellation/)
  — 657,031 exact service caps from a pinned catalog of 10,771 objects.
- [Earth-observation revisit](https://jochimmaene.github.io/polypix/examples/earth-observation-constellation/)
  — 144,000 swept swath intervals from 10 satellites.

For small dependency-free examples, begin with the
[Guide and recipes](https://jochimmaene.github.io/polypix/guide/). Resolution,
output-memory, candidate, and threading guidance is collected in
[Performance and memory](https://jochimmaene.github.io/polypix/performance/).

## Documentation

Published at <https://jochimmaene.github.io/polypix/>:

- [Install](https://jochimmaene.github.io/polypix/install/)
- [Guide and recipes](https://jochimmaene.github.io/polypix/guide/)
- [Concepts](https://jochimmaene.github.io/polypix/concepts/)
- [API reference](https://jochimmaene.github.io/polypix/api/)
- [Performance and memory](https://jochimmaene.github.io/polypix/performance/)
- [Interoperability](https://jochimmaene.github.io/polypix/interoperability/)
- [Development](https://jochimmaene.github.io/polypix/development/)
- [Project goal](https://jochimmaene.github.io/polypix/project-goal/)

## License

Apache License 2.0. See `THIRD_PARTY_NOTICES.md` for dependency and
embedded-code notices.
