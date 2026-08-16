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

Polypix maps Cartesian directions and spherical regions to fixed-resolution
HEALPix RING cells. Its native batch kernels cover exact caps, convex
great-circle polygons, and sampled sweeps, returning NumPy arrays throughout.

## Install

```bash
pip install polypix
```

Wheels cover CPython 3.12+ on Linux x86-64 and ARM64, macOS 11+ on Intel and
Apple Silicon, and Windows x86-64. NumPy is the only runtime dependency.

## Quick start

```python
import numpy as np
import polypix as px

centers = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
radii = np.deg2rad([5.0, 8.0])

coverage = px.cover_cap(centers, radii, resolution=8)
print(coverage.counts)  # [1502 3824]
```

`coverage.cells` holds standard HEALPix RING indices as `uint64`;
`coverage.offsets` splits the flat array into one segment per input cap.

## What it provides

- direction-to-cell indexing with `cell_at()`;
- explicit membership for caps, convex footprints, and paired-edge sweeps;
- dense or selected-cell cap counts without materializing cap membership;
- a compact `Coverage` result for large and ragged batches;
- native threading with deterministic result order.

Polypix expects geometry in one caller-defined Cartesian frame. Coordinate
transforms, propagation, sensor models, interpolation, plotting, and map algebra
belong in the surrounding scientific Python ecosystem.

## Coverage rule

A cell is included when its center lies inside the cap or footprint, or on its
boundary. Boundary-touching cells whose centers fall outside are excluded.

## Documentation

Start with the [getting-started guide](https://jochimmaene.github.io/polypix/guide/),
then use the [API reference](https://jochimmaene.github.io/polypix/api/) for the
complete call contracts. Resolution, batching, memory, and threading are covered
in [Performance and memory](https://jochimmaene.github.io/polypix/performance/).

Two executable case studies are rebuilt with the documentation:

Two constellation studies run during every documentation build, so their maps
and timings match the code that produced them:

- [Starlink snapshot visibility](https://jochimmaene.github.io/polypix/examples/communication-constellation/)
  — 657,031 exact service caps from a pinned catalog of 10,771 objects.
- [Earth-observation revisit](https://jochimmaene.github.io/polypix/examples/earth-observation-constellation/)
  — 144,000 swept swath intervals from 10 satellites.

## License

Apache License 2.0. See `THIRD_PARTY_NOTICES.md` for dependency and
embedded-code notices.
