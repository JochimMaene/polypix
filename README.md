# Polypix

Fast HEALPix center-in-polygon coverage for convex spherical polygons.

[![PyPI](https://img.shields.io/pypi/v/polypix.svg)](https://pypi.org/project/polypix/)
[![Python](https://img.shields.io/pypi/pyversions/polypix.svg)](https://pypi.org/project/polypix/)
[![License](https://img.shields.io/pypi/l/polypix.svg)](LICENSE)
[![Tests](https://github.com/JochimMaene/polypix/actions/workflows/run-tests.yml/badge.svg)](https://github.com/JochimMaene/polypix/actions/workflows/run-tests.yml)
[![Docs](https://github.com/JochimMaene/polypix/actions/workflows/docs.yml/badge.svg)](https://github.com/JochimMaene/polypix/actions/workflows/docs.yml)

[Documentation](https://jochimmaene.github.io/polypix/) |
[PyPI](https://pypi.org/project/polypix/) |
[Repository](https://github.com/JochimMaene/polypix)

Polypix is a small Python package for workloads that already have clean
spherical footprints and need fast, NumPy-friendly coverage results. It returns
HEALPix cells whose centers fall inside each polygon.

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

## Scope

Polypix supports:

- convex spherical polygons with great-circle edges,
- longitude/latitude vertices in degrees,
- unit-vector vertices as `(x, y, z)`,
- dense and ragged polygon batches.

Polypix does not support holes, non-convex polygons, planar geometry semantics,
or conservative overlap coverage. A cell is included only when its center is
inside the polygon.

Windows wheels are not published because `healpix_cxx` is not currently
available as a conda-forge `win-64` package.

## Documentation

The public documentation is published at
<https://jochimmaene.github.io/polypix/>.

## Development

Polypix uses Pixi for the supported development environments:

- Linux x86_64
- macOS 11 or newer on Intel
- macOS 11 or newer on Apple Silicon

Install Pixi, then run:

```bash
pixi run test
```

That command creates a conda-forge based environment with Python, NumPy, CMake,
nanobind, and `healpix_cxx`, installs Polypix in editable mode without build
isolation, and runs the test suite.

To build a local wheel:

```bash
pixi run wheel
```

The local wheel bundles runtime libraries from the active build environment for
smoke testing. Publishable wheels are built by the release workflow with
`cibuildwheel` and repaired with the platform wheel repair tools.

Build the documentation site with Zensical:

```bash
pixi run docs-build
```

Preview the documentation while editing:

```bash
pixi run docs-serve
```

## License

Polypix is distributed under GPL-3.0-or-later. This reflects the binary
distribution relationship with HEALPix C++, which is GPL-2.0-or-later. See
`THIRD_PARTY_NOTICES.md` for native dependency notices.
