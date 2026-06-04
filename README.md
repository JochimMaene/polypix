# Polypix

Fast HEALPix center-in-polygon coverage for convex spherical polygons.

Polypix takes one polygon or a batch of polygons on the unit sphere and returns
the HEALPix cells whose centers fall inside each polygon. It is meant for
throughput-oriented workloads where the input is already a valid convex
spherical polygon.

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

## Install

Polypix supports Python 3.12 and newer on Linux x86_64, macOS Intel, and macOS
Apple Silicon:

```bash
python -m pip install polypix
```

Windows wheels are not enabled yet because `healpix_cxx` is not currently
available as a conda-forge `win-64` package.

## Development

Polypix uses Pixi for the supported development environments:

- Linux x86_64
- macOS Intel
- macOS Apple Silicon

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
