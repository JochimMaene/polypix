<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/polypix-dark.svg">
    <img src="docs/_static/polypix.svg" alt="Polypix" height="110">
  </picture>
</p>

# Polypix

Batch coverage of circles, polygons, and swept paths on a HEALPix grid.

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

Polypix accepts whole batches and returns standard HEALPix RING cell IDs as
NumPy arrays. Its native kernel handles spherical geometry and can use multiple
cores for large calls.

## Install

```bash
pip install polypix
```

Wheels cover CPython 3.12+ on Linux x86-64 and ARM64, macOS 11+ on Intel and
Apple Silicon, and Windows x86-64. NumPy is the only runtime dependency.

## Quick start

Cover two spherical caps. Centers are Cartesian directions and radii are in
radians:

```python
import numpy as np
import polypix as px

centers = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
radii = np.deg2rad([5.0, 8.0])

coverage = px.cover_cap(centers, radii, resolution=8)
print(np.diff(coverage.offsets))  # [1502 3824]
```

`coverage[i]` contains the cells selected for cap `i`. Selection is based on
cell centers, not partial cell overlap.

## Documentation

Start with [Getting started](https://jochimmaene.github.io/polypix/guide/).
The site also has the [API reference](https://jochimmaene.github.io/polypix/api/),
[resolution guide](https://jochimmaene.github.io/polypix/resolutions/), and two
executable constellation studies.

## License

Apache License 2.0. See `THIRD_PARTY_NOTICES.md` for dependency and
embedded-code notices.
