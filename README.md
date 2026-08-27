<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://jochimmaene.github.io/polypix/_static/polypix-dark.svg">
    <img src="https://jochimmaene.github.io/polypix/_static/polypix.svg" alt="Polypix" height="110">
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

Hand Polypix a batch of spherical regions and it gives you back the HEALPix
cells they cover, as standard RING IDs in NumPy arrays. The spherical geometry
runs in a native kernel, which uses several cores for large calls. Use the
result for coverage maps, visibility counts, and revisit analysis.
Polygon coverage also accepts GeoJSON-like mappings and objects implementing
`__geo_interface__`, including individual Shapely geometries.

## How many Starlink satellites can you see?

[![Global map of the mean number of catalogued Starlink objects in view](https://jochimmaene.github.io/polypix/generated/communications-availability.png)](https://jochimmaene.github.io/polypix/examples/communication-constellation.html)

All 10,771 catalogued Starlink objects, propagated for an hour and counted cell
by cell. Two dense bands around 40° north and south see 70 to 90 objects at
once, the equator about 37, the poles about 20. Behind it are 137 million
cap-cell hits, counted by one `cover_cap()` call per timestamp, never stored, and
done in well under a second.

[Read the case study](https://jochimmaene.github.io/polypix/examples/communication-constellation.html) for the
propagation, the service-cap geometry, and the timings.

## Install

```bash
pip install polypix
```

There are wheels for CPython 3.12 and newer on Linux x86-64 and ARM64, macOS 11+
on Intel and Apple Silicon, and Windows x86-64. NumPy is the only runtime
dependency.

## Quick start

Two satellites overhead, each serving a circle on the ground. Centers are
Cartesian directions and radii are in radians:

```python
import numpy as np
import polypix as px

centers = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
radii = np.radians([6.0, 4.0])

coverage = px.cover_cap(centers, radii, resolution=6)
print(np.diff(coverage.offsets))  # [134  56]
```

`coverage[i]` holds the cells covered by satellite `i`. A cell is covered when
its center falls inside the circle, so partial overlap at the boundary does not
count.

Ask for counts instead and the region-cell pairs are never built, which is how
the map above is made:

```python
counts = px.cover_cap(centers, radii, resolution=6, reduce=px.Count())
print(counts.shape)  # (49152,) one value per cell on the grid
```

## Documentation

Start with [Getting started](https://jochimmaene.github.io/polypix/guide.html). The
site also carries the [API reference](https://jochimmaene.github.io/polypix/api.html),
a [resolution guide](https://jochimmaene.github.io/polypix/resolutions.html), and two
constellation case studies that run end to end.

## License

Apache License 2.0. See `THIRD_PARTY_NOTICES.md` for the dependency and
embedded-code notices.
