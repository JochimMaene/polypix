<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/polypix-dark.svg">
    <img src="docs/_static/polypix.svg" alt="Polypix" height="110">
  </picture>
</p>

# Polypix

Which grid cells does this region cover? Answered for millions of satellite
footprints, swaths, and survey fields at once.

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

You have ten thousand satellite footprints, or a survey's worth of telescope
fields, and you need to know what each one covers on the ground or on the sky.
Polypix turns those regions into cell IDs on an equal-area grid — the whole
batch in one call, as plain NumPy arrays.

## Why Polypix

- **One call, not a loop.** Ten thousand regions go in as one array and come
  back as one array. There is no Python-level iteration over regions.
- **Equal-area cells.** Every cell covers exactly the same solid angle, so hit
  counts are directly comparable. No `cos(lat)` weighting before you take a
  mean, and no cells crowding together at the poles the way a
  longitude/latitude grid does.
- **Fast.** A native kernel that releases the GIL, so one call uses every core.
  The [Starlink example](https://jochimmaene.github.io/polypix/examples/communication-constellation/)
  covers 657,031 spherical caps in roughly half a second.
- **No special cases.** The poles and the date line are ordinary 3D vectors.
- **Small.** NumPy is the only dependency, and the wheels need no compiler and
  no system HEALPix library.

## Install

```bash
pip install polypix
```

Wheels cover CPython 3.12+ on Linux x86-64 and ARM64, macOS 11+ on Intel and
Apple Silicon, and Windows x86-64. NumPy is the only runtime dependency.

## Quick start

Two satellites, each looking down at a circle of ground. Polypix works in
Cartesian directions rather than longitude and latitude, so here is a short
helper to convert:

```python
import numpy as np
import polypix as px


def unit_vector(lon_deg, lat_deg):
    lon, lat = np.radians(lon_deg), np.radians(lat_deg)
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        axis=-1,
    )


centers = unit_vector([4.4, -74.1], [50.8, 4.6])  # Brussels, Bogota
radii = np.deg2rad([5.0, 8.0])                    # footprint half-angles

coverage = px.cover_cap(centers, radii, resolution=8)
print(coverage.counts)  # [1500 3829]
```

At resolution 8 a cell is roughly 25 km across, which is why the 5° footprint
lands on 1,500 of them and the 8° one on 3,829. `coverage[0]` and `coverage[1]`
hold the cell IDs for each footprint, as standard HEALPix RING indices in a
plain NumPy `uint64` array.

## What you call

| You have | Call |
| --- | --- |
| Visibility circles, elevation-mask footprints, instantaneous fields of view | `cover_cap()` |
| Scenes, frames, convex sensor footprints | `cover_footprint()` |
| The swath a sensor paints as it moves | `cover_sweep()` |
| Individual pointings, ground tracks, sample points | `cell_at()` |

All four take a whole batch at once and measure angles in radians. Results come
back as a compact `Coverage`: one flat `cells` array plus `offsets` delimiting
one segment per input region. If you only need counts per cell,
`count_caps_per_cell()` accumulates them without building the region–cell pairs
at all.

## What Polypix leaves to you

A cell counts as covered when its **center** falls inside the region, or on its
boundary. That is one test per cell rather than a partial-overlap calculation,
and a large part of why it is fast — but it means Polypix is the wrong tool if
you need every cell a region touches even slightly.

Anything upstream of the geometry stays in your own code. Polypix expects
geometry in one caller-defined Cartesian frame; orbit propagation, attitude,
sensor models, ellipsoid intersection, coordinate transforms, plotting, and map
algebra all belong in the surrounding scientific Python ecosystem.

## Documentation

Start with the [getting-started guide](https://jochimmaene.github.io/polypix/guide/),
where every code block runs and shows its real output, then use the
[API reference](https://jochimmaene.github.io/polypix/api/) for the complete
call contracts. Resolution, batching, memory, and threading are covered in
[Performance and memory](https://jochimmaene.github.io/polypix/performance/).

Two constellation studies run during every documentation build, so their maps
and timings match the code that produced them:

- [Starlink snapshot visibility](https://jochimmaene.github.io/polypix/examples/communication-constellation/)
  covers 657,031 exact service caps from a pinned catalog of 10,771 objects.
- [Earth-observation revisit](https://jochimmaene.github.io/polypix/examples/earth-observation-constellation/)
  covers 144,000 swept swath intervals from 10 satellites.

## License

Apache License 2.0. See `THIRD_PARTY_NOTICES.md` for dependency and
embedded-code notices.
