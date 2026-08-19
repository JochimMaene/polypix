---
html_theme.sidebar_secondary.remove: true
---

{.polypix-title}
# Polypix

:::{container} polypix-hero
<p class="tagline">Batch coverage of satellite footprints, swaths, and survey fields on an equal-area grid.</p>

{.polypix-badges}
[![PyPI](https://img.shields.io/pypi/v/polypix.svg)](https://pypi.org/project/polypix/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB.svg?logo=python&logoColor=white)](https://pypi.org/project/polypix/)
[![License](https://img.shields.io/pypi/l/polypix.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Tests](https://github.com/JochimMaene/polypix/actions/workflows/run-tests.yml/badge.svg)](https://github.com/JochimMaene/polypix/actions/workflows/run-tests.yml)
[![Benchmarks](https://github.com/JochimMaene/polypix/actions/workflows/codspeed.yml/badge.svg)](https://github.com/JochimMaene/polypix/actions/workflows/codspeed.yml)
:::

Polypix turns batches of spherical regions into standard HEALPix cell IDs. Use
the result for coverage maps, visibility counts, and revisit analysis.

## Why Polypix

- **Batch-first.** Arrays of regions go in; compact NumPy arrays come back.
- **Equal area.** Counts can be compared without latitude weighting.
- **Native execution.** Large calls can use multiple cores while the GIL is
  released.
- **Small runtime.** NumPy is the only dependency. Wheels include the native
  kernel.

## The grid

Polypix answers on a HEALPix grid. It divides the sphere into cells of
**exactly equal area**, starting from 12 cells and splitting each one into four
at every step up in resolution. Each cell has an integer ID, and those IDs are
what Polypix gives back.

Because the cells have equal area, per-cell counts can be compared directly.

```{figure} assets/generated/sphere-levels.png
:alt: The same sphere partitioned at HEALPix resolutions 0 to 3, cell count rising from 12 to 768.
:width: 100%
:align: center

Resolutions 0 to 3. Polypix goes to 29, where a cell is about 12 mm across on
the ground. [Resolutions](resolutions.md) has the whole table.
```

## A first example

Here are two circular ground footprints centered near Brussels and Bogotá.
Polypix takes Cartesian directions, so the example first converts longitude and
latitude:

```{doctest}
>>> import numpy as np
>>> import polypix as px

>>> def unit_vector(lon_deg, lat_deg):
...     lon, lat = np.radians(lon_deg), np.radians(lat_deg)
...     return np.stack(
...         [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
...         axis=-1,
...     )

>>> centers = unit_vector([4.4, -74.1], [50.8, 4.6])   # Brussels, Bogota
>>> radii = np.deg2rad([5.0, 8.0])                     # footprint half-angles

>>> coverage = px.cover_cap(centers, radii, resolution=8)
>>> coverage.segment_sizes
array([1500, 3829])
```

At resolution 8 a nominal cell is about 25 km across. The cell IDs for each
footprint are available as `coverage[0]` and `coverage[1]`:

```{doctest}
>>> coverage[0][:4]
array([68085, 68086, 68087, 68088])
```

These are standard HEALPix RING indices in a NumPy `int64` array.

To run that yourself:

```bash
pip install polypix
```

NumPy is the only runtime dependency. [Installation](install.md) covers wheels
and source builds; [Getting started](guide.md) walks through the other region
shapes.

## What that looks like at scale

Two executable studies show the same operations at constellation scale.

<div class="example-gallery">
  <a class="example-card" href="examples/communication-constellation.html">
    <img src="generated/communications-availability.png" alt="Global Starlink visibility map">
    <div>
      <h2>How many satellites can you see?</h2>
      <p>A one-hour Starlink snapshot, mapped cell by cell from real orbital data.</p>
    </div>
  </a>
  <a class="example-card" href="examples/earth-observation-constellation.html">
    <img src="generated/earth-observation-count.png" alt="Global Earth-observation count map">
    <div>
      <h2>How often does a satellite fly over?</h2>
      <p>Ten days of sampled coverage, mapped as observed-cell internal gaps.</p>
    </div>
  </a>
</div>

## What Polypix leaves to you

A cell counts as covered when its center falls inside your region. Polypix does
not return every cell touched by the boundary.
[Center-sampled coverage](concepts.md#center-sampled-coverage) shows exactly
what that includes and excludes.

Anything upstream of the geometry stays in your own code. Orbit propagation,
attitude, sensor models, and ellipsoid intersection all happen before Polypix
sees anything, and you hand it the caps or footprints that come out.

```{toctree}
:caption: Guides
:hidden:
:maxdepth: 2

guide
install
concepts
resolutions
performance
```

```{toctree}
:caption: Examples
:hidden:

examples/communication-constellation
examples/earth-observation-constellation
```

```{toctree}
:caption: Reference
:hidden:
:maxdepth: 2

api
release-notes
development
```
