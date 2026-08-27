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

How much of the ground did this constellation actually see, and how often did it
come back? Which cells did a ten-day survey never reach?

Polypix helps answer those questions You hand it batches of spherical regions and it hands back standard HEALPix cell IDs, which you then use for coverage maps, visibility counts, and revisit analysis.

Polygon coverage accepts Cartesian directions as well as GeoJSON-like mappings
and individual Shapely or other `__geo_interface__` geometries.

## Why Polypix

The library was written for mission analysis, and a few decisions follow from
that:

- Everything takes batches. Whole arrays of regions go in, and what comes back
  is a NumPy array, not a Python object per region.
- The grid has cells of equal area, so per-cell counts can be compared without
  weighting anything by latitude.
- It is fast. The geometry runs in Rust, and large calls release the GIL and use
  several cores. Both case studies below cover millions of cells in well under a
  second.
- NumPy is the only runtime dependency. The wheels carry the compiled kernel, so
  there is no system HEALPix library to find and no compiler to arrange.

## The grid

Polypix uses a HEALPix grid. HEALPix divides the sphere into cells of exactly equal area, starting from 12 cells and splitting each one into four at
every step up in resolution. Each cell has an integer ID, and those IDs are what Polypix gives back.

```{figure} assets/generated/sphere-levels.png
:alt: The same sphere partitioned at HEALPix resolutions 0 to 3, cell count rising from 12 to 768.
:width: 100%
:align: center

Resolutions 0 to 3. Polypix goes to 29, where a cell is about 12 mm across on
the ground. [Resolutions](resolutions.md) has the whole table.
```

## A first example

Suppose you have two circular ground footprints, one near Brussels and one near Bogotá. Polypix works in Cartesian directions instead of longitude and latitude, so the example converts them on the way in:

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
>>> np.diff(coverage.offsets)
array([1500, 3829])
```

At resolution 8 a typical cell is about 25 km across, which is why the smaller
footprint came back with 1500 cells and the larger one with 3829. The IDs themselves are available per footprint, as `coverage[0]` and `coverage[1]`:

```{doctest}
>>> coverage[0][:4]
array([68085, 68086, 68087, 68088])
```

Those are standard HEALPix RING indices in a NumPy `int64` array.

To run the example yourself:

```bash
pip install polypix
```

[Installation](install.md) covers the wheels and source builds, and
[Getting started](guide.md) walks through the other region shapes.

## What that looks like at scale

Both of the case studies below run the same handful of operations at
constellation scale. Each one is measured while the documentation is built, so
the timings on those pages come from the same run that produced their figures.

<div class="example-gallery">
  <a class="example-card" href="examples/communication-constellation.html">
    <img src="generated/communications-availability.png" alt="Global Starlink visibility map">
    <div>
      <h2>How many Starlink satellites can you see?</h2>
      <p>All 10,771 catalogued objects, propagated for an hour and counted cell by cell.</p>
    </div>
  </a>
  <a class="example-card" href="examples/earth-observation-constellation.html">
    <img src="generated/earth-observation-revisit.png" alt="Global map of mean time between Sentinel-2 overflights">
    <div>
      <h2>What is Sentinel-2's revisit time?</h2>
      <p>Three real spacecraft, 14 days of swaths, and the wait between overflights cell by cell.</p>
    </div>
  </a>
</div>

## What Polypix does not do

One thing is worth knowing before you get any further. A cell counts as covered
when its center falls inside your region, which means Polypix does not return
every cell the boundary touches.
[Center-sampled coverage](concepts.md#center-sampled-coverage) shows exactly what
that rule includes and leaves out.

Everything upstream of the geometry stays in your own code. Orbit propagation,
attitude, sensor models, and ellipsoid intersection all happen before Polypix
sees anything; what you hand over are the caps or footprints that come out of
them.

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
