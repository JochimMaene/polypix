---
html_theme.sidebar_secondary.remove: true
---

{.polypix-title}
# Polypix

:::{container} polypix-hero
<p class="tagline">Which grid cells does this region cover? Answered for millions of satellite footprints, swaths, and survey fields at once.</p>

{.polypix-badges}
[![PyPI](https://img.shields.io/pypi/v/polypix.svg)](https://pypi.org/project/polypix/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB.svg?logo=python&logoColor=white)](https://pypi.org/project/polypix/)
[![License](https://img.shields.io/pypi/l/polypix.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Tests](https://github.com/JochimMaene/polypix/actions/workflows/run-tests.yml/badge.svg)](https://github.com/JochimMaene/polypix/actions/workflows/run-tests.yml)
[![Benchmarks](https://github.com/JochimMaene/polypix/actions/workflows/codspeed.yml/badge.svg)](https://github.com/JochimMaene/polypix/actions/workflows/codspeed.yml)
:::

You have ten thousand satellite footprints, or a survey's worth of telescope
fields, and you need to know what each one covers on the ground or on the sky.
Polypix turns those regions into cell IDs on an equal-area grid — the whole
batch in one call.

## Why Polypix

- **One call, not a loop.** Ten thousand regions go in as one array and come
  back as one array. There is no Python-level iteration over regions.
- **Equal-area cells.** Every cell covers exactly the same solid angle, so hit
  counts are directly comparable. No `cos(lat)` weighting before you take a
  mean, and no cells crowding together at the poles the way a
  longitude/latitude grid does.
- **Fast.** A native kernel that releases the GIL, so one call uses every core.
  The [Starlink example](examples/communication-constellation.md) covers 657,031
  spherical caps in roughly half a second.
- **No special cases.** The poles and the date line are ordinary 3D vectors.
- **Small.** NumPy is the only dependency, and the wheels need no compiler and
  no system HEALPix library.

## The grid

Polypix answers on a HEALPix grid. It divides the sphere into cells of
**exactly equal area**, starting from 12 cells and splitting each one into four
at every step up in resolution. Each cell has an integer ID, and those IDs are
what Polypix gives back.

Equal area is the property that matters for analysis. Count how many satellites
see each cell, and you can compare those counts, average them, or histogram
them directly — every cell represents the same amount of sky or ground. The
same count on a longitude/latitude grid has to be area-weighted first, because
its cells shrink towards the poles.

```{figure} assets/generated/sphere-levels.png
:alt: The same sphere partitioned at HEALPix resolutions 0 to 3, cell count rising from 12 to 768.
:width: 100%
:align: center

Resolutions 0 to 3. Polypix goes to 29, where a cell is about 12 mm across on
the ground. [Resolutions](resolutions.md) has the whole table.
```

## A first example

Here are two satellites, each looking down at a circle of ground. Polypix works
in Cartesian directions rather than longitude and latitude, so here is a short
helper to convert:

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
>>> coverage.counts
array([1500, 3829])
```

That single call covered both footprints. At resolution 8 a cell is roughly
25 km across, which is why the 5° footprint lands on 1,500 of them and the 8°
one on 3,829. The cell IDs for each footprint come back as `coverage[0]` and
`coverage[1]`:

```{doctest}
>>> coverage[0][:4]
array([68085, 68086, 68087, 68088], dtype=uint64)
```

These are standard HEALPix RING indices in a plain NumPy `uint64` array.

To run that yourself:

```bash
pip install polypix
```

NumPy is the only runtime dependency. [Installation](install.md) covers wheels
and source builds; [Getting started](guide.md) walks through the other region
shapes.

## What that looks like at scale

The same two calls, run over real constellations. Both studies rebuild on every
documentation build, so their maps and timings come from the code shown on the
page.

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
      <p>Ten days of Earth-observation coverage, mapped as revisit time.</p>
    </div>
  </a>
</div>

## What Polypix leaves to you

A cell counts as covered when its center falls inside your region, so Polypix
is the wrong tool if you need every cell a region touches even slightly.
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
