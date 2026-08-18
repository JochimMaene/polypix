---
html_theme.sidebar_secondary.remove: true
---

{.polypix-title}
# Polypix

:::{container} polypix-hero
<p class="tagline">Which grid cells does this region cover? Answered for a whole batch in one call.</p>

{.polypix-badges}
[![PyPI](https://img.shields.io/pypi/v/polypix.svg)](https://pypi.org/project/polypix/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB.svg?logo=python&logoColor=white)](https://pypi.org/project/polypix/)
[![License](https://img.shields.io/pypi/l/polypix.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Tests](https://github.com/JochimMaene/polypix/actions/workflows/run-tests.yml/badge.svg)](https://github.com/JochimMaene/polypix/actions/workflows/run-tests.yml)
[![Benchmarks](https://github.com/JochimMaene/polypix/actions/workflows/codspeed.yml/badge.svg)](https://github.com/JochimMaene/polypix/actions/workflows/codspeed.yml)

<p class="polypix-actions"><a href="guide.html">Get started</a><a href="api.html">API reference</a></p>

<div class="polypix-install"><span>pip install polypix</span></div>
:::

Say you have ten thousand satellite footprints, or a sky survey's worth of
telescope fields, and you need the grid cells each one lands on. Looping in
Python is slow, and the geometry gets awkward at the poles and the date line.

Two circles, one with a 5° radius and one with 8°:

```{doctest}
>>> import numpy as np
>>> import polypix as px

>>> centers = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
>>> radii = np.deg2rad([5.0, 8.0])
>>> coverage = px.cover_cap(centers, radii, resolution=8)

>>> coverage.counts
array([1502, 3824])
>>> coverage[0][:4]
array([358912, 358913, 358914, 359934], dtype=uint64)
```

One call, both circles. `coverage[0]` holds the cell IDs for the first,
`coverage[1]` the second.

New to HEALPix? This is the grid those IDs refer to. It starts as 12 equal-area
cells and splits each one into four at every step up in resolution:

```{figure} assets/generated/sphere-levels.png
:alt: The same sphere partitioned at HEALPix resolutions 0 to 3, cell count rising from 12 to 768.
:width: 100%
:align: center

Polypix goes to resolution 29, where a cell is about 12 mm across on the ground.
[Resolutions](resolutions.md) has the whole table.
```

Polypix stops there, on purpose. Orbits, pointing, ellipsoid intersection: that
stays in your code, or in the libraries you already use.

## Why Polypix?

- **Fast**: a native kernel that releases the GIL, so one call uses every core.
  See real timings in the [examples](#examples).
- **Batch-first**: 10,000 regions come back as two arrays, not 10,000 objects.
- **No special cases**: the poles and the date line are ordinary 3D vectors.
- **No wasted work**: if you only need counts per cell, it never builds the pairs.
- **Small**: NumPy is the only dependency.

## Examples

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

```{toctree}
:caption: Guides
:hidden:
:maxdepth: 2

guide
install
concepts
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
