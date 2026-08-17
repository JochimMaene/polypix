{.polypix-title}
# Polypix

:::{container} polypix-hero
<p class="tagline">Which grid cells does this region cover? Answered for millions of regions at once.</p>

<p class="scope">Give Polypix circles, polygons, or the path a moving sensor sweeps out. Get back the HEALPix cells they cover, as plain NumPy arrays.</p>

<p class="polypix-actions"><a href="guide.html">Get started</a><a href="api.html">API reference</a></p>
:::

<div class="polypix-install"><span>pip install polypix</span></div>

Say you have ten thousand satellite footprints, or a sky survey's worth of
telescope fields. For each one you need the list of grid cells it lands on.
Looping over them in Python is slow, and the geometry has awkward corners at the
poles and the date line. Polypix does the whole batch in one call.

Here are two circles on the sphere, one with a 5° radius and one with 8°, plus
the cells each of them covers:

```python
import numpy as np
import polypix as px

centers = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
radii = np.deg2rad([5.0, 8.0])

coverage = px.cover_cap(centers, radii, resolution=8)
coverage.counts
# array([1502, 3824])
```

That's the whole idea. `coverage[0]` is the cell IDs for the first circle,
`coverage[1]` for the second.

If you have not used HEALPix before, this is the grid those IDs refer to. It
starts as 12 equal-area cells covering the sphere, and every step up in
resolution splits each cell into four:

```{figure} assets/generated/sphere-levels.png
:alt: The same sphere partitioned at HEALPix resolutions 0 to 3, cell count rising from 12 to 768.
:width: 100%
:align: center

Resolution 0 through 3. Polypix goes to 29, where a cell is about 12 mm across
on the ground. [Resolutions](resolutions.md) has the whole table.
```

Polypix stops there, on purpose. Propagating orbits, pointing sensors,
intersecting an ellipsoid: all of that stays in your code, or in the libraries
you already use. Polypix picks up once you know where your regions are.

<div class="polypix-paths">
  <div>
    <h3><a href="guide.html">Start with a region</a></h3>
    <p>Circles, polygons, swept sensor paths, and turning points into cells.</p>
  </div>
  <div>
    <h3><a href="concepts.html">Understand the result</a></h3>
    <p>How cells get picked, what comes back, and how batches stay separated.</p>
  </div>
  <div>
    <h3><a href="performance.html">Plan a large run</a></h3>
    <p>Picking a resolution, sizing the output, and knowing when it won't fit.</p>
  </div>
</div>

## Why Polypix?

- **Fast**: the work happens in a native kernel that releases the GIL, so one
  call can use every core you have.
- **Batch-first**: 10,000 regions come back as two arrays, not 10,000 Python
  objects. Nothing loops in Python.
- **No awkward corners**: the poles and the date line are ordinary. Everything
  is computed with 3D vectors, so there is no seam to work around and no
  latitude where the answer quietly gets worse.
- **Skips work you don't need**: only want counts per cell? Polypix adds them up
  as it goes, instead of building every region–cell pair and reducing it
  afterwards.
- **Small**: NumPy is the only dependency. No HEALPix C library, no compiler, no
  coordinate framework to adopt.

## Case studies

Two worked examples, both run from pinned inputs every time these docs are
built. The maps and timings you see below came out of the checked-in source, so
they cannot drift from it.

<div class="example-gallery">
  <a class="example-card" href="examples/communication-constellation.html">
    <img src="generated/communications-availability.png" alt="Global Starlink visibility map">
    <div>
      <h2>Snapshot visibility</h2>
      <p>657,031 exact caps reduced directly to per-cell counts.</p>
    </div>
  </a>
  <a class="example-card" href="examples/earth-observation-constellation.html">
    <img src="generated/earth-observation-count.png" alt="Global Earth-observation count map">
    <div>
      <h2>Earth-observation revisit</h2>
      <p>144,000 swept intervals summarized into observations and revisit gaps.</p>
    </div>
  </a>
</div>

```{toctree}
:hidden:
:maxdepth: 2

guide
concepts
examples/index
api
development
```
