{.polypix-title}
# Polypix

:::{container} polypix-hero
<p class="tagline">Fast batch rasterization of spherical regions onto the HEALPix RING grid.</p>

<p class="scope">Pass Cartesian directions, caps, convex footprints, or sampled sweep edges. Polypix returns NumPy arrays with deterministic cell-center membership.</p>

<p class="polypix-actions"><a href="guide.html">Get started</a><a href="api.html">API reference</a></p>
:::

<div class="polypix-install"><span>pip install polypix</span></div>

If you work with satellite footprints, sensor beams, or survey fields, sooner or
later you need to know which grid cells each region touches. Polypix answers that
for a whole batch at once: hand it Cartesian directions, get back NumPy arrays of
HEALPix RING cell IDs.

It stops there, on purpose. Propagating orbits, pointing sensors, intersecting an
ellipsoid — that stays in your code, or in the libraries you already use. Polypix
picks up once you have directions on a sphere.

```python
import numpy as np
import polypix as px

centers = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
radii = np.deg2rad([5.0, 8.0])

coverage = px.cover_cap(centers, radii, resolution=8)
coverage.counts
# array([1502, 3824])
```

<div class="polypix-paths">
  <div>
    <h3><a href="guide.html">Start with a region</a></h3>
    <p>Caps, convex footprints, sampled sweeps, and direction-to-cell indexing.</p>
  </div>
  <div>
    <h3><a href="concepts.html">Understand the result</a></h3>
    <p>Center sampling, RING cell IDs, segmented batches, and occupancy bins.</p>
  </div>
  <div>
    <h3><a href="performance.html">Plan a large run</a></h3>
    <p>Resolution, output size, sparse queries, batching, and native threads.</p>
  </div>
</div>

## Why Polypix?

- **One call, thousands of regions.** You get back two flat arrays — `cells` and
  `offsets` — so a batch of 10,000 footprints costs you two allocations, not
  10,000 Python objects.
- **Poles and the date line are not special cases.** Caps and great-circle
  polygons are evaluated in direction space, so there is no seam to work around
  and no latitude where the answer quietly degrades.
- **Skip the middle step when you can.** If you only need counts per cell,
  `count_caps_per_cell()` accumulates them directly instead of building every
  cap–cell pair and reducing it afterwards.
- **NumPy, and nothing else.** Frames, propagation, plotting, interpolation, and
  map algebra all have good homes already. Polypix stays out of them.

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
