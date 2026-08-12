{.polypix-title}
# Polypix

:::{container} polypix-hero
<p class="tagline">Fast batch rasterization of spherical regions onto the HEALPix RING grid.</p>

<p class="scope">Pass Cartesian directions, caps, convex footprints, or sampled sweep edges. Polypix returns NumPy arrays with deterministic cell-center membership.</p>

<p class="polypix-actions"><a href="guide.html">Get started</a><a href="api.html">API reference</a></p>
:::

<div class="polypix-install"><span>python -m pip install polypix</span></div>

Polypix is a small native Python library for workloads where many spherical
regions must be mapped to a fixed-resolution grid. Geometry preparation stays
with the caller; Polypix handles validation, HEALPix traversal, parallel batch
execution, and result allocation.

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

- **Batch first.** One call can cover thousands of regions while preserving
  their boundaries in a compact `cells` and `offsets` representation.
- **Exact spherical primitives.** Caps and convex great-circle polygons are
  evaluated directly in direction space, including poles and longitude seams.
- **Useful fused operations.** Per-cell cap counts avoid materializing every
  cap–cell pair when explicit membership is not the result you need.
- **Small runtime surface.** NumPy is the only runtime dependency. Coordinate
  frames, propagation, plotting, interpolation, and map algebra stay in their
  established libraries.

## Case studies

The documentation build runs two larger examples from pinned inputs. Their
figures and measurements are regenerated from the checked-in source.

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
