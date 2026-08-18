# Scope and stability

Polypix maps already-resolved spherical regions to HEALPix RING cells, and does
very little else. Knowing where that line sits saves you from asking it for
things it will never do.

## The boundary

```text
your model      state, propagation, attitude, sensors, terrain, frames
                    ↓  directions, caps, footprint vertices, sweep edges
Polypix         validation, RING traversal, segmented membership
                    ↓  cells, offsets
downstream      maps, joins, statistics, MOCs, plotting, storage
```

Upstream answers "what angular region is valid here, under this model?" Polypix
answers "which cell centers fall inside these regions?" Everything after that is
yours again: time, units, map algebra, persistence.

Access constraints are the case worth spelling out. A minimum elevation or
off-nadir limit depends on body radius, platform position, attitude, sensor
shape, terrain, and refraction. Resolve it yourself into a cap or a convex
footprint and hand that over. Polypix will not grow `minimum_elevation=`,
`orbit=`, or `sensor=` arguments.

## What it will not do

- physical models: propagation, attitude, clocks, ellipsoids, terrain,
  atmosphere;
- coordinate frames, WGS84, CRS, GeoJSON, or the Shapely/Astropy/Skyfield object
  models (array recipes and optional adapters are fine, runtime dependencies
  are not);
- concave polygons, holes, multipolygons, or geometry repair;
- coverage rules other than center sampling; anything conservative or
  area-based would need its own verb and its own contract;
- NESTED ordering, mixed-resolution results, MOCs, neighbors, interpolation,
  harmonics, FITS, or plotting;
- GPU, distributed, or streaming execution, and no pure-Python fallback.

Center sampling is not a conservative spatial index. A small region can contain
no cell center at all, and a region can overlap a cell whose center sits outside
it. If you need no-false-negative indexing, Polypix does not promise it today.

## What you would otherwise write

For this workflow, the usual approach today is a loop:

```python
import healpy as hp
import numpy as np

counts = np.zeros(hp.nside2npix(nside), dtype=int)
for center, radius in zip(centers, radii):
    counts[hp.query_disc(nside, center, radius, inclusive=False)] += 1
```

That is the same coverage rule Polypix uses. healpy's `inclusive=False` returns
"the exact set of pixels whose pixel centers lie within the disk", so the answer
matches; what differs is the loop. Both healpy's `query_disc` and cdshealpix's
`cone_search` take one region per call, so Python overhead grows with the number
of regions, and the cell IDs get built even when all you wanted was counts.
Polypix takes the batch in one call, and `count_caps_per_cell()` skips the IDs
entirely.

The neighbours worth knowing, if the batch is not your problem:

- **[healpy](https://healpy.readthedocs.io/)** is the reference implementation
  and does far more: maps, spherical harmonics, FITS, plotting. If you already
  depend on it and are not covering regions in bulk, you may not need anything
  else. It is GPL-2.0-only, which matters if you ship closed source.
- **[cdshealpix](https://cds-astro.github.io/cds-healpix-python/)** is the same
  shape as Polypix, a Rust core wrapped for Python, with cone, polygon and
  elliptical-cone search. It is NESTED-oriented and can report partial overlap,
  which Polypix deliberately will not.
- **[astropy-healpix](https://astropy-healpix.readthedocs.io/)** is BSD-licensed
  and much lighter than healpy, and it fits if you already live in Astropy.
- **[H3](https://h3geo.org/)** and **[S2](https://s2geometry.io/)** do a similar
  job on different tessellations. Worth a look if nothing ties you to HEALPix,
  though their cell IDs are not interchangeable with it.

Polypix makes no published speed claims against any of these yet.

## Adding things

A new feature has to serve a real workload inside the boundary above, consume
resolved geometry or standard cell IDs, keep one clear result type, and not slow
down the paths that already matter. Then it needs a reason to exist in the
library rather than in your own NumPy: either it closes a correctness gap and
removes a measured bottleneck, or it prevents a mistake people keep making.

API symmetry is not a reason. Neither is "someone might need this."

## Stability

While Polypix is on `0.x`, breaking changes are preferred over deprecation
shims and compatibility layers. Every release is tested and usable, but the
surface can move. There is no deadline for 1.0. After 1.0 it follows semantic
versioning with normal deprecation periods.

`summarize_occupancy()` is the least settled part of the API. It is correct and
benchmarked, but its exact result fields may change before 1.0.

The full design rationale, workload matrix, and evidence gates live in
[`decisions/project-goal.md`](https://github.com/JochimMaene/polypix/blob/main/decisions/project-goal.md)
alongside the other architecture decision records.
