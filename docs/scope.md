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
answers "which cell centers fall inside these regions?" Everything after that —
time, units, map algebra, persistence — is yours again.

Access constraints are the case worth spelling out. A minimum elevation or
off-nadir limit depends on body radius, platform position, attitude, sensor
shape, terrain, and refraction. Resolve it yourself into a cap or a convex
footprint and hand that over. Polypix will not grow `minimum_elevation=`,
`orbit=`, or `sensor=` arguments.

## What it will not do

- physical models: propagation, attitude, clocks, ellipsoids, terrain,
  atmosphere;
- coordinate frames, WGS84, CRS, GeoJSON, or the Shapely/Astropy/Skyfield object
  models — array recipes and optional adapters are fine, runtime dependencies
  are not;
- concave polygons, holes, multipolygons, or geometry repair;
- coverage rules other than center sampling; anything conservative or
  area-based would need its own verb and its own contract;
- NESTED ordering, mixed-resolution results, MOCs, neighbors, interpolation,
  harmonics, FITS, or plotting;
- GPU, distributed, or streaming execution, and no pure-Python fallback.

Center sampling is not a conservative spatial index. A small region can contain
no cell center at all, and a region can overlap a cell whose center sits outside
it. If you need no-false-negative indexing, Polypix does not promise it today.

## Adding things

A new feature has to serve a real workload inside the boundary above, consume
resolved geometry or standard cell IDs, keep one clear result type, and not slow
down the paths that already matter. Then it needs a reason to exist in the
library rather than in your own NumPy: either it closes a correctness gap and
removes a measured bottleneck, or it prevents a mistake people keep making.

API symmetry is not a reason. Neither is "someone might need this."

## Stability

While Polypix is on `0.x`, breaking changes are preferred over deprecation
shims and compatibility layers — every release is tested and usable, but the
surface can move. There is no deadline for 1.0. After 1.0 it follows semantic
versioning with normal deprecation periods.

`summarize_occupancy()` is the least settled part of the API. It is correct and
benchmarked, but its exact result fields may change before 1.0.

The full design rationale, workload matrix, and evidence gates live in
[`decisions/project-goal.md`](https://github.com/JochimMaene/polypix/blob/main/decisions/project-goal.md)
alongside the other architecture decision records.
