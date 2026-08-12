# Concepts

## Resolution and cell IDs

Polypix uses fixed-resolution HEALPix RING ordering and calls the HEALPix order
`resolution`:

```text
nside      = 2 ** resolution
cell_count = 12 * 4 ** resolution
```

| Resolution | `nside` | Cells |
| ---: | ---: | ---: |
| 0 | 1 | 12 |
| 1 | 2 | 48 |
| 2 | 4 | 192 |
| 8 | 256 | 786,432 |
| 12 | 4,096 | 201,326,592 |

Resolutions 0 through 29 are accepted. Cell values are ordinary RING pixel
indices in `[0, cell_count)`. They are not packed tokens and do not encode a
resolution; one result carries one resolution.

Each resolution increment quadruples the cell count and approximately halves
the nominal linear cell scale. Resolution 12 already needs about 1.5 GiB for
one dense `int64` value per cell. Much higher resolutions remain useful for
sparse transforms and selected-cell queries, not complete dense maps. See
[Performance and memory](performance.md) for sizing guidance.

Outside the library: NESTED ordering, mixed-resolution cells, MOCs, neighbors,
hierarchy traversal, and map operations.

## Direction geometry

Inputs and geometry outputs are Cartesian direction vectors `(x, y, z)` on the
unit sphere. Input magnitude is ignored and normalized. The caller-defined
frame may be body-fixed for Earth or another sphere, or celestial for a sky
survey; Polypix attaches no frame, WGS84, geodetic, ellipsoid, or CRS meaning to
it and does not transform between frames.

Working in three dimensions is what makes longitude wraparound and the poles
need no special handling.

`cell_at(vectors_xyz, resolution)` quantizes one direction or a batch of
directions to standard RING IDs. `centers(cells, resolution)` returns their
representative centers; it does not reconstruct arbitrary original directions.
Every finite nonzero direction is assigned to one cell. Inputs numerically on
or extremely near a mathematical cell edge or vertex are floating-point tie
cases: results are repeatable for the same input, build, and platform, but the
API does not promise which adjacent cell owns an exact boundary direction
across platforms. This point transform does not change the center-sampling rule
for regions.

Orbit propagation, attitude, sensor projection, and ellipsoid intersection
belong upstream. Polypix can represent an already-derived circular angular
region directly as an exact spherical cap; arbitrary footprints still arrive
as their spherical boundary vectors.

## Center-sampled coverage

A cell is covered when its center lies inside a cap or footprint, or on its
boundary.
This is a single representative sample per cell, not conservative intersection,
full containment, or fractional area:

- a small or thin region whose interior misses every center returns nothing;
- a cell straddling a footprint edge is included only if its center is inside.

This is not a conservative spatial index. It may miss cells that merely
intersect a region, so it cannot by itself provide a no-false-negative
candidate set for arbitrary points or scenes.

The accepted geometry and its numerical limits are specified in the
[geometry contract](api.md#geometry-contract).

Exact caps use the same center-sampling rule. `cover_cap()` returns segmented
cell IDs, while `count_caps_per_cell()` directly accumulates how many caps
contain each center. The latter is often the right result for visibility-density
maps because it avoids materializing the same cell ID once per covering cap.

## Batches and segments

`cover_footprint()` takes one footprint, a dense batch, or a ragged sequence.
`cover_cap()` accepts one center or a flat batch with scalar or pairwise radii.
`cover_sweep()` turns consecutive pairs from two sampled edges into independent
quadrilaterals. All three return one `Coverage`:

```text
cells for item i = cells[offsets[i] : offsets[i + 1]]
```

One flat array plus offsets avoids allocating a Python object per input item
while keeping input boundaries intact. `len(coverage)` gives the item count and
`coverage[i]` returns a read-only zero-copy view of one segment. Imported
segmented arrays use the copying, validating `Coverage.from_arrays()` factory.

Sweep sampling density is part of the input contract, because consecutive samples
are joined by the shorter great-circle arc:

- steps approaching 180° bow strongly on the sphere;
- a step beyond 180° selects the opposite arc;
- an exactly ambiguous step is rejected.

Polypix cannot distinguish intentional minor-arc geometry from an undersampled
trajectory, so sample densely enough that each arc is the boundary you mean.

## Occupancy summaries

`summarize_occupancy()` consumes the segments of one or more `Coverage` results
as aligned, ordered occupancy bins. It counts consecutive runs independently
for each source, then merges all sources to measure the uncovered bins between
occupied windows. Hits in bins 0 and 2 therefore have one uncovered gap bin;
this is not a start-to-start acquisition period.

The reducer deliberately stops at ordinal steps. It does not require equal bin
durations and does not own timestamps, calendars, propagation,
variable-duration integration, or physical units. Equal duration is a caller
assertion only when converting gap steps to physical time; another application
can retain the ordinal counts.

The result is sparse and sorted by observed RING ID. At moderate resolutions a
bounded dense state machine gives maximum throughput; large sparse grids switch
to state keyed only by touched cells without changing semantics.

## Candidate cells

Pass `candidate_cells` when only a sparse existing set matters:

```python
coverage = px.cover_sweep(
    left_edge_xyz,
    right_edge_xyz,
    resolution=12,
    candidate_cells=aoi_cells,
)
```

The kernel tests those grid centers directly. Candidates have set semantics:
order and duplicates are discarded. A candidate filter remains center sampled
and does not become a conservative spatial index. Dense candidate sets can be
slower than unrestricted RING scanning.

See [Performance and memory](performance.md) for candidate planning, geometry
shape, chunking, and output sizing.

## Parallel execution

Large batches run across native worker threads with the GIL released:

```python
sequential = px.cover_footprint(batch, resolution=9, threads=1)
automatic = px.cover_footprint(batch, resolution=9)
```

`threads=None` selects the automatic policy; a positive integer sets the
worker-pool maximum, capped by the host. Calls below the measured parallel
crossover stay sequential and never initialize a pool. Results are identical
across thread settings on the same build and platform.

If an outer executor already runs several calls, use `threads=1` inside each
call to avoid nested oversubscription. `cell_at()`, `centers()`, and `corners()`
auto-parallelize large arrays without exposing thread controls. Detailed memory
and batching guidance lives in [Performance and memory](performance.md).
