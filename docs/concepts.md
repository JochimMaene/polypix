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

Outside the library: NESTED ordering, mixed-resolution cells, MOCs, neighbors,
hierarchy traversal, and map operations.

## Body-centered geometry

Inputs and geometry outputs are body-centered `(x, y, z)` vectors on the unit
sphere. Input magnitude is ignored and normalized. The frame may represent Earth
or any other sphere; Polypix attaches no WGS84, geodetic, ellipsoid, or CRS
meaning to it.

Working in three dimensions is what makes longitude wraparound and the poles
need no special handling.

Footprint generation — orbit propagation, attitude, sensor projection, ellipsoid
intersection — belongs upstream.

## Center-sampled coverage

A cell is covered when its center lies inside a footprint or on its boundary.
This is a single representative sample per cell, not conservative intersection,
full containment, or fractional area:

- a large footprint whose interior misses every center returns nothing;
- a cell straddling a footprint edge is included only if its center is inside.

The accepted geometry and its numerical limits are specified in the
[geometry contract](api.md#geometry-contract).

## Batches and segments

`cover_footprint()` takes one footprint, a dense batch, or a ragged sequence.
`cover_strip()` turns consecutive pairs from two sampled edges into independent
quadrilaterals. Both return one `Coverage`:

```text
cells for item i = cells[offsets[i] : offsets[i + 1]]
```

One flat array plus offsets avoids allocating a Python object per footprint while
keeping input boundaries intact.

Strip sampling density is part of the input contract, because consecutive samples
are joined by the shorter great-circle arc:

- steps approaching 180° bow strongly on the sphere;
- a step beyond 180° selects the opposite arc;
- an exactly ambiguous step is rejected.

Polypix cannot distinguish intentional minor-arc geometry from an undersampled
trajectory, so sample densely enough that each arc is the boundary you mean.

## Candidate cells

Pass `candidate_cells` when only a sparse existing set matters:

```python
coverage = px.cover_strip(
    left_edge_xyz,
    right_edge_xyz,
    resolution=12,
    candidate_cells=aoi_cells,
)
```

The kernel tests those centers directly instead of materializing global coverage
first. Two costs come with it:

- Normalized vertices and edge normals are retained for the whole batch while
  shared candidate ranges are planned, so peak memory grows with batch size as
  well as candidate count. For very large batches this can exceed the streaming
  full-scan path.
- When many footprints revisit the same candidates, the kernel may cache center
  vectors for the bounding candidate span at 24 bytes per candidate, capped at
  64 MiB. Larger spans fall back to on-demand reconstruction.

Full scans use one conservative longitude bound per footprint. That is fast for
small footprints and short strip segments, but work follows the spherical
bounding box rather than the output size, so large diagonal or pole-containing
footprints cost disproportionately more per returned cell. Per-ring edge
intersection stays deferred until such footprints are a measured primary
workload.

For elongated swaths, prefer `cover_strip()` with dense samples over one large
diagonal polygon; short per-segment footprints keep the scan bounds much
tighter.

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

Parallel execution builds ordered per-worker chunks and concatenates them, so
while merging a very large result peak native memory can approach twice the
final `cells` array. Use `threads=1` when that matters more than throughput.
