# Performance and memory

Timing a Polypix call includes input normalization, geometry validation, native
work, and output allocation. For large jobs, the size of the requested result
often matters more than the cost of the geometric predicate.

## Resolution and grid size

Polypix defines `nside = 2**resolution`. Increasing resolution by one
quadruples the grid and halves the nominal linear cell scale.

| Resolution | Cells | Nominal cell scale | Dense `int64` array |
| ---: | ---: | ---: | ---: |
| 4 | 3,072 | 3.66° | 24 KiB |
| 6 | 49,152 | 0.92° | 384 KiB |
| 8 | 786,432 | 0.23° | 6 MiB |
| 10 | 12,582,912 | 3.44 arcmin | 96 MiB |
| 12 | 201,326,592 | 0.86 arcmin | 1.5 GiB |

The scale is `sqrt(cell area)`, not a bound on a cell's width. Resolutions up
to 29 are accepted because sparse transforms and selected-cell queries can use
very large IDs; a complete grid is not practical at those resolutions.

## Account for the result first

Approximate public output storage, excluding temporary native chunks:

| Result | Storage |
| --- | --- |
| Explicit `Coverage` | `8 * hit_count + 8 * (segment_count + 1)` bytes |
| `cell_at()` | 8 bytes per direction |
| `centers()` | 24 bytes per cell |
| `corners()` | 96 bytes per cell |
| Dense cap counts | `8 * cell_count` bytes |
| Sparse `OccupancySummary` | `32 * observed_cell_count` bytes |

Parallel explicit coverage builds ordered worker chunks and merges them. Peak
native output memory can therefore approach twice the final `cells` array.
Allocation failures while materializing coverage raise `MemoryError`.

If a complete explicit result is too large, batch the input regions and consume
each result before continuing:

```python
for start in range(0, len(footprints), 10_000):
    chunk = px.cover_footprint(footprints[start : start + 10_000], resolution=8)
    consume(chunk)
```

Concatenating all chunks afterward does not reduce final output memory.

## Avoid materialized intermediates

Choose the result closest to the downstream question:

| Needed result | Preferred path | Avoided intermediate |
| --- | --- | --- |
| Explicit membership per region | `cover_cap()`, `cover_footprint()`, or `cover_sweep()` | None; membership is the result |
| Number of caps per cell | `count_caps_per_cell()` | One cell ID per cap-cell hit plus `bincount()` |
| Runs and merged gaps over aligned bins | `summarize_occupancy()` | Python work per source and segment |

Occupancy summarization consumes `Coverage`, so sweep membership is still
materialized. It fuses the measured temporal reduction, not the geometry and
reducer into one domain-specific operation. See the
[architecture decisions](decisions.md) for the benchmark evidence.

## Dense counts versus selected cells

Dense `count_caps_per_cell()` consumes analytic RING span endpoints. It is
often faster than evaluating many individual query cells:

```python
dense = px.count_caps_per_cell(centers_xyz, radii_rad, resolution=8)
sparse = px.count_caps_per_cell(
    centers_xyz,
    radii_rad,
    resolution=20,
    cells=small_site_cell_list,
)
```

Use dense mode when the full array comfortably fits. Use `cells=` when the grid
would be too large and the requested set is genuinely small. In both cases the
predicate is evaluated at HEALPix cell centers, not at original directions
that may have been quantized with `cell_at()`.

## Candidate cells

`candidate_cells=` restricts coverage to a **set of grid centers**. Inputs are
sorted and deduplicated when necessary; candidate order and duplicates have no
output meaning. This differs from the positional `cells=` argument to cap
counting:

| Argument | Semantics | Output alignment |
| --- | --- | --- |
| `candidate_cells=` | Set filter for explicit coverage | Native cell order |
| `cells=` | Positional cap-count query | Preserves order and duplicates |

Candidate filtering is useful for a sparse existing AOI. A dense candidate set
can be slower than unrestricted RING scanning. Selection still happens at cell
centers, so cells that only intersect the AOI by area can be absent.

Candidate planning retains normalized geometry for the batch and may cache a
bounded span of candidate centers. Chunk very large batches when this retained
preparation state matters.

## Geometry shape affects polygon work

Polygon coverage scans a conservative spherical bounding box and tests cell
centers against every edge. Compact convex footprints are the intended fast
path. Large diagonal or pole-containing footprints can require much more work
per returned cell.

For elongated sampled regions, `cover_sweep()` keeps each interval's bounds
tight. Sampling density remains part of the geometry contract because samples
are connected by minor great-circle arcs.

Exact caps use analytic per-ring longitude spans. Their dense counts avoid
expanding cap-cell membership entirely.

## Input layout

Compatible C-contiguous `float64` direction and geometry arrays are borrowed
without conversion. Non-contiguous or other real numeric inputs are converted
once. Ragged footprint sequences are validated and concatenated. Prefer dense
contiguous batches when input preparation is visible in a high-rate workload.

Do not mutate borrowed arrays from another Python thread while a Polypix call
is running; native kernels release the GIL.

## Threading

Coverage and cap-count operations accept `threads=`:

```python
serial = px.cover_footprint(batch, resolution=8, threads=1)
automatic = px.cover_footprint(batch, resolution=8)  # threads=None
```

Automatic mode stays sequential below measured crossovers. `cell_at()`,
`centers()`, and `corners()` also auto-parallelize large arrays but expose no
thread control.

If an outer executor already runs several Polypix calls concurrently, use
`threads=1` inside each call to avoid nested oversubscription. Thread settings
do not change result membership or ordering on the same build and platform.
