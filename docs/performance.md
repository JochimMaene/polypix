# Performance and memory

Timing a Polypix call covers input normalization, validation, native work, and
output allocation. On large jobs, how big a result you asked for usually matters
more than how fast the geometric predicate is.

## Grid size

Each step up in resolution quadruples the cell count and halves the cell size,
so a dense global map goes from 6 MiB at resolution 8 to 1.5 GiB at 12.
[Resolutions](resolutions.md) has the full table.

## Size the result first

Public output storage, ignoring temporary native chunks:

| Result | Storage |
| --- | --- |
| Explicit `Coverage` | `8 * hit_count + 8 * (segment_count + 1)` bytes |
| `cell_at()` | 8 bytes per direction |
| `centers()` | 24 bytes per cell |
| `corners()` | 96 bytes per cell |
| Dense cap counts | `8 * cell_count` bytes |
| Sparse `OccupancySummary` | `32 * observed_cell_count` bytes |

Parallel coverage builds ordered worker chunks and merges them, so peak native
memory can reach roughly twice the final `cells` array. If allocation fails you
get a `MemoryError` rather than a crash.

When the whole result will not fit, batch the input and consume each chunk:

```python
for start in range(0, len(footprints), 10_000):
    chunk = px.cover_footprint(footprints[start : start + 10_000], resolution=8)
    consume(chunk)
```

Concatenating the chunks afterwards obviously gets you back to where you started.

## Ask for the smallest useful result

| What you need | Use | What you avoid |
| --- | --- | --- |
| Membership per region | `cover_cap()`, `cover_footprint()`, `cover_sweep()` | nothing; membership is the point |
| Caps per cell | `count_caps_per_cell()` | one cell ID per cap-cell hit, plus a `bincount()` |
| Runs and merged gaps over aligned bins | `summarize_occupancy()` | Python work per source and segment |

Occupancy summarization still consumes a `Coverage`, so sweep membership does
get materialized. It fuses the reduction, not the geometry. The
[architecture decisions](decisions.md) carry the benchmark evidence.

## Dense counts versus selected cells

Dense `count_caps_per_cell()` consumes analytic RING spans and is often faster
than evaluating individual query cells:

```python
dense = px.count_caps_per_cell(centers_xyz, radii_rad, resolution=8)
sparse = px.count_caps_per_cell(
    centers_xyz,
    radii_rad,
    resolution=20,
    cells=small_site_cell_list,
)
```

Go dense whenever the array fits comfortably. Use `cells=` when the grid would
be enormous and your query set is genuinely small — its cost grows with both the
cap count and the number of cells you ask for, so it is not a general-purpose
escape hatch.

Either way the predicate is evaluated at cell centers. If those IDs came out of
`cell_at()`, you are testing the cell, not the direction you started with.

## Two arguments that look similar

| Argument | Semantics | Output |
| --- | --- | --- |
| `candidate_cells=` | set filter for coverage | native cell order |
| `cells=` | positional cap-count query | preserves your order and duplicates |

They are not interchangeable. Candidate planning also retains normalized
geometry for the whole batch and may cache a bounded span of candidate centers,
so chunk very large batches if that retained state starts to matter.

## Geometry shape

Polygon coverage scans a conservative spherical bounding box and tests centers
against every edge, which makes compact convex footprints the fast path. A large
diagonal or pole-containing footprint can cost far more per returned cell.

For long thin regions, `cover_sweep()` keeps each interval's bounds tight — that
is what it is for. Caps use analytic per-ring longitude spans, and their dense
counts skip cap-cell membership entirely.

## Input layout

C-contiguous `float64` arrays are borrowed as-is. Anything else gets converted
once, and ragged footprint sequences are validated and concatenated first. If
input preparation shows up in your profile, feed it dense contiguous batches.

Native kernels release the GIL, so do not mutate a borrowed array from another
thread while a call is running.

## Threading

```python
serial = px.cover_footprint(batch, resolution=8, threads=1)
automatic = px.cover_footprint(batch, resolution=8)  # threads=None
```

Automatic mode stays sequential below measured crossovers. `cell_at()`,
`centers()`, and `corners()` parallelize large arrays too, but expose no
control.

If you already run several Polypix calls concurrently from your own executor,
pass `threads=1` inside each to avoid oversubscription. Thread count never
changes membership or ordering on the same build and platform.
