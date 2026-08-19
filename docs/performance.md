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
| `cell_centers()` | 24 bytes per cell |
| `cell_corners()` | 96 bytes per cell |
| Dense cap counts | `8 * cell_count` bytes |
| Sparse `OccupancyRuns` | `8 * (2 * represented_cell_count + 2 * run_count + 1)` bytes |
| `OccupancyStats` | `48 * represented_cell_count` bytes |

Parallel coverage builds ordered worker chunks and merges them, so peak native
memory can reach roughly twice the final `cells` array. If allocation fails you
get a `MemoryError` rather than a crash.

When the whole result will not fit, batch the input and consume each chunk:

```python
for start in range(0, len(polygons), 10_000):
    chunk = px.cover_convex_polygon(polygons[start : start + 10_000], resolution=8)
    consume(chunk)
```

Concatenating all chunks recreates the original memory requirement.

## Ask for the smallest useful result

| What you need | Use | What you avoid |
| --- | --- | --- |
| Membership per region | `cover_cap()`, `cover_convex_polygon()`, `cover_sweep()` | nothing; membership is the point |
| Counts or weighted values per cell | `into=Count()`, `into=Sum(values)` | sorting, Python accumulation, one repeated value per hit |
| Caps per cell | `cover_cap(..., into=Count())` | one cell ID per cap-cell hit, plus a `bincount()` |
| Complete occupied-bin runs | `occupancy()` | expanding and sorting every hit as an event |
| Per-cell counts and internal gaps | `occupancy(..., into=Stats())` | materializing every run to reduce it away |

## Choosing a dense or selected reduction

`Count()` and `Sum(values)` return a dense fixed-grid array when their
`cells` is `None`. That array is the requested result, so
its cost follows the resolution rather than the coverage: 384 KiB at resolution
6, 96 MiB at resolution 10, and 1.5 GiB at resolution 12. Passing `cells`
returns one value per requested ID instead and never allocates the full grid.

Sparse coverage above resolution 8 should pass `cells`. At or below that the
dense grid is small, so a `cells` query is served from a dense scratch grid and
costs about the same as the dense result; above it the query accumulates
through a hash table instead and keeps memory flat.

A reducer names the result, not the algorithm. Polypix fuses the accumulation
into the geometry kernel where that is faster and materializes membership
otherwise, returning the same array either way. Today
`cover_cap(..., into=Count())` is the case that fuses, because the cap kernel
accumulates private RING spans and never allocates cap-cell membership; polygon
and sweep reducers materialize first. The
[architecture decisions](decisions.md) carry the benchmark evidence.

`OccupancyRuns` is lossless, so it is not guaranteed to be smaller than its
input. A cell hit in alternating bins creates one run per hit. Moderate grids
use bounded dense state and write the exact cell-major result directly; sparse
high-resolution grids use a map-backed path rather than allocating by global
cell count. Size the unavoidable output with the table above.

This is the common case for a scanning constellation: a cell is observed
briefly and revisited hours later, so the run count approaches the hit count and
runs compress nothing. When the runs only feed per-cell counts and complete
internal gaps, `occupancy(..., into=Stats())` accumulates them in one pass and
allocates by represented cell instead, which is smaller by orders of magnitude
on that shape of workload. Reach for the default `Runs()` when the boundaries
themselves are the answer.

## Dense counts versus selected cells

A dense `cover_cap(..., into=Count())` consumes analytic RING spans and is
often faster than evaluating individual query cells:

```python
dense = px.cover_cap(centers_xyz, radii_rad, resolution=8, into=px.Count())
sparse = px.cover_cap(
    centers_xyz,
    radii_rad,
    resolution=20,
    into=px.Count(cells=small_site_cell_list),
)
```

Go dense whenever the array fits comfortably. Use `cells=` when the grid would
be enormous and your query set is genuinely small. Its cost grows with both the
cap count and the number of cells you ask for, so it is not a general-purpose
escape hatch.

Either way the predicate is evaluated at cell centers. If those IDs came out of
`cell_at()`, you are testing the cell, not the direction you started with.

## Two arguments that look similar

| Argument | Semantics | Output |
| --- | --- | --- |
| `candidate_cells=` | set filter for coverage | native cell order |
| `cells=` | positional reduction or cap-count query | preserves your order and duplicates |

They are not interchangeable. Candidate planning also retains normalized
geometry for the whole batch and may cache a bounded span of candidate centers,
so chunk very large batches if that retained state starts to matter.

## Geometry shape

Polygon coverage scans a conservative spherical bounding box and tests centers
against every edge, which makes compact convex footprints the fast path. A large
diagonal or pole-containing footprint can cost far more per returned cell.

For long thin regions, `cover_sweep()` keeps each interval's bounds tight. That
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
serial = px.cover_convex_polygon(batch, resolution=8, threads=1)
automatic = px.cover_convex_polygon(batch, resolution=8)  # threads=None
```

Automatic mode stays sequential below measured crossovers. `cell_at()`,
`cell_centers()`, and `cell_corners()` parallelize large arrays too, but expose no
control.

If you already run several Polypix calls concurrently from your own executor,
pass `threads=1` inside each to avoid oversubscription. Thread count never
changes membership or ordering on the same build and platform.
