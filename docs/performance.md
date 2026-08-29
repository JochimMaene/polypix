# Performance and memory

When we time a Polypix call, the measurement covers input normalization,
validation, the native work, and allocating the output. That is deliberate,
because on large jobs the size of the result you asked for usually matters more
than the speed of the geometric predicate.

## Grid size

Each step up in resolution quadruples the cell count and halves the cell size, so
a dense global map grows from 6 MiB at resolution 8 to 1.5 GiB at resolution 12.
[Resolutions](resolutions.md) has the full table.

## Sizing the result

This is the storage each public result needs, ignoring the temporary native
chunks:

| Result | Storage |
| --- | --- |
| Explicit `Coverage` | `8 * hit_count + 8 * (segment_count + 1)` bytes |
| `cell_at()` | 8 bytes per direction |
| `cell_centers()` | 24 bytes per cell |
| `cell_corners()` | 96 bytes per cell |
| `cell_neighbors()` | up to `64 * input_cell_count + 8 * (input_cell_count + 1)` bytes |
| Dense cap counts | `8 * cell_count` bytes |
| `RevisitStats` | `48 * represented_cell_count` bytes |

Parallel coverage builds ordered worker chunks and merges them, so peak native
memory can reach roughly twice the size of the final `cells` array. Coverage
result buffers use fallible allocations and raise `MemoryError` when the
allocator refuses them.

When the whole result will not fit, batch the input and consume the chunks as
they come:

```python
for start in range(0, len(polygons), 10_000):
    chunk = px.cover_polygon(polygons[start : start + 10_000], resolution=8)
    consume(chunk)
```

Concatenating the chunks afterwards, of course, recreates the original memory
requirement.

## Asking for less

| What you need | Use | What you avoid |
| --- | --- | --- |
| Membership per region | `cover_cap()`, `cover_polygon()`, `cover_sweep()` | nothing; membership is the point |
| Counts or weighted values per cell | `reduce=Count()`, `reduce=Sum(values)` | sorting, Python accumulation, one repeated value per hit |
| Caps per cell | `cover_cap(..., reduce=Count())` | one cell ID per cap-cell hit, plus a `bincount()` |
| Per-cell counts and internal gaps | `revisit()` | expanding every hit as an event, then building every run just to reduce it away |

(choosing-a-reducer)=
## Choosing a reducer

A reducer is a request for a result, not an instruction about the algorithm. We
fuse the accumulation into the geometry kernel where that is faster and build the
membership first otherwise, and the answer is identical either way.

With `cells=None`, `Count()` and `Sum(values)` return a dense array over the
whole grid. That array is the result you asked for, so its cost follows the
resolution and not the coverage: 384 KiB at resolution 6, 96 MiB at
resolution 10, 1.5 GiB at resolution 12. Passing `cells` returns one value per
requested ID instead, and never scales with the grid.

Sparse coverage above resolution 8 should pass `cells`. At or below that, a
`cells` query may still be served out of a dense scratch grid, which costs about
what the dense result would, and that is only sensible when the query and the
coverage between them touch a reasonable share of the grid. A small query against
a large grid, and every query above resolution 8, accumulates through a hash
table instead and keeps memory flat.

Counting caps is where fusing currently wins by the widest margin. The cap
kernel accumulates private RING spans and never allocates the cap-cell pairs at
all, and it does this for a dense grid and for a selection alike, falling back to
covering once and counting when it judges that cheaper. The architecture decision
records in the repository carry the benchmark evidence.

`revisit()` takes no reducer at all. It allocates by cell rather than by run, and
the reason is worth a moment. Materializing the runs is not guaranteed to be smaller than the input,
because a cell hit in alternating bins produces one run per hit. That is the
common case for a scanning constellation, where a cell is seen briefly and then
revisited hours later, so the run count approaches the hit count and the runs
compress nothing at all. Accumulating counts and complete internal gaps in one
pass is smaller by orders of magnitude on that shape of workload.

Moderate grids use bounded dense state, while sparse high-resolution grids take a
map-backed path instead of allocating by global cell count. Either way, size the
output with the table above.

`Coverage.reduce()` answers the same questions against a coverage you already
hold, which is what you want when several reductions share one expensive covering
pass, or when the coverage came back from storage:

```{literalinclude} ../examples/coverage_archive.py
:language: python
:start-after: "--8<-- [start:archive-region]"
:end-before: "--8<-- [end:archive-region]"
:dedent:
```

## Dense counts versus selected cells

A dense `cover_cap(..., reduce=Count())` consumes analytic RING spans, and is
often faster than evaluating individual query cells one at a time:

```python
dense = px.cover_cap(centers_xyz, radii_rad, resolution=8, reduce=px.Count())
sparse = px.cover_cap(
    centers_xyz,
    radii_rad,
    resolution=20,
    candidate_cells=small_site_cell_list,
    reduce=px.Count(),
)
```

Go dense whenever the array fits comfortably, and name `candidate_cells` when the
grid would be enormous and your query set is genuinely small. The selected path
costs more as either the cap count or the number of cells you ask for grows, so it
is not a general escape hatch.

With the default `mode="center"`, the predicate is evaluated at cell centers. If
those IDs came out of `cell_at()`, remember that you are testing the cell rather
than the direction you started with. `mode="overlap"` evaluates curved cell
boundaries and is correspondingly more expensive; reducers retain the same
binary per-region meaning.

## Two readings of one argument

`candidate_cells` is the only way to name a subset of the grid, and what it
means follows the operation it is given to.

| Call | Semantics | Output |
| --- | --- | --- |
| `cover_*(..., candidate_cells=)` | set filter for the coverage | native cell order |
| `cover_*(..., candidate_cells=, reduce=)` | positional query | your order, duplicates kept |
| `coverage.reduce(..., cells=)` | positional query | your order, duplicates kept |

Only the first of those restricts the scan unconditionally, since there the
selection is the result. Under a reducer the restriction cannot change the
answer, so we take it only while it is the cheaper plan. A large selection is
therefore not a mistake, only a different shape of query. Candidate planning also
holds on to normalized geometry for the whole batch, and may cache a bounded span
of candidate centers, so chunk very large batches if that retained state starts
to matter.

## Geometry shape

Polygon coverage scans a conservative spherical bounding box. Convex components
keep the existing half-space shortcut. Concave boundaries and holes use a flat
crossing check; detailed boundaries are grouped by height so a center usually
checks only nearby edges. A large diagonal footprint, one containing a pole, or
a deeply notched boundary can still cost more per cell returned.

Overlap mode reuses the center scan's one-ring latitude guard, pads each
per-ring longitude interval by one cell on either side, and tests candidate
cells against the analytical HEALPix edge curves. The cells it visits stay
proportional to the same local RING bounds, but the per-cell test does not.
A cell that no vertex lands in is checked edge by edge against all four curved
cell edges, with none of the height binning the center path uses, so the cost
of one cell is linear in the vertex count with a much larger constant than a
center predicate.

That scaling is worth planning around. Single-threaded at resolution 7 over a
circle covering roughly 6000 cells, and returning the same cells every time,
overlap mode took about 8 ms with 4 vertices, 90 ms with 64, and 708 ms with
512, against 0.4 ms for the center scan. Overlap mode is therefore for coarse
footprints; feed a detailed boundary to `cover_sweep()` in short segments
instead of passing one polygon with hundreds of vertices. Caps carry no vertex
cost and scale with their boundary alone.

Long thin regions are what `cover_sweep()` is for, since it keeps each
interval's bounds tight. Center-selected caps use analytic per-ring longitude
spans, and their dense counts skip cap-cell membership altogether.

## Input layout

Aligned, C-contiguous `float64` arrays are borrowed as they are. Anything else is
converted once, and a ragged footprint sequence is concatenated and validated
first. If input preparation shows up in your profile, feed it dense contiguous
batches. One case that surprises people: an array viewed out of a packed byte
buffer is contiguous but unaligned, which the kernel cannot borrow, so it is
copied like any other non-conforming input.

The native kernels release the GIL, so do not mutate a borrowed array from
another thread while a call is running.

## Reference benchmark

The communications example's Polypix stage is tracked by the seeded
[`test_cover_cap_dense_count_constellation_batch` benchmark][reference-benchmark].
It measures one timestamp for 10,771 caps at resolution 6 with dense
`Count()` reduction; the example repeats that call for its 61 timestamps.
Astroz propagation and file parsing are outside the benchmark.

CodSpeed's simulation mode is the canonical source for regression results. It
reports a stable, hardware-independent performance metric rather than a
promise about wall-clock time on every machine. Cite the benchmark definition
and the [Polypix CodSpeed history][codspeed-history] together, and round any
number published here to an order-of-magnitude figure such as `~10 ms`.

[reference-benchmark]: https://github.com/JochimMaene/polypix/blob/main/benchmarks/test_polypix_benchmarks.py
[codspeed-history]: https://codspeed.io/JochimMaene/polypix

## Threading

```python
serial = px.cover_polygon(batch, resolution=8, threads=1)
automatic = px.cover_polygon(batch, resolution=8)  # threads=None
```

Automatic mode stays sequential below measured crossovers. `cell_at()`,
`cell_centers()`, `cell_corners()`, and `cell_neighbors()` parallelize large
arrays as well, but they expose no control over it. Where each crossover falls
depends on the machine, so treat the batch size that starts to benefit as
something to measure rather than a fixed number.

If you already run several Polypix calls at once from your own executor, pass
`threads=1` inside each of them to avoid oversubscription. The thread count never
changes membership or ordering on one build and platform.
