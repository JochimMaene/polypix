# Performance

Most performance problems in Polypix come from asking for more cells than you
need. A few choices make the biggest difference.

## Start with the resolution

Every step up in resolution gives you four times as many cells. A dense map
needs about 6 MiB at resolution 8, 96 MiB at resolution 10, and 1.5 GiB at
resolution 12. See [Resolutions](resolutions.md) for the full table.

Use the lowest resolution that still answers your question. High resolutions
are fine for small regions or selected-cell queries, but a complete dense map
gets big quickly.

## Ask for the result you need

The default result contains one cell ID for every hit. That's useful when you
need membership, but wasteful when you only need a value per cell. Use a
reducer instead:

```python
counts = px.cover_cap(
    centers_xyz,
    radii_rad,
    resolution=8,
    reduce=px.Count(),
)
```

`Count()` and `Sum(values)` do the accumulation inside the geometry operation.
They avoid building a large list of hits and then collecting it in Python.
`cover_cap(..., reduce=px.Count())` is usually the best way to count caps per
cell.

Without a selection, a reducer returns one value for every cell in the grid.
For covering calls, pass `candidate_cells` when you only need a small
selection:

```python
counts = px.cover_cap(
    centers_xyz,
    radii_rad,
    resolution=20,
    candidate_cells=site_cells,
    reduce=px.Count(),
)
```

For a small selection on a large grid, this keeps memory tied to the selection
instead of the whole grid. For a dense result that fits comfortably, the dense
path is often faster. A large selection isn't automatically better as a sparse
query, so measure if it matters.

If you already have a `Coverage`, use `Coverage.reduce()` rather than covering
the same geometry again. This is handy when several reductions share one
covering pass, or when you loaded the coverage from storage.

## Coverage mode

The default `mode="center"` includes a cell when its center is inside the
region. It is quick and predictable, but a thin region can miss cells and a
small region can return nothing.

Use `mode="overlap"` when every touched cell matters:

```python
coverage = px.cover_polygon(
    footprint_xyz,
    resolution=8,
    mode="overlap",
)
```

Overlap checks are more expensive, especially for polygons with many
vertices. Keep boundaries reasonably simple. For long, thin regions, use
`cover_sweep()` with short consecutive segments instead of one huge polygon.

## Keep large jobs in pieces

Coverage stores all cell IDs in one flat array. A parallel coverage call can
also need roughly twice the final `cells` size while it merges its work. If a
whole job might be too large, process it in batches and consume each result:

```python
for start in range(0, len(polygons), 10_000):
    coverage = px.cover_polygon(
        polygons[start : start + 10_000],
        resolution=8,
    )
    consume(coverage)
```

Don't concatenate the batches afterwards unless you have room for the complete
result. `revisit()` is also designed to work from coverage data without first
materialising every run.

## Give it NumPy arrays it can borrow

Polypix can use an aligned, C-contiguous `float64` array without copying it.
Other dtypes, layouts, and ragged inputs are converted or joined first. If
input preparation shows up in a profile, prepare dense contiguous batches
upstream.

The native work releases the GIL. Don't change an input array from another
thread while a call is running.

## Threads

Coverage operations accept `threads`. Leave it unset to let Polypix choose;
small jobs stay sequential:

```python
coverage = px.cover_polygon(batch, resolution=8)
serial = px.cover_polygon(batch, resolution=8, threads=1)
```

If your own executor is already running several Polypix calls, use
`threads=1` inside each call so the workers don't compete with each other.
Threading doesn't change the cells or their order.
