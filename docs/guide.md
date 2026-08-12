# Guide and recipes

Polypix has a small surface because each operation answers one direction-space
question. Start from the data you already have and the result you need.

## Choose an operation

| You have | You need | Use |
| --- | --- | --- |
| Cartesian directions | One RING cell per direction | `cell_at()` |
| Cap centers and angular radii | Explicit cells per cap | `cover_cap()` |
| Convex spherical boundaries | Explicit cells per footprint | `cover_footprint()` |
| Two sampled boundary curves | Explicit cells per swept interval | `cover_sweep()` |
| Many caps | Number of caps covering each cell center | `count_caps_per_cell()` |
| Imported segmented cell arrays | A validated result object | `Coverage.from_arrays()` |
| Aligned occupancy bins from independent sources | Source runs and merged gaps | `summarize_occupancy()` (provisional in 0.x) |

All region operations use **cell-center inclusion**. They rasterize a spherical
region; they do not return every cell whose area intersects it.

## Assign directions to cells

Use `cell_at()` for catalog objects, Monte Carlo directions, event samples, or
fixed sites that need ordinary RING IDs:

```python
import numpy as np
import polypix as px

directions = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [-2.0, 1.0, 0.5],
    ]
)
cells = px.cell_at(directions, resolution=8)
```

Input magnitude is ignored. `cell_at()` quantizes each direction to a cell;
`centers(cells, resolution)` returns the representative cell centers, not the
original directions. The exact center round trip is:

```python
sample = np.asarray([0, 17, 123], dtype=np.uint64)
assert np.array_equal(px.cell_at(px.centers(sample, 3), 3), sample)
```

Passing these IDs to another Polypix call evaluates their **cell centers**. It
does not preserve an exact predicate on the original directions.

## Cover circular and convex regions

An exact spherical cap is defined by a direction and an angular radius in
radians:

```python
cap = px.cover_cap([1.0, 0.0, 0.0], np.deg2rad(5.0), resolution=8)
cap_cells = cap[0]
```

Use a footprint for a convex boundary whose edges are minor great-circle arcs.
A ragged Python sequence is accepted when footprints have different vertex
counts:

```python
triangles_and_quads = [
    np.asarray([[1.0, -0.1, -0.1], [1.0, 0.1, -0.1], [1.0, 0.0, 0.1]]),
    np.asarray(
        [[1.0, -0.2, -0.1], [1.0, 0.2, -0.1], [1.0, 0.2, 0.1], [1.0, -0.2, 0.1]]
    ),
]
coverage = px.cover_footprint(triangles_and_quads, resolution=8)

for footprint_cells in coverage:
    print(footprint_cells.size)  # one read-only NumPy view per input footprint
```

For maximum batch throughput, prefer a C-contiguous `float64` dense array when
every footprint has the same vertex count. Ragged sequences are normalized and
concatenated before entering the native kernel.

## Cover a paired-edge sweep

Two sample-aligned curves define one quadrilateral per adjacent sample pair:

```python
left = np.asarray([[1.0, -0.1, -0.1], [1.0, -0.1, 0.1], [1.0, -0.1, 0.3]])
right = np.asarray([[1.0, 0.1, -0.1], [1.0, 0.1, 0.1], [1.0, 0.1, 0.3]])

swept = px.cover_sweep(left, right, resolution=8)
assert len(swept) == 2
```

A zero-motion interval has zero spherical area and is rejected. If samples are
aligned time bins, do not simply delete a repeated sample: model that bin
upstream while preserving the alignment expected by downstream analysis.

## Count caps without materializing membership

`Coverage.counts` and `count_caps_per_cell()` point in opposite directions:

| Expression | Meaning |
| --- | --- |
| `coverage.counts[i]` | Number of cells in input region `i` |
| `count_caps_per_cell(...)[j]` | Number of input caps containing cell `j`'s center |

For a moderate grid, dense cap counting is usually the most efficient form:

```python
cap_centers = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
radii_rad = np.deg2rad([5.0, 8.0])
counts = px.count_caps_per_cell(cap_centers, radii_rad, resolution=8)
```

At high resolution, pass a small positional `cells=` query to avoid allocating
the complete grid. Query work grows with both cap count and requested-cell
count, so it is a sparse-query path rather than a generally faster dense path.

## Import segmented coverage

Use the copying validator when another tool already produced segmented RING
IDs:

```python
coverage = px.Coverage.from_arrays(
    cells=[2, 7, 9],
    offsets=[0, 2, 3],
    resolution=1,
)

assert len(coverage) == 2
assert not coverage.cells.flags.writeable
```

Cells must be unique within a segment. Their supplied order is retained.

## Summarize aligned occupancy

One `Coverage` is one independent source; its segments are aligned occupancy
bins. Runs are counted independently by source, while gaps are measured after
merging all sources:

```text
bin                 0  1  2  3  4
source A, cell 7    X  X  .  .  .    one source run
source B, cell 7    .  .  X  .  X    two source runs
merged              X  X  X  .  X    one merged gap of one bin
```

```python
source_a = px.Coverage.from_arrays([7, 7], [0, 1, 2, 2, 2, 2], resolution=1)
source_b = px.Coverage.from_arrays([7, 7], [0, 0, 0, 1, 1, 2], resolution=1)
summary = px.summarize_occupancy([source_a, source_b])

assert summary.run_counts[0] == 3
assert summary.merged_gap_counts[0] == 1
assert summary.merged_gap_steps_sum[0] == 1
```

Splitting one physical source into several `Coverage` objects changes run
counts. The function returns aggregate counts and gap sums, not individual
windows, occupied durations, timestamps, or physical time units. This API
remains provisional during `0.x` while independent occupancy workloads are
evaluated. Bins must be aligned and ordered across sources, but they need not
have equal duration for the ordinal reduction. Equal duration is required only
if the returned gap-step counts are converted to physical time by multiplying
by one bin duration.

For resolution choice, memory limits, candidate filtering, and threading, see
[Performance and memory](performance.md).
