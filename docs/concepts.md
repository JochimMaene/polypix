# User guide

## Resolution and cell IDs

Polypix uses fixed-resolution HEALPix RING ordering, and calls the HEALPix order
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

Every step up quadruples the grid and roughly halves the cell scale, so the
numbers get away from you quickly. Resolution 12 already needs about 1.5 GiB
for one `int64` per cell. Resolutions run to 29, which is only useful for sparse
transforms and selected-cell queries, never for a complete dense map.
[Resolutions](resolutions.md) lists every one with its angular size, its size on
the ground, and what a dense map would cost.

What you get back are ordinary RING pixel indices in `[0, cell_count)`. They are
not packed tokens and they do not encode their resolution. The result object
carries that. If you need ordering conversion, neighbors, hierarchy, or map
algebra, hand these IDs to a fuller HEALPix library; Polypix does not do any of
it.

## Direction geometry

Everything in and out of Polypix is a Cartesian direction `(x, y, z)` on the
unit sphere. Magnitude is ignored, so position vectors work as well as unit
vectors. The frame is yours: body-fixed for a planet, celestial for a sky
survey. Polypix neither labels it nor transforms between frames. Just make sure
every vector in one call lives in the same one.

Working in three dimensions is why the poles and the longitude seam need no
special handling.

Use `cell_at()` when your input is points rather than regions, and `centers()`
to go back the other way. Be careful about what "back" means: `centers()` gives
you the grid representative, not the direction you started with. Cell centers
round-trip exactly; arbitrary directions do not. A direction sitting numerically
on a cell edge is a floating-point tie. It is repeatable for one build and
platform, but the API does not promise it across platforms.

Orbit propagation, attitude, sensor projection, ellipsoid intersection: all of
that happens before Polypix sees anything.

## Center-sampled coverage

A cell is covered when its center falls inside a cap or footprint, or on the
boundary. That is one sample per cell, and it is worth being clear about what
that costs you:

- a region too small or too thin to contain a center returns nothing at all;
- a cell straddling an edge is included only if its center is inside.

So this is not a conservative spatial index. If you need every cell a region
touches, Polypix will not give you that today.

Caps and footprints use the same rule. The accepted geometry and its numerical
limits are in the [geometry contract](api.md#geometry-contract).

When you only want to know *how many* caps cover each cell, reach for
`count_caps_per_cell()` rather than `cover_cap()`. It accumulates counts
directly instead of emitting the same cell ID once per covering cap, which for
visibility-density maps is usually the whole game.

## Batches and segments

`cover_footprint()` takes one footprint, a dense batch, or a ragged sequence.
`cover_cap()` takes one center or a batch, with a shared radius or one per cap.
`cover_sweep()` turns consecutive pairs of two sampled edges into independent
quadrilaterals. All three hand back a single `Coverage`:

```text
cells for item i = cells[offsets[i] : offsets[i + 1]]
```

One flat array plus offsets keeps the input boundaries without allocating a
Python object per region. `len(coverage)` is the item count and `coverage[i]` is
a read-only zero-copy view of one segment. If segmented arrays arrive from
somewhere else, `Coverage.from_arrays()` copies and validates them.

Sweep sampling is part of the input contract, because Polypix joins consecutive
samples with the shorter great-circle arc. Steps approaching 180° bow noticeably;
past 180° you get the opposite arc; exactly ambiguous steps are rejected. Polypix
cannot tell a deliberate minor arc from an undersampled trajectory, so sample
densely enough that each arc is the boundary you meant.

## Occupancy summaries

`summarize_occupancy()` reads the segments of one or more `Coverage` results as
aligned, ordered bins. It counts consecutive runs per source, then merges every
source to measure the uncovered bins between occupied windows.

Gaps are counts of bins, not durations. Polypix has no clock. Hits in bins 0
and 2 give a gap of one, so this is an uncovered interval and not a
start-to-start period. Equal bin duration only matters when you multiply those
counts out into real time.

The result is sparse and sorted by cell ID, so a high resolution does not force
a dense global allocation.

## Restricting coverage to known cells

Pass `candidate_cells` when only a sparse set of cells matters to you:

```python
coverage = px.cover_sweep(
    left_edge_xyz,
    right_edge_xyz,
    resolution=12,
    candidate_cells=aoi_cells,
)
```

Candidates are a set: order and duplicates are discarded. Filtering is still
center sampled, so it does not become a conservative index, and a dense
candidate set can end up slower than just scanning the rings.

See [Performance and memory](performance.md) for candidate planning, geometry
shape, chunking, output sizing, and threads.

```{toctree}
:hidden:
:maxdepth: 1

resolutions
performance
interoperability
```
