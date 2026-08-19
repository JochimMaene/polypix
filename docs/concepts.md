# How it works

## Resolution and cell IDs

HEALPix is a Hierarchical Equal Area isoLatitude Pixelation of a sphere, and all
three parts of that name show up in how you use it. Cells subdivide into four,
every cell covers exactly the same solid angle, and cell centers sit on rings of
constant latitude.

Equal area is what lets you count hits per cell and compare them without
weighting by cell size. The rings are why Polypix returns RING ordering: a
region's coverage falls into a contiguous span on each ring it touches, which is
also why coverage cost tracks a region's latitude extent more than its shape.

Polypix uses fixed-resolution RING ordering throughout, and calls the HEALPix
order `resolution`:

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
[Resolutions](resolutions.md) shows what the grid looks like at each level, and
lists every one with its angular size, its size on the ground, and what a dense
map would cost.

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
special handling. If another library hands you component-major `(3, N)`, move
the axis yourself:

```python
vectors_n3 = np.moveaxis(vectors_3n, 0, -1)
cells = px.cell_at(vectors_n3, resolution=8)
```

Use `cell_at()` when your input is points rather than regions, and `cell_centers()`
to go back the other way. Be careful about what "back" means: `cell_centers()` gives
you the grid representative, not the direction you started with. Cell centers
round-trip exactly; arbitrary directions do not. A direction sitting numerically
on a cell edge is a floating-point tie. It is repeatable for one build and
platform, but the API does not promise it across platforms.

Polypix attaches no frame, datum, unit, or epoch metadata. Coordinate
transforms, orbit propagation, attitude, and sensor projection happen upstream;
pass the resulting directions or regions to Polypix.

## Center-sampled coverage

A cell is covered when its center falls inside a cap or footprint, or on the
boundary. That is one sample per cell, and it is worth being clear about what
that costs you:

- a region too small or too thin to contain a center returns nothing at all;
- a cell straddling an edge is included only if its center is inside.

So this is not a conservative spatial index. If you need every cell a region
touches, Polypix will not give you that today.

```{figure} assets/generated/center-sampling.svg
:alt: A circular region on a HEALPix grid. Cells whose centers fall inside are filled; one cell the region overlaps is left out because its center is outside.
:width: 100%
:align: center

Blue cells are what `cover_cap()` returned. Grey dots are cell centers. The orange cell is overlapped by the region but left out, because the rule asks about the center and nothing else.
```

Caps and footprints use the same rule. The accepted geometry and its numerical
limits are in the [geometry contract](api.md#geometry-contract).

When you only want to know *how many* caps cover each cell, reach for
`count_caps_per_cell()` rather than `cover_cap()`. It accumulates counts
directly instead of emitting one cell ID per cap–cell pair.

## Batches and segments

`cover_convex_polygon()` takes one footprint, a dense batch, or a ragged sequence.
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

## Occupancy runs

`occupancy_runs()` reads the segments of one or more `Coverage` results as
aligned, ordered bins. It returns every maximal half-open `[start, stop)` run,
grouped by cell, where at least `minimum_sources` source entries cover the cell.
Matching indices must describe identical bin boundaries, and consecutive bins
must really be adjacent; those are caller assertions because `Coverage` carries
no clock. Split an analysis at time discontinuities or insert an empty separator
bin so a run cannot bridge them.

Runs use segment indices, not durations. Polypix has no clock. Map `starts` and
`stops` through your own array of time edges, then choose whether revisit means
end-to-start, start-to-start, a finite-horizon edge gap, or a cyclic gap.

The result is sparse and sorted by cell ID, so high resolution does not force a
dense global allocation. Sequence positions count independently, so callers
also own source uniqueness. Multi-source thresholding intentionally drops
source identity; extract runs once per source when observer attribution is
required. A sampled sweep yields occupied-bin runs, not exact continuous access
events; physical boundary precision is limited by the sampling cadence.

## Handing cells to other libraries

Polypix exchanges exactly two things with the rest of the ecosystem: `(N, 3)`
direction arrays and fixed-resolution RING IDs. There is no frame object model
to adopt and no Astropy or geospatial runtime to install.

Signed `int64` RING IDs go straight to healpy, astropy-healpix, or cdshealpix
for everything Polypix leaves out: ordering conversion, neighbors,
interpolation, resampling, harmonics, and file formats.

Two things to watch when you hand data over:

- `cell_corners()` returns four corner vectors, and HEALPix cell edges are curved.
  Those four points are not a sampled boundary, so do not round-trip them as an
  exact great-circle polygon.
- A MOC represents whole cells by area. Converting center-selected cells into a
  MOC changes what the result means. It does not retroactively turn your query
  into an intersection query.

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
