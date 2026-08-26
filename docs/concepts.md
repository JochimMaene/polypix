# How it works

## Resolution and cell IDs

HEALPix stands for Hierarchical Equal Area isoLatitude Pixelation of a sphere,
and all three parts of that name turn up in how you use it. Cells subdivide into
four, every cell covers exactly the same solid angle, and cell centers sit on
rings of constant latitude.

The equal-area part is what lets you count hits per cell and compare the counts
without weighting them by cell size. The rings are the reason Polypix returns
RING ordering: a region's coverage lands in one contiguous span on each ring it
touches, which is also why the cost of a covering call follows a region's
latitude extent more closely than its shape.

Polypix works in fixed-resolution RING ordering throughout and calls the HEALPix
order `resolution`. Every step up quadruples the grid and roughly halves the cell
scale, so the numbers grow faster than most people expect: resolution 12 already
needs about 1.5 GiB to hold one `int64` per cell. Resolutions run to 29, which is
useful for sparse transforms and selected-cell queries but never for a complete
dense map. [Resolutions](resolutions.md) has the formulas, a picture of the grid
at each level, and a table of every resolution with its angular size, its size on
the ground, and what a dense map would cost.

What comes back are ordinary RING pixel indices in `[0, cell_count)`. They are
not packed tokens, and they do not encode their own resolution; the result object
carries that. For ordering conversion, hierarchy, or map algebra, hand the IDs
to a fuller HEALPix library, since Polypix does none of it.

## Direction geometry

Everything going in and coming out is a Cartesian direction `(x, y, z)` on the
unit sphere. Magnitudes are ignored, so position vectors work as well as unit
vectors. The frame is yours to choose: body-fixed for a planet, celestial for a
sky survey. We neither label it nor transform between frames, so the only rule is
that every vector in one call lives in the same one.

Working in three dimensions is also why the poles and the longitude seam need no
special handling here at all. If another library hands you component-major
`(3, N)`, move the axis yourself:

```python
vectors_n3 = np.moveaxis(vectors_3n, 0, -1)
cells = px.cell_at(vectors_n3, resolution=8)
```

Use `cell_at()` when your input is points rather than regions, and
`cell_centers()` to go back the other way. Be careful about what "back" means:
what you get is the cell's representative, not the direction you started with.
Cell centers round-trip exactly and arbitrary directions do not. A direction
sitting numerically on a cell edge is a floating-point tie; the answer is
repeatable for one build and platform, but we do not promise it across platforms.

No frame, datum, unit, or epoch metadata is attached to any of this. Coordinate
transforms, orbit propagation, attitude, and sensor projection all happen
upstream, and what you pass in are the directions and regions that come out of
them.

## Center-sampled coverage

A cell is covered when its center falls inside a cap or footprint, or exactly on
the boundary. That is one sample per cell, and the cost of sampling once is worth
spelling out:

- a region too small or too thin to contain any center returns nothing at all;
- a cell straddling an edge is included only when its center is inside.

This is therefore not a conservative spatial index. If you need every cell a
region touches, Polypix cannot give you that today.

```{figure} assets/generated/center-sampling.svg
:alt: A circular region on a HEALPix grid. Cells whose centers fall inside are filled; one cell the region overlaps is left out because its center is outside.
:width: 100%
:align: center

Blue cells are what `cover_cap()` returned. Grey dots are cell centers. The orange cell is overlapped by the region but left out, because the rule asks only about the center.
```

Caps and footprints use the same rule. The accepted geometry and its numerical
limits are in the [geometry contract](api.md#geometry-contract).

When all you want to know is how many caps cover each cell, reach for
`cover_cap(..., reduce=px.Count())` instead of the default `Coverage`. It
accumulates the counts directly instead of emitting one cell ID per cap-cell
pair.

## Batches and segments

`cover_polygon()` takes one simple polygon, a dense or ragged batch, a `Polygon`
with holes, or a `MultiPolygon`. Each shape object is one result segment.
`cover_cap()` takes one center or a batch, with a shared radius or one per cap.
`cover_sweep()` turns consecutive pairs of two sampled edges into independent
quadrilaterals. All three hand back a single `Coverage`:

```text
cells for item i = cells[offsets[i] : offsets[i + 1]]
```

One flat array plus offsets keeps the input boundaries without allocating a
Python object per region. `len(coverage)` is the item count, and `coverage[i]` is
a read-only zero-copy view of one segment. When segmented arrays arrive from
somewhere else, `Coverage.from_arrays()` copies and validates them.

That constructor is public because it is also the way back from storage. Coverage
is usually the expensive product of a campaign, so the realistic shape of the
work is to compute it once and query it many times afterwards, and two flat
arrays plus a resolution will store in anything that holds arrays:

```{literalinclude} ../examples/coverage_archive.py
:language: python
:start-after: "--8<-- [start:archive-store]"
:end-before: "--8<-- [end:archive-store]"
:dedent:
```

Reading them back validates once, at the boundary, so nothing downstream has to
re-check a hit:

```{literalinclude} ../examples/coverage_archive.py
:language: python
:start-after: "--8<-- [start:archive-load]"
:end-before: "--8<-- [end:archive-load]"
:dedent:
```

A reloaded coverage is an ordinary coverage. `revisit()` and `reduce()` apply
unchanged and return exactly what they would have returned in the process that
built it.

Sweep sampling is part of the input contract, because consecutive samples are
joined with the shorter great-circle arc. Steps approaching 180° bow noticeably,
steps past 180° give you the opposite arc, and exactly ambiguous steps are
rejected. We cannot tell a deliberate minor arc from an undersampled trajectory,
so sample densely enough that each arc is the boundary you meant.

## Occupancy runs

`revisit()` reads the segments of one or more `Coverage` results as aligned,
ordered bins, and reports the runs of consecutive bins in which at least
`minimum_sources` sources cover a cell. Two things it cannot check for itself,
because a `Coverage` carries no clock: matching indices must describe identical
bin boundaries, and consecutive bins must really be adjacent in time. Split an
analysis at a time discontinuity, or insert an empty separator bin, so that a run
cannot bridge one.

Runs are counted in segment indices, not durations, since there is no
clock anywhere in the library. Map `first_start` and `last_stop` through your own
array of time edges, and decide there whether revisit means end-to-start,
start-to-start, a finite-horizon edge gap, or a cyclic gap.

The runs themselves are usually only an intermediate. If what you need per cell
is how many times it was occupied, the total and largest complete gap between
occupations, and the bounds of the window it was seen in, `revisit()` computes
exactly that in one pass and never builds the runs at all. Every field describes
the same thresholded, source-unioned axis, and the leading, trailing, and cyclic
policies stay with you, because the result reports `first_start` and `last_stop`
rather than choosing for you.

The result is sparse and sorted by cell ID, so a high resolution does not force a
dense global allocation. Sequence positions are counted independently, which
means source uniqueness is yours to guarantee, and thresholding several sources
deliberately drops source identity. Extract runs once per source when you need to
know which observer saw what. A sampled sweep gives occupied bins rather than
exact access events, and their boundaries are only as precise as the sampling
cadence.

## Handing cells to other libraries

Polypix exchanges two things with the rest of the ecosystem: `(N, 3)` direction
arrays and fixed-resolution RING IDs. There is no frame object model to adopt,
and no Astropy or geospatial runtime to install alongside it.

Signed `int64` RING IDs go straight to healpy, astropy-healpix, or cdshealpix for
everything Polypix leaves out, which includes ordering conversion,
interpolation, resampling, harmonics, and file formats.

Two things are worth watching when you hand data over:

- `cell_corners()` returns four corner vectors, but HEALPix cell edges are
  curved. Those four points are not a sampled boundary, so do not round-trip
  them as an exact great-circle polygon.
- A MOC represents whole cells by area, so turning center-selected cells into a
  MOC quietly changes what the result means. It does not retroactively make your
  query an intersection query.

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

Without a reducer, the candidates are a set: order and duplicates are discarded,
and the restriction always applies, because it is what defines the result.
Filtering is still center sampled, so it does not become a conservative index,
and a dense candidate set can end up slower than simply scanning the rings.

With a reducer, the same argument fixes the shape of the output instead: one value
per requested cell, in your order, duplicates preserved, and zero where nothing
covered it. Whether to restrict the scan is then our choice rather than yours,
since it cannot change the answer, and we take it only while testing the
selection beats scanning the rings and gathering afterwards. To reduce a stored
`Coverage`, pass the same selection as `coverage.reduce(Count(), cells=...)`.

See [Performance and memory](performance.md) for candidate planning, geometry
shape, chunking, output sizing, and threads.
