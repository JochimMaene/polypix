# API Surface Beyond the Constellation Examples

Status: **accepted; first consolidation completed**

## Decision

The Starlink and Earth-observation examples are performance evidence, not a
complete product specification. Polypix will not freeze its 1.0 API until the
surface has been evaluated as an operation-by-geometry matrix against several
unrelated spherical-indexing workloads.

The use-case-neutral operation families are:

1. index directions to grid cells;
2. cover regions with segmented cell membership;
3. accumulate region values into cells;
4. restrict or join regions against selected cells;
5. retain or reorganize large membership efficiently;
6. perform domain analysis downstream or in an explicitly separate layer.

Polypix was already strong in region coverage, sparse candidate joins, and batch
direction indexing. This review admitted generic native `Coverage` count/sum
reducers, packed ragged polygons, signed public indices, and lossless ordinal
occupancy runs. It retained fused cap counting on performance evidence rather
than cloning it for every geometry. Remaining candidates still require their
own evidence before 1.0.

This decision does not admit all of the candidates below. It establishes the
evidence that must be gathered before deciding which small subset belongs.

## Current surface assessment

| API | Pre-1.0 assessment |
| --- | --- |
| `cover_cap()` | Foundational spherical primitive; retain. |
| `cover_convex_polygon()` | Foundational convex-region primitive; canonical name states the geometry. The former `cover_footprint()` name is removed. |
| `cover_sweep()` | The paired-edge sweep is broadly useful; this name avoids conflict with the established HEALPix meaning of a latitude strip. |
| `Count` / `Sum` reducers, via `into=` and `Coverage.reduce()` | Geometry-neutral native reductions over `Coverage`; admitted for dense and positional selected-cell outputs. The separate `count_coverage_per_cell()` and `sum_coverage_per_cell()` verbs are removed, and the fused cap accelerator survives as `cover_cap(..., into=Count())` rather than the removed `count_caps_per_cell()`. |
| `occupancy()` | Lossless cell-major ordinal windows by default, or fused per-cell statistics via `into=Stats()`; replaces the mixed, lossy canonical `summarize_occupancy()` result and the separate `occupancy_runs()` and `occupancy_stats()` verbs. |
| `cell_at()` | Foundational direction-to-cell bridge; retain. |
| `cell_centers()` | Foundational cell-to-direction bridge; explicit canonical name. |
| `cell_corners()` | Explicitly describes the four returned cell vertices without implying that curved HEALPix cell edges are sampled. |
| `Coverage` | Validated, read-only segmented interchange type with copied imports and zero-copy native results; retain. |

## Cross-domain workload matrix

The API must be exercised by workload shapes that do not depend on satellite
propagation or the two executable constellation examples:

| Workload | Needed primitives | Current fit or gap |
| --- | --- | --- |
| Astronomy survey tiling and focal-plane exposure | Cap/polygon coverage, point-to-cell indexing, per-cell counts or weighted exposure | Coverage, indexing, generic counts, and constant per-region weighted sums fit. |
| Astronomy catalog cone search | Point-to-cell indexing, conservative region candidates, exact point filtering | Point indexing exists; center-selected coverage is not a no-false-negative candidate cover. |
| Planetary image and scene catalogs | Ragged polygon batches, conservative selected-cell joins, cell-to-scene inversion | Packed ragged center rasterization exists; conservative candidates and inversion need evidence. |
| RF, optical, radar, or acoustic beam maps | Caps or convex contours, additive power/capacity values | Geometry and constant per-region accumulation fit. |
| Aerial, maritime, or telescope scan sweeps | Paired-edge sweeps, per-segment counts, dwell accumulation | Sweep coverage plus generic count/sum reduction fits; a fused count-only path still needs evidence. |
| Monte Carlo footprints and uncertainty ensembles | Large cap/polygon batches, weighted probability or frequency maps | Explicit coverage, fused cap counts, and generic weighted accumulation fit. |
| Static sites, facilities, or catalog directions | Direction-to-cell indexing and exact direction predicates or conservative candidates | Direction indexing exists; positional cap queries still evaluate cell centers, not the original site directions. |
| Global spatial joins and acquisition lookup | Region-to-cell and cell-to-region incidence | `Coverage` stores the first orientation only. |

Deterministic synthetic fixtures for at least astronomy exposure, scene lookup,
and uncertainty accumulation belong in the benchmark suite before the 1.0
surface is declared complete.

## Implemented foundational decision

### Batch direction to cell

A function such as:

```python
cell_at(vectors_xyz, resolution) -> ndarray
```

quantizes directions to cells and exactly round-trips the centers returned by
`cell_centers()`. It lets users reach `cells=` and `candidate_cells=` without a second runtime HEALPix
dependency. Every surveyed HEALPix or discrete-global-grid toolkit provides
the equivalent operation. It was admitted after independent fixtures through
the poles, seams, face transitions, exact transition latitudes,
cell-boundary neighborhoods, scale extremes, and maximum supported resolution,
plus batch benchmarks and exhaustive low-resolution center round trips.

Every finite nonzero direction is assigned to one cell, but floating-point
inputs numerically on a mathematical cell edge or vertex do not carry a public
cross-platform adjacent-cell tie-break guarantee. The result is repeatable for
the same input, build, and platform. A workload that requires a portable
boundary tie policy must define that policy upstream.

### HEALPix size parameter: retain `resolution`

Polypix deliberately keeps `resolution` as the public parameter and defines it
everywhere by:

```text
nside = 2 ** resolution
```

This is more approachable outside the existing HEALPix community, matches the
monotonic concept users choose, and avoids making every call accept two
equivalent grid-size representations. Polypix therefore supports the
power-of-two HEALPix subset rather than every valid non-power-of-two RING
`nside`. Documentation and errors must state the equation prominently instead
of assuming that `resolution`, `order`, `depth`, and `nside` are synonyms.

A derived `nside` property or helper may be considered if real integrations
repeatedly need it, but `nside` will not become an input alias. General RING
`nside` support requires a new evidence-backed decision rather than silently
changing what `resolution` means.

### Center rasterization versus conservative indexing

Center-selected coverage is a sound raster statistic, but it is not a
no-false-negative spatial index. A region smaller than one cell may contain no
cell center, and a region can intersect a boundary cell whose center lies
outside. Mapping an arbitrary site to a cell and testing that cell center also
does not test the original site direction.

Before 1.0, Polypix must either remove unqualified spatial-indexing claims or
admit separate conservative candidate-cover operations with a guaranteed
superset contract and exact downstream filtering. Center inclusion must not be
silently changed, and an approximate-overlap flag is insufficient for a
top-quality index. Cap, polygon, full-containment, intersection, and fractional
coverage semantics require separate names and independent oracles.

### Coverage as an interchange type

`Coverage` now has a validating, copying `from_arrays()` constructor and a
trusted zero-copy native construction path. Arrays are read-only, segments
must contain unique cells but retain their supplied order, `len(coverage)` is
the segment count, and integer indexing returns a read-only segment view.

The review resolved two interoperability decisions: public cells and offsets
are signed `int64`, and `cover_convex_polygon(..., vertex_offsets=...)` accepts
packed ragged input directly. Both avoid unnecessary copies or casts in common
NumPy and upstream-library workflows.

## Performance experiments

The first experiment below produced an admitted bounded API; the remaining
candidates are ranked experiments rather than promises.

### Per-cell accumulation across geometry types

The accepted design reduces already-materialized `Coverage` with one unweighted
count and one constant per-segment weighted sum. It applies equally to caps,
convex polygons, and sweeps. The relevant complexity is:

```text
materialize then reduce: O(region-cell hits) memory
fused accumulation:      O(grid cells or query cells) memory
```

Constant per-region weights cover exposure, probability mass, capacity, dwell,
and ensemble weighting. The public API remains explicit and deterministic;
there is no arbitrary reducer callback. Floating output is `float64`, input and
intermediate non-finite values are rejected, and addition follows deterministic
segment/hit order. Dense and positional-query outputs are admitted; a sparse
touched-cell result remains an experiment because callers do not always know
the touched cells in advance and dense grids are impossible at high resolution.

Fused cap counting remains because direct RING-span accumulation is materially
faster than cap coverage followed by reduction. Polygon- or sweep-specific
fused reducers require the same end-to-end evidence; naming symmetry does not
admit them. See
[Coverage Reductions and Revisit Statistics](coverage-reductions-and-revisit-statistics.md).

### Counts without cell materialization

Many callers need only the number of selected cells per input region for
discrete area, output sizing, or quality control. A native count sink can use
O(regions) memory instead of constructing O(region-cell hits) IDs. Prototype
caps, footprints, and sweep segments before choosing whether the utility
justifies geometry-specific public verbs.

### Range-compressed segmented coverage

Half-open RING ranges can represent large center-selected regions in work
proportional to crossed rings rather than emitted cells. The cap kernel already
uses them privately, and current healpy exposes `[start, end)` query ranges.
Prototype a segmented range result only with consumers that can retain, join,
store, or reduce ranges without immediately expanding them. A second canonical
result type is not justified by compression ratios alone.

### Cell-to-region incidence

Scene and catalog lookup often needs the transpose of `Coverage`: for each
cell, which input regions contain it? A native moderate-grid implementation can
avoid a global O(hits log hits) sort, but the output still contains O(hits)
region identities and introduces another segmented type. Benchmark it before
admission.

### Fixed-resolution cell-set composition

Grouped union and difference could let upstream geometry packages decompose
concave regions and holes into additive and subtractive convex pieces without
making Polypix a general polygon library. NumPy, MOCPy, S2, and other map tools
already compose cells, so Polypix should add this only if grouped batch
composition is a demonstrated bottleneck.

### Shapes and coverage rules to watch

Annuli have clean frame-neutral semantics and can reuse cap ranges. Ellipses
also recur in survey and beam software but have a much larger semantic and
numerical design cost. Both remain evidence-gated.

Center inclusion, full cell containment, cell intersection, and fractional
coverage are distinct numerical products. Other ecosystems expose several of
them, so the demand is real, but they must never become an ambiguous `mode=`
flag on the current center-sampled functions. Any future admission needs a
separate verb, independent oracle, and performance contract.

## Private kernel direction

The native traversal should be organized around private hit/range sinks so the
same validated geometry can be measured with:

- explicit cell materialization;
- per-input counts;
- per-cell counts or bounded sums;
- range retention;
- potentially an inverted incidence fill.

This is private architecture, not a public callback, iterator, backend, or
reducer protocol. It lets Polypix evaluate a candidate without first committing
to its API and prevents another example from dictating the kernel structure.

## Features that remain outside

The audit does not justify absorbing the rest of a general HEALPix or geometry
stack. The following remain outside unless a later decision independently
changes the product boundary:

- propagation, attitude, sensors, access constraints, bodies, terrain, time,
  and coordinate transforms;
- arbitrary spherical polygon construction, repair, and boolean geometry;
- MOCs, mixed resolutions, hierarchy traversal, neighbors, paths, map
  resampling, interpolation, spherical harmonics, FITS, and plotting;
- longitude/latitude boxes, zones, or other frame-privileging geometry in the
  core Cartesian API;
- generic grids, backends, callbacks, map algebra, and arbitrary reducers.

RING-to-NESTED conversion remains an ecosystem utility rather than a coverage
output mode. It can be reconsidered as a small optional interoperability bridge
only if real downstream integrations otherwise require a large dependency.

## Evidence gate

A candidate enters the 1.0 surface only when it meets all of these conditions:

1. It is foundational across the ecosystem or appears in at least two
   independent workload families.
2. Its native implementation closes a measured correctness, memory, or runtime
   gap that a small NumPy composition cannot close.
3. Its shape, dtype, ordering, empty-input, boundary, memory, and threading
   semantics are specified before implementation is accepted.
4. It has independent correctness oracles and deterministic adversarial tests.
5. It has a public-call benchmark, including conversion and allocation, against
   the best simple composition.
6. It fits the region-to-grid boundary without introducing a domain model.

## Ecosystem evidence

The review used official APIs rather than treating adjacent projects as one
undifferentiated competitor:

- [healpy pixel queries](https://healpy.readthedocs.io/en/latest/healpy_query.html)
  cover discs, convex polygons, and latitude strips with center or approximate
  overlap semantics and can return half-open ranges;
- [astropy-healpix](https://astropy-healpix.readthedocs.io/en/latest/api.html),
  [cdshealpix](https://cds-astro.github.io/cds-healpix-python/api.html), and
  [HPGeom](https://hpgeom.readthedocs.io/en/latest/basic_interface.html) expose
  point/cell conversion and broader HEALPix utilities;
- [Rasterio rasterization](https://rasterio.readthedocs.io/en/latest/api/rasterio.features.html)
  demonstrates value burning and additive merge as general raster operations;
- [S2 coverings](https://s2geometry.io/devguide/s2cell_hierarchy),
  [H3 region functions](https://h3geo.org/docs/api/regions/), and
  [MOCPy](https://cds-astro.github.io/mocpy/stubs/mocpy.MOC.html) demonstrate
  compressed cell sets, composition, and explicitly different containment
  contracts.
