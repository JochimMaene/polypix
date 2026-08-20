# Project goal

## Goal

Polypix will become the default permissively licensed Python engine for mapping
large batches of already-resolved spherical regions to deterministic,
center-sampled HEALPix RING membership. It is a focused region-to-grid kernel,
with only those fused reductions whose HEALPix-aware implementation avoids a
prohibitively large intermediate. Its value is correctness, batch throughput,
predictable semantics, easy installation, and a small NumPy-first API.

This document defines the direction of the project while it remains in `0.x`.
It is narrower than a roadmap. Some target API decisions below
are breaking changes from the current release and will be introduced only with
the necessary correctness, packaging, and performance evidence.

## Primary user and job

Polypix is for engineers and researchers running high-volume coverage
simulations or center-sampled spherical rasterization pipelines. They already
have angular regions in a caller-defined Cartesian frame and need to associate
many regions with cells quickly or reduce the resulting ordered occupancy
segments.

Satellite and sensor coverage is the leading use case and should lead the
documentation and benchmarks. The geometry and API remain generic enough for
beams, access regions, aerial systems, astronomy, and other spherical domains.

Polypix is useful beyond satellites wherever the same computational shape
appears: many circular or convex regions, many swept intervals, a standard
spherical grid, and enough output that Python loops or materialized repeated
cell IDs become expensive.

## Pipeline boundary

The durable boundary is:

```text
physical model or source geometry
    -> resolved directions, caps, convex-polygon vertices, or paired sweep edges
Polypix
    -> validated HEALPix center membership
    -> optional native per-cell accumulation or ordinal occupancy runs
maps, joins, scheduling, statistics, storage, and visualization
```

Upstream code answers: "what angular region is valid in this frame under this
model?" It owns propagation, interpolation, attitude, field of view, body
shape, ray intersection, terrain, refraction, clocks, and coordinate-frame
transforms.

Polypix answers: "which fixed-resolution HEALPix cell centers are in these
already-resolved regions?" It owns region validation, deterministic RING
traversal, segmentation, and a small number of measured fused reductions.

Downstream code owns physical time, arbitrary statistics, map algebra,
hierarchical coverage, persistence, plotting, and domain decisions.

### Access constraints

Minimum elevation, maximum off-nadir angle, horizon or occultation rules, and
similar access constraints are upstream modeling concepts, not arguments to a
Polypix coverage function. Their meaning depends on some combination of body
radius or ellipsoid, observer or platform position, attitude and boresight,
sensor shape, limb clipping, terrain, atmospheric refraction, target height,
and frame conventions.

Under explicit simplifying assumptions, a constraint may reduce to an exact
cap. In other cases it produces a convex footprint or paired-edge sweep. Polypix
accepts that derived geometry, but it will not add parameters such as
`minimum_elevation`, `maximum_off_nadir`, `body`, `orbit`, `attitude`, or
`sensor` to `cover_*()`. Documentation may provide tested recipes, and separate
optional adapters may be considered when they remove recurring integration
work, but their assumptions must remain visible and they must not become core
runtime dependencies.

## Workloads that define the product

The following uses exercise the same small set of primitives and should guide
examples, benchmarks, and future API decisions:

| Workload | Input to Polypix | Useful result |
| --- | --- | --- |
| Satellite visibility, service redundancy, and spot beams | Exact caps or already-projected beam contours | Segmented coverage or per-cell counts |
| Earth-observation, aerial, maritime, radar, or telescope sweeps | Paired sampled sweep edges | Per-interval coverage and ordinal occupancy runs |
| Astronomy survey tiling and repeated sky exposure | Exact caps or convex focal-plane footprints | Visit-count maps or segmented coverage |
| Planetary imaging and mapping | Body-fixed directions for the Moon, Mars, or another sphere | Coverage and ordinal revisit windows |
| Frame, scene, or acquisition rasterization | Large dense or ragged footprint batches | Center-selected input-to-cell membership for inversion by a database or NumPy |
| Preselected grid centers and static cell masks | Regions plus selected RING cells | Sparse candidate membership or positional cap counts at those cell centers |
| Monte Carlo regions, uncertainty cones, and repeated event ensembles | Large batches of caps or convex regions | Per-cell frequency or explicit membership |

These are application examples, not new domain objects. Weighted dwell,
probability, capacity, continuous access windows, scheduling decisions, and
physical time remain downstream. Arbitrary concave geography, exact
cell-intersection rasterization, and ellipsoid or terrain visibility are poor
fits for the committed kernel.

The implemented examples are not sufficient evidence that the API is complete.
The pre-1.0 discovery gate and cross-domain workload matrix are recorded in the
[API-surface decision](https://github.com/JochimMaene/polypix/blob/main/decisions/api-surface-beyond-constellations.md).

## Product contract

### Geometry

Polypix accepts finite, nonzero Cartesian direction vectors `(x, y, z)` in a
caller-defined frame and normalizes them internally. Magnitude and origin are
not part of the geometry. A frame may be body-fixed for planetary mapping or a
celestial frame for astronomy, but every vector in one operation must already
be expressed in the same frame. Polypix does not label or transform frames.
Datum, ellipsoid, geodetic, and coordinate-reference-system interpretation
belongs upstream or downstream.

A convex polygon (often a projected sensor footprint) is a convex spherical
polygon:

- it is contained within an open hemisphere, so a hemisphere or larger region
  cannot be represented;
- each edge follows the shorter great-circle arc;
- the covered region is closed, so a cell center on an edge is included;
- either vertex orientation is accepted;
- one repeated closing vertex is accepted and removed;
- redundant vertices on one great-circle edge are accepted within
  floating-point precision;
- degenerate, antipodal, self-intersecting, and non-convex geometry is rejected.

Validation is subject to a documented numerical scale floor. Polygons below
roughly `1e-8` radians in angular extent are unsupported and may be rejected as
degenerate; the exact threshold depends on vertex layout and conditioning.

Validation is mandatory. There is no unsafe or validation-free mode.
Validation rejects detectable ambiguity such as antipodal edges and
exact-hemisphere boundaries. It cannot infer an unexpressed intention to use
the longer arc or the other side of an otherwise valid minor-arc boundary.

`cover_sweep()` accepts two sample-aligned boundary curves. Each consecutive
sample pair normally forms one convex quadrilateral; a one-sided repeated
sample forms a pinched triangular segment. Repeating both samples is a
zero-area error. For `N` paired samples, the result contains `N - 1` segments.
Polypix does not implicitly merge or deduplicate the complete sweep.

The paired curves must be sampled densely enough that each shorter great-circle
arc between consecutive samples represents the intended physical boundary.
Polypix rejects exactly ambiguous geometry but cannot detect an undersampled
trajectory whose minor arcs are mathematically valid.

`cover_cap()` accepts one center vector or a flat batch and either one shared
angular radius or one radius per center. Caps are exact spherical regions, not
polygon approximations. Finite radii from zero through pi are accepted; their
unambiguous point, hemisphere, larger-than-hemisphere, and full-sphere meanings
do not inherit the polygon open-hemisphere restriction.

"Exact" is relative to the supplied spherical geometry: cap boundaries are
small circles, footprint edges are minor great-circle arcs, and sweep geometry
is piecewise spherical between samples. Polypix does not claim that sampled
inputs exactly reproduce a physical sensor projection, terrain boundary, or
ellipsoid intersection.

### Coverage rule

A cell is covered when its center lies inside the cap or footprint, or on its
boundary.
Center sampling is the only coverage rule. Intersection, full-containment,
fractional-area, and approximate bounding-box modes are not part of the
product.

Center sampling is not a conservative spatial index. A small region can contain
no cell center, and a region may intersect a cell whose center lies outside.
Likewise, mapping an arbitrary site to a cell and testing that cell center does
not test the original site direction. Until a separate guaranteed-superset
candidate-cover API is admitted, Polypix must not promise no-false-negative
region indexing.

Coverage may be restricted to a sparse set of `candidate_cells`. Candidate
inputs have set semantics and use the requested resolution and RING ordering.
The implementation may choose the fastest equivalent algorithm internally.
Polygon center inclusion uses one documented nominal floating-point predicate
tolerance. Cap inclusion uses a documented angular tolerance and stable chord
distance. Numerical uncertainty also depends on edge conditioning and the
equivalent center-evaluation path; strategy differences are confined to
centers numerically indistinguishable from a boundary.

### Grid and cell IDs

HEALPix is the first and only committed grid. Polypix uses fixed-resolution
RING ordering and calls the HEALPix order a `resolution`:

```text
nside = 2 ** resolution
```

The name is deliberate even though adjacent HEALPix packages often use
`nside`, `order`, `level`, or `depth`. Polypix supports the power-of-two subset
and uses one approachable input representation throughout the API. It does not
accept `nside` as an alias. A derived `nside` property or helper remains a small
interoperability option if recurring integrations justify it.

Results contain standard HEALPix RING pixel indices. Polypix does not define a
custom packed cell token, support NESTED indices, or mix resolutions in one
result. RING earns its place because the center-only coverage kernel emits
contiguous spans on HEALPix iso-latitude rings directly; converting every
result to NESTED measurably penalizes the primary high-output workloads.

HEALPix-first does not permanently rule out another discrete global grid. A
second grid must first demonstrate a substantial user need or a material
workload advantage. It would use an independently optimized implementation and
ecosystem-native IDs. Polypix will not build a speculative grid protocol before
that evidence exists.

### Result model

Every coverage operation returns one `Coverage`, including a single footprint.
Its canonical representation is:

- `cells`: one eager, flat signed `int64` array of non-negative standard RING
  indices;
- `offsets`: signed `int64` segment boundaries for the input caps, polygons, or
  sweep intervals;
- `resolution`: stored once for the result;
- `segment_sizes`: derived from `offsets`.

Segments preserve input region or interval order and contain no duplicate cells.
Within a segment, native traversal order is deterministic but is not
promised to be ascending. Polypix never sorts solely for presentation.

Explicit cells remain the canonical geometry result. A caller who wants an
accumulation rather than membership names the result with a reducer, and every
producing operation accepts one through `into=`:

- `Count(cells=None)` counts the segments containing each cell;
- `Sum(values, cells=None)` adds one finite `float64` value per segment into the
  cells that segment covers;
- `Stats()`, accepted only by `occupancy()`, accumulates per-cell occupancy
  statistics without building the runs.

Omitting `into=` keeps the full result: a `Coverage`, or every occupancy run.
`Coverage.reduce()` applies the same `Count` and `Sum` vocabulary to coverage
that is already materialized, so covering once and reducing several times is one
concept. `Count` and `Sum` return either a dense grid or positional values for
explicitly requested cells, and the selected-cell path does not allocate the
full grid. This supplies the useful symmetry across caps, polygons, and sweeps
without multiplying the public surface into geometry-specific count and sum
verbs.

A reducer names a result, not an algorithm. Polypix fuses the accumulation into
the geometry kernel where measurement shows that wins and materializes
membership otherwise, so the two are interchangeable at the call site. Only the
cap kernel currently fuses, and only where it estimates that fusing is cheaper
than covering once and reducing.

Two bounded results address cases where merely retaining a summary would
otherwise discard essential information:

- `occupancy(sources, *, minimum_sources=1)` returns every maximal half-open
  ordinal run per cell after thresholding aligned `Coverage` sources;
- `occupancy(..., into=Stats())` returns per-cell run counts, complete internal
  gap sums and maxima, and the observed window bounds, accumulated in one pass
  without building the runs.

`OccupancyRuns` is lossless with respect to the thresholded ordinal state. It
does not choose a cadence, convert to timestamps, silently wrap a period, omit
boundary gaps, or hard-code mean/max/median revisit statistics. Those choices
remain downstream. It retains the applied threshold and input source-entry
count as provenance, as does `OccupancyStats`. The older mixed
`OccupancySummary` result and `summarize_occupancy()` were removed because they
combined source-local run counts with source-unioned mean-gap ingredients.

`Coverage` is the public interchange seam between region generation, Polypix,
and downstream tools. Native operations wrap their newly owned buffers without
copying. Imported arrays use
`Coverage.from_arrays(cells, offsets, resolution)`, which copies and validates
offsets, cell ranges, and within-segment uniqueness. Result arrays are
read-only. `OccupancyRuns` follows the same ownership policy and is constructed
only by Polypix. Public cell IDs, offsets, and ordinal run indices are signed
`int64` for frictionless NumPy and neighboring-library use; the native kernel
may retain unsigned counters internally.

### Current candidate public API

The implemented canonical surface is a candidate rather than the complete 1.0
target:

```python
Count(cells=None)
Sum(values, cells=None)
Stats()

cover_convex_polygon(
    polygons_xyz,
    resolution,
    *,
    vertex_offsets=None,
    candidate_cells=None,
    threads=None,
    into=None,
) -> Coverage | ndarray

cover_cap(
    centers_xyz,
    radii_rad,
    resolution,
    *,
    candidate_cells=None,
    threads=None,
    into=None,
) -> Coverage | ndarray

cover_sweep(
    left_edge_xyz,
    right_edge_xyz,
    resolution,
    *,
    candidate_cells=None,
    threads=None,
    into=None,
) -> Coverage | ndarray

Coverage.reduce(reducer) -> ndarray[int64] | ndarray[float64]

occupancy(
    sources,
    *,
    minimum_sources=1,
    into=None,
) -> OccupancyRuns | OccupancyStats

cell_at(vectors_xyz, resolution)
cell_centers(cells, resolution)
cell_corners(cells, resolution)
cell_count(resolution)
```

The paired-edge operation is named `cover_sweep()` to avoid conflict with the
established HEALPix meaning of a latitude strip. `cover_convex_polygon()` names
the actual accepted geometry instead of overloading "footprint," which may also
mean a point, cap, line, or non-convex region in upstream libraries. The former
`cover_footprint()`, `centers()`, and `corners()` spellings were removed while
the project is pre-1.0 instead of creating a permanent parallel vocabulary. The
same window removed `count_caps_per_cell()`, `count_coverage_per_cell()`,
`sum_coverage_per_cell()`, `occupancy_runs()`, `occupancy_stats()`, and
`summarize_occupancy()` in favour of the `into=` reducer form.

`cover_convex_polygon()` accepts one `(vertices, 3)` array, a dense
`(polygons, vertices, 3)` batch, a sequence of arrays for a ragged batch, or a
packed vertex array with `vertex_offsets=`. Ordinary numeric array-like inputs
are accepted and converted once when needed. Compatible contiguous arrays use a
zero-copy fast path; the API exposes no memory-layout or copying controls.

`cover_cap()` accepts `(3,)` or `(caps, 3)` centers with scalar or pairwise
radii.

`Count` and `Sum` are accepted as `into=` by every covering call and by
`Coverage.reduce()` for coverage that already exists. `values` is a scalar or one
finite `float64` value per segment. `cells=None` returns a dense array indexed by
RING ID; `cells=` preserves query order and duplicates without allocating a
resolution-sized grid, unlike the set semantics of coverage candidates. Floating
sums are accumulated deterministically in segment and hit order.

`occupancy()` accepts one `Coverage` or one per source entry with the same
resolution and number of aligned, ordered bins. The caller owns source
uniqueness, identical bin boundaries, and temporal adjacency. For each cell it
returns every maximal half-open `[start, stop)` interval where at least
`minimum_sources` entries cover that cell. Runs use ordinal segment indices.
Callers map segment boundaries to constant or variable time edges and choose
complete, leading, trailing, or cyclic revisit-gap semantics themselves.

`cell_at()` maps finite nonzero direction vectors to their fixed-resolution
RING cells and completes the inverse bridge to `cell_centers()`. Cell centers
must round-trip, but the API does not promise which adjacent cell owns a direction
numerically on an exact mathematical edge or vertex across platforms.
`cell_corners()` returns only the four HEALPix corner vectors for each cell;
HEALPix cell edges are curved, so those corners must not be round-tripped as an exact
great-circle polygon. Polypix does not otherwise grow into a general
cell-manipulation library.

There are no public grid objects, polygon classes, configuration objects,
backend selectors, or algorithm controls.

### API evolution rules

- Public verbs describe both their geometry and result. Polypix does not grow a
  generic `cover(kind=..., output=...)` dispatcher whose return type changes
  with flags.
- Batching and segmentation are first-class. One call handles one or many
  regions, and `Coverage.offsets` preserves the input axis without returning a
  Python list of arrays.
- `candidate_cells` always has set semantics and restricts a coverage
  computation. A query parameter named `cells` is positional and preserves
  order and duplicates. Those boundaries are not interchangeable.
- Standard RING IDs and ordinary NumPy arrays are the interoperability layer.
  Dependency-specific coordinate, orbit, geometry, and map objects do not enter
  the core API.
- A fused verb is admitted only when it has one stable result type and avoids a
  measured intermediate or Python loop that callers cannot otherwise remove.
  Private RING spans, traversal sinks, and parallel strategies stay private.
- Geometry-neutral reducers operate on `Coverage`. Geometry-specific fused
  reducers are retained only where benchmarks show that materializing
  `Coverage` is itself the dominant avoidable cost. This is why fused cap
  counting coexists with generic coverage counting, but there is no automatic
  family of polygon-, sweep-, and cap-specific count/sum names.
- Fused candidates are evaluated as an operation-by-geometry matrix: explicit
  membership, per-input counts, dense/selected/sparse per-cell accumulation,
  and compressed membership. API symmetry alone does not admit every cell in
  that matrix.
- Ordinal occupancy exposes maximal runs rather than a fixed statistic.
  Physical time, observation attribution, gap censoring, cyclic periods,
  cadence conversion, and revisit statistics stay in downstream analysis.
- New convenience features should normally be conversion recipes or optional
  adapters. They enter the package only when they repeatedly prevent user error
  without adding a physical model or runtime dependency.

## Performance contract

Polypix optimizes first for complete public-call throughput over large batches
of exact caps, small convex footprints, swept intervals, and focused fused
operations. Benchmarks include input conversion, validation, native
computation, allocation, and result construction.

Performance claims must be based on public, reproducible comparisons against
the fastest applicable alternatives. Those cross-library comparisons live in a
separate benchmark repository so competitor dependencies and adapters do not
become part of Polypix. That repository is not public yet, so no cross-library
claim is made until a URL and reproducible results are published. This
repository's regression benchmarks cover:

- representative dense and ragged footprint batches;
- sweep intervals;
- exact cap materialization and fused per-cell cap counts;
- generic per-cell coverage counts and weighted sums;
- segmented ordinal occupancy runs;
- multiple useful resolutions and output sizes;
- sparse candidate-cell workloads;
- single-threaded and automatic parallel execution;
- single-footprint latency as a guardrail, not the primary objective.

Before 1.0, deterministic non-constellation fixtures must also cover astronomy
exposure accumulation, ragged scene lookup, uncertainty ensembles, and batch
direction-to-cell indexing. This guards against optimizing the API itself for
the two executable constellation examples.

The external comparison set initially includes equivalent `healpy` and
`cdshealpix` polygon-coverage workflows. Comparisons use the same grid,
resolution, coverage rule, thread budget, and materialized output; cases with
different semantics are labeled rather than presented as direct wins.

Polypix aims to lead the primary batch workloads, not every microbenchmark.
Optimizations for obscure cases do not justify public complexity or regressions
in the primary path.

Large batches are parallelized inside the native kernel when measurements show
a benefit. `threads=None` selects an automatic policy, `threads=1` disables
internal parallelism, and larger values are maximums rather than raw
thread-spawn requests. Calls below the measured crossover remain sequential
without initializing a worker pool. Polypix releases the GIL, exposes no
scheduler or chunk controls, and returns identical membership and ordering
regardless of thread count on the same build and platform.

Correctness, licensing, and reliable installation are constraints rather than
performance trade-offs. Public simplicity normally wins over marginal speed.
Material end-to-end gains may justify contained internal complexity when the
gain and maintenance cost are both measured.

## Ecosystem position and interoperability

Polypix is an accelerator within the open-source spherical-geometry ecosystem,
not a replacement for it:

- [Skyfield](https://rhodesmill.org/skyfield/),
  [Astropy coordinates](https://docs.astropy.org/en/stable/coordinates/),
  Orekit, SPICE, and domain simulators own state, time, frames, propagation,
  attitude, fields of view, and physical access constraints upstream.
- [spherical-geometry](https://spherical-geometry.readthedocs.io/) and other
  geometry packages may construct or decompose regions before Polypix validates
  the supported convex pieces.
- [healpy](https://healpy.readthedocs.io/),
  [astropy-healpix](https://astropy-healpix.readthedocs.io/), and
  [cdshealpix](https://cds-astro.github.io/cds-healpix-python/) own broader
  HEALPix conversion, map, hierarchy, interpolation, and astronomy workflows.
- [MOCPy](https://cds-astro.github.io/mocpy/) and the IVOA MOC ecosystem own
  mixed-resolution region exchange, boolean operations, persistence, and
  space-time MOCs downstream.
- H3, S2, and other discrete global grids solve related jobs with different
  identifiers and geometry. Polypix does not hide those differences behind a
  nominally generic grid interface.

Interoperability is a positive product goal. Its stable seams are ordinary
NumPy direction arrays, documented radians, segmented arrays, and standard
fixed-resolution HEALPix RING IDs. Tested recipes should show how to move data
to and from common neighboring projects. Core functions do not accept their
classes, attach frame metadata, or require them at runtime. An adapter that
proves broadly useful should remain optional and translate at the boundary
rather than pulling another project's object model into Polypix.

Polypix differentiates itself through validated large-batch region coverage,
sweep throughput, deterministic segmented output, and focused fused
reductions. It does not claim to be a smaller general-purpose HEALPix library.

## Architecture and distribution

The architecture consists of:

- a thin Python and NumPy layer for ergonomic inputs and results;
- one owned Rust CPU kernel for geometry, coverage, and the
  current focused fused operations;
- no pure-Python or GPU fallback.

The implementation language and native ABI are private. The current owned
kernel changes only when evidence shows a better result across correctness,
end-to-end throughput, threading, build reliability, wheel size, platform
coverage, provenance, and maintenance cost. Users never select a backend.

NumPy is the only runtime dependency. Published wheels contain the native
kernel and require no system HEALPix installation, compiler, or geometry
package.

Scientific Python SPEC 0 informs Polypix's minimum Python and NumPy versions,
but is not a requirement to drop an older compatible NumPy release. Polypix may
retain inexpensive compatibility when it benefits users, and every declared
minimum remains tested. The supported wheel matrix is:

- Linux x86-64 and ARM64;
- macOS Intel and Apple Silicon;
- Windows x86-64.

Other source builds are best effort. Support for 32-bit systems, PyPy,
musllinux, and Windows ARM requires evidence of meaningful demand.

Polypix is licensed under Apache-2.0. Distributed source and wheels must retain
clean provenance, notices, and license-compatible dependencies; release checks
guard that boundary.

## Correctness

For the same valid input, resolution, candidate set, released version, build,
and platform, Polypix returns the same membership and native order across batch
partitioning, repeated execution, and thread counts. Platform `libm`
differences may affect only centers inside the documented floating-point
boundary tolerance.

The native kernel must be tested against independent oracles with randomized
and adversarial polygons, caps, reductions, and segmented occupancy sources. Tests cover
poles, longitude wraparound, cell boundaries, hemisphere limits, invalid
geometry, empty batches, ragged inputs, candidate sets, sparse high-resolution
state, and parallel execution. Performance work may change strategy, never
results.

## Feature admission

A proposed feature first has to satisfy all of these conditions:

1. It serves a demonstrated workload that fits the pipeline boundary.
2. It consumes resolved spherical geometry, standard cell IDs, or Polypix
   results rather than introducing a physical or coordinate model.
3. Its API remains a bounded concept with one clear semantic result.
4. It does not regress the primary paths and can be maintained with
   proportionate tests, documentation, benchmarks, and packaging work.

It must then pass at least one admission lane:

- **Kernel or fused operation:** it closes a correctness gap, eliminates a
  measured bottleneck, or avoids a material intermediate or Python loop that
  downstream code cannot remove cleanly.
- **Boundary or ergonomics:** it repeatedly prevents real user error or
  interoperability friction, with negligible hot-path and dependency cost.

Speculative abstractions, convenience aliases, and "someone might need this"
are not sufficient. Feature trade-offs are assessed case by case with measured
costs; there is no universal percentage threshold.

## Explicit non-goals

The following are outside the committed product:

- propagation, interpolation, attitude, sensor, clock, calendar, body,
  ellipsoid, terrain, and atmospheric models;
- minimum-elevation, off-nadir, horizon, occultation, or field-of-view
  constraint evaluation before it has been resolved to supported spherical
  geometry;
- native longitude/latitude, WGS84, CRS, GeoJSON, Shapely, Skyfield, Orekit, or
  Astropy object models and runtime dependencies; array-based recipes and
  optional adapters are welcome;
- concave polygons, holes, multipolygons, or geometry repair;
- ambiguous coverage-rule flags; any conservative, containment, intersection,
  or fractional rule requires a separately admitted verb and contract;
- general-purpose geometry or map boolean algebra, arbitrary reducers,
  access-window lists, timestamps, and variable-duration time integration; the
  ordinal `occupancy()` result deliberately leaves those policies downstream;
- NESTED or mixed-resolution results;
- MOCs, map operations, neighbors, hierarchy traversal, interpolation,
  spherical harmonics, FITS, or plotting;
- distributed, streaming, GPU, CuPy, or JAX execution;
- multiple native backends or a generic grid abstraction;
- a pure-Python fallback.

## Evidence-gated experiments

Experiments are not roadmap promises. They exist to answer a measured question
and are discarded when they do not justify their cost.

The first recorded experiment was a range-compressed result representation for
very large workflows. Exact cap coverage now uses RING spans privately, and
fused cap counts consume those spans without expansion. A public compressed
result remains evidence-gated: it enters the API only if workloads need to
retain ranges rather than explicit cells or one of the focused aggregations.

The operation-by-geometry experiments still include per-input counts without
cell materialization, sparse touched-cell results, and cell-to-region incidence.
Dense and selected generic counts, one bounded constant-weight sum, and packed
ragged polygon input have been admitted. A polygon- or sweep-specific fused
accumulator enters the API only when an unrelated workload proves that explicit
`Coverage` materialization is the limiting cost. API symmetry alone is not
evidence.

A conservative guaranteed-superset candidate cover is the most consequential
unresolved coverage-rule experiment. It is required for no-false-negative
spatial indexing, but it is a different numerical product from center
rasterization and may not justify its correctness and maintenance cost. The
project will either admit explicit conservative verbs or keep spatial indexing
outside its claims before 1.0.

The complete candidate matrix and evidence gates live in the
[API-surface decision](https://github.com/JochimMaene/polypix/blob/main/decisions/api-surface-beyond-constellations.md).
Other possible experiments
include a second grid or GPU execution. Each must independently pass the
feature-admission test.

## Stability and success

While Polypix remains in `0.x`, clean breaking changes are preferred over
deprecation aliases and compatibility layers. Each release is still tested,
documented, and usable. There is no deadline for 1.0.

Before 1.0, center-raster versus conservative-indexing claims, the complete
public API, supported wheel matrix, deterministic correctness,
provenance, and benchmark contract must all be settled and proven. After 1.0,
Polypix follows semantic versioning and normal deprecation periods.

The goal is to become the default Python choice for this focused job.
PyPI downloads, downstream dependents, citations, and recurring real-world
users are evidence of adoption, not reasons to broaden scope. Popularity should
follow from being fast, correct, easy to install, easy to use, and small.
