# Project Goal

## Goal

Polypix will become the default permissively licensed Python engine for
converting large batches of valid convex spherical footprints and paired-edge
strips into deterministic, center-sampled HEALPix RING cells through a
minimal NumPy-first API and leading measured end-to-end CPU performance.

This document defines the direction of the project while it remains in `0.x`.
It is deliberately narrower than a roadmap. Some target API decisions below
are breaking changes from the current release and will be introduced only with
the necessary correctness, packaging, and performance evidence.

## Primary User And Job

Polypix is for engineers and researchers running high-volume coverage
simulations or spatial-indexing pipelines. They already have body-centered
footprint geometry and need to associate many footprints with cells quickly.

Satellite and sensor coverage is the leading use case and should lead the
documentation and benchmarks. The geometry and API remain generic enough for
beams, access regions, aerial systems, astronomy, and other spherical domains.

Polypix begins after footprint generation. It does not model orbits,
trajectories, attitude, sensors, fields of view, ellipsoid intersections, time,
or access intervals.

## Product Contract

### Geometry

Polypix accepts finite, nonzero, body-centered `(x, y, z)` vectors and
normalizes them internally. Geometry enters and leaves Polypix only as vectors
on the unit sphere. Datum, ellipsoid, geodetic, and coordinate-reference-system
interpretation belongs upstream or downstream.

A footprint is a convex spherical polygon:

- it is contained within an open hemisphere, so a hemisphere or larger region
  cannot be represented;
- each edge follows the shorter great-circle arc;
- the covered region is closed, so a cell center on an edge is included;
- either vertex orientation is accepted;
- one repeated closing vertex is accepted and removed;
- redundant vertices on one great-circle edge are accepted within
  floating-point precision;
- degenerate, antipodal, self-intersecting, and non-convex geometry is rejected.

Validation is mandatory. There is no unsafe or validation-free mode.
Validation rejects detectable ambiguity such as antipodal edges and
exact-hemisphere boundaries. It cannot infer an unexpressed intention to use
the longer arc or the other side of an otherwise valid minor-arc boundary.

`cover_strip()` accepts two equally sampled boundary curves. Each consecutive
sample pair forms one convex quadrilateral. For `N` paired samples, the result
contains `N - 1` segments. Polypix does not implicitly merge or deduplicate the
complete strip.

The paired curves must be sampled densely enough that each shorter great-circle
arc between consecutive samples represents the intended physical boundary.
Polypix rejects exactly ambiguous geometry but cannot detect an undersampled
trajectory whose minor arcs are mathematically valid.

### Coverage Rule

A cell is covered when its center lies inside the footprint or on its boundary.
Center sampling is the only coverage rule. Intersection, full-containment,
fractional-area, and approximate bounding-box modes are not part of the
product.

Coverage may be restricted to a sparse set of `candidate_cells`. Candidate
inputs have set semantics and use the requested resolution and RING ordering.
The implementation may choose the fastest equivalent algorithm internally.
Center inclusion uses one documented nominal floating-point predicate
tolerance. Numerical uncertainty also depends on edge conditioning and the
equivalent center-evaluation path; strategy differences are confined to
centers numerically indistinguishable from a boundary.

### Grid And Cell IDs

HEALPix is the first and only committed grid. Polypix uses fixed-resolution
RING ordering and calls the HEALPix order a `resolution`:

```text
nside = 2 ** resolution
```

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

### Result Model

Every coverage operation returns one `Coverage`, including a single footprint.
Its canonical representation is:

- `cells`: one eager, flat array of standard RING indices;
- `offsets`: segment boundaries for the input footprints or strip intervals;
- `resolution`: stored once for the result;
- `counts`: derived from `offsets`.

Segments preserve input footprint or interval order and contain no duplicate
cells. Within a segment, native traversal order is deterministic but is not
promised to be ascending. Polypix never sorts solely for presentation.

Explicit cells are the only committed result representation. Users can form a
strip union explicitly with NumPy when needed.

### Intended Public API

The target public surface is intentionally small:

```python
cover_footprint(
    footprints_xyz,
    resolution,
    *,
    candidate_cells=None,
    threads=None,
) -> Coverage

cover_strip(
    left_edge_xyz,
    right_edge_xyz,
    resolution,
    *,
    candidate_cells=None,
    threads=None,
) -> Coverage

centers(cells, resolution)
boundaries(cells, resolution)
```

`cover_footprint()` accepts one `(vertices, 3)` array, a dense
`(footprints, vertices, 3)` batch, or a sequence of arrays for a ragged batch.
Ordinary numeric array-like inputs are accepted and converted once when needed.
Compatible contiguous arrays use a zero-copy fast path; the API exposes no
memory-layout or copying controls.

`centers()` returns normalized center vectors. `boundaries()` returns the four
HEALPix corner vectors for each cell. These are the complete supporting
HEALPix utilities: Polypix does not grow into a general cell-manipulation
library.

There are no public grid objects, polygon classes, configuration objects,
backend selectors, or algorithm controls.

## Performance Contract

Polypix optimizes first for complete public-call throughput over large batches
of small convex footprints and strip intervals. Benchmarks include input
conversion, validation, native computation, allocation, and result
construction.

Performance claims must be based on public, reproducible comparisons against
the fastest applicable alternatives. Those cross-library comparisons live in a
separate benchmark repository so competitor dependencies and adapters do not
become part of Polypix. This repository's regression benchmarks cover:

- representative dense and ragged footprint batches;
- strip intervals;
- multiple useful resolutions and output sizes;
- sparse candidate-cell workloads;
- single-threaded and automatic parallel execution;
- single-footprint latency as a guardrail, not the primary objective.

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

## Architecture And Distribution

The architecture consists of:

- a thin Python and NumPy layer for ergonomic inputs and results;
- one permissively licensed native CPU kernel for geometry and coverage;
- no pure-Python or GPU fallback.

The implementation language and native ABI are private. C++, Rust, and
permissive third-party components must compete on correctness, end-to-end
throughput, threading, build reliability, wheel size, platform coverage, and
maintenance cost. Users never select a backend.

NumPy is the only runtime dependency. Published wheels contain the native
kernel and require no system HEALPix installation, compiler, or geometry
package.

Scientific Python SPEC 0 informs Polypix's minimum Python and NumPy versions,
but is not a requirement to drop an older compatible NumPy release. Polypix may
retain inexpensive compatibility when it benefits users, and every declared
minimum remains tested. The initial supported wheel matrix is:

- Linux x86-64 and ARM64;
- macOS Intel and Apple Silicon;
- Windows x86-64.

Other source builds are best effort. Support for 32-bit systems, PyPy,
musllinux, and Windows ARM requires evidence of meaningful demand.

The target project license is Apache-2.0. Relicensing occurs only after all
distributed GPL-derived dependencies have been removed and contributor rights
have been verified. Existing GPL releases remain under their published terms.

## Correctness

For the same valid input, resolution, candidate set, released version, build,
and platform, Polypix returns the same membership and native order across batch
partitioning, repeated execution, and thread counts. Platform `libm`
differences may affect only centers inside the documented floating-point
boundary tolerance.

The native kernel must be tested against an independent oracle with randomized
and adversarial footprints. Tests cover poles, longitude wraparound, cell
boundaries, hemisphere limits, invalid geometry, empty batches, ragged inputs,
candidate sets, and parallel execution. Performance work may change strategy,
never results.

## Feature Admission

A proposed feature earns a place only when all of the following are true:

1. It serves a demonstrated primary-user workload.
2. Users cannot implement it cleanly with a small amount of upstream or
   downstream NumPy code.
3. It preserves or measurably improves the primary performance path.
4. Its API does not introduce a new conceptual subsystem.
5. Its correctness and performance can be maintained with proportionate tests
   and benchmarks.

Speculative abstractions, convenience aliases, and "someone might need this"
are not sufficient. Feature trade-offs are assessed case by case with measured
costs; there is no universal percentage threshold.

## Explicit Non-Goals

The following are outside the committed product:

- orbit, attitude, sensor, time, and ellipsoid models;
- longitude/latitude, WGS84, CRS, GeoJSON, Shapely, or Astropy integration;
- concave polygons, holes, multipolygons, or geometry repair;
- coverage rules other than center inclusion;
- implicit unions or aggregation;
- NESTED or mixed-resolution results;
- MOCs, map operations, neighbors, hierarchy traversal, interpolation,
  spherical harmonics, FITS, or plotting;
- distributed, streaming, GPU, CuPy, or JAX execution;
- multiple native backends or a generic grid abstraction;
- a pure-Python fallback.

## Evidence-Gated Experiments

Experiments are not roadmap promises. They exist to answer a measured question
and are discarded when they do not justify their cost.

The first recorded experiment is a range-compressed result representation for
very large workflows. It should be prototyped only against workloads where
explicit cell materialization dominates CPU time or peak memory. It enters the
public API only if the end-to-end improvement outweighs the additional result
model.

Other possible experiments include a second grid, GPU execution, or another
coverage rule. Each must independently pass the feature-admission test.

## Stability And Success

While Polypix remains in `0.x`, clean breaking changes are preferred over
deprecation aliases and compatibility layers. Each release is still tested,
documented, and usable. There is no deadline for 1.0.

Before 1.0, the target license, standard cell IDs, public API, supported wheel
matrix, deterministic correctness, and benchmark contract must all be proven.
After 1.0, Polypix follows semantic versioning and normal deprecation periods.

The north star is to become the default Python choice for this focused job.
PyPI downloads, downstream dependents, citations, and recurring real-world
users are evidence of adoption, not reasons to broaden scope. Popularity should
follow from being fast, correct, easy to install, easy to use, and deliberately
small.
