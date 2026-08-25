# Project goal

Last updated: 2026-08-22

Polypix has a narrow aim. It takes large batches of already-resolved spherical
regions and maps them to deterministic, center-sampled HEALPix cells. We want it
to be fast at that job, easy to install, and small enough to read.

## Where it fits

```text
physical model or source geometry
    -> resolved directions, caps, polygons, or sweep edges
Polypix
    -> validated RING cell membership and reductions
maps, time handling, statistics, and visualisation
```

The code before Polypix owns orbit propagation, attitude, sensor models,
coordinate frames, body shape, terrain, clocks, and the physical access rules.
The code after it owns time, persistence, map algebra, plotting, and whatever
statistics the domain calls for. Polypix accepts the resolved geometry and does
not try to rebuild the model that produced it.

## What it accepts

The core inputs are:

- finite, non-zero Cartesian direction vectors in a caller-defined frame;
- spherical caps;
- convex spherical polygons whose edges use the shorter great-circle arcs;
- paired sampled edges, with one sweep segment between each pair of samples.

Coverage is based on cell centres: a cell is included when its centre lies
inside the region or on its boundary. That is not a promise about cell
intersection, containment, area coverage, or a conservative index without false
negatives.

The grid is fixed-resolution HEALPix in RING order, from resolution 0 through 29.
Results carry standard RING IDs and preserve the input segments. On top of that
the library supports sparse queries, per-cell counts and sums, and ordinal
revisit summaries, which callers map to their own time edges and gap rules.

The full interface and its validation rules are in the API documentation.

## Why the kernel is ours

The main workloads are large batches of small regions, and sweeps. The first
external approach we tried built an overlap intermediate before filtering on
centres; traversing the RING structure directly removes that work, and the
benchmarks showed the difference clearly enough to settle the question. Keeping
the kernel here also means nobody needs a system HEALPix installation to use the
library.

Python handles inputs and results, Rust handles geometry, traversal, reductions,
and parallel work, and NumPy is the only runtime dependency.

We measure the complete public call, including conversion, validation,
allocation, and result construction. Large-batch throughput is the priority and
single-call latency is a guardrail. A faster path is not worth having if it
changes the result, makes installation fragile, or needs a confusing public
switch to reach.

## What stays out

Polypix does not provide orbit or sensor models, longitude/latitude or CRS
objects, concave geometry, holes, exact cell intersections, NESTED or
mixed-resolution results, MOCs, neighbours, map algebra, timestamps, arbitrary
reducers, GPU or distributed execution, or a generic grid interface.

Optional adapters and conversion recipes are welcome at the boundary. They must
not pull another project's object model or runtime dependencies into the core.

## Adding something new

A feature needs a real workload behind it, one clear result, tests,
documentation, and a maintenance cost that fits a library this size. Work in the
kernel should close a correctness gap, or remove a measured bottleneck or a
material intermediate. Convenience work should prevent user error or
interoperability trouble often enough to earn its place. That someone might need
it one day is not enough.

Polypix is pre-1.0, so we would rather make a clean breaking change than carry a
permanent compatibility alias. Before 1.0, the centre-sampling contract, the
public API, the supported wheels, the correctness tests, and the performance
checks all have to be settled.
