# Project goal

Last updated: 2026-08-22

Polypix has a narrow aim: take large batches of already-resolved spherical
regions and map them to deterministic, center-sampled HEALPix cells. It should
be fast at that job, easy to install, and small enough to understand.

## Where it fits

```text
physical model or source geometry
    -> resolved directions, caps, polygons, or sweep edges
Polypix
    -> validated RING cell membership and reductions
maps, time handling, statistics, and visualisation
```

The code before Polypix owns orbit propagation, attitude, sensor models,
coordinate frames, body shape, terrain, clocks, and physical access rules. The
code after it owns time, persistence, map algebra, plotting, and domain-specific
statistics. Polypix accepts the resolved geometry; it does not rebuild the
model that produced it.

## What it accepts

The core inputs are:

- finite, non-zero Cartesian direction vectors in a caller-defined frame;
- spherical caps;
- convex spherical polygons whose edges use the shorter great-circle arcs;
- paired sampled edges, with one sweep segment between each pair of samples.

Coverage is based on cell centres. A cell is included when its centre is inside
the region or on its boundary. This is not a promise about cell intersection,
containment, area coverage, or a conservative no-false-negative index.

The grid is fixed-resolution HEALPix in RING order, from resolution 0 through
29. Results contain standard RING IDs and preserve the input segments. The
library supports sparse queries, per-cell counts and sums, and ordinal revisit
summaries. Callers map those summaries to their own time edges and gap rules.

The full interface and its validation rules live in the API documentation.

## Why the kernel is ours

The main workloads are large batches of small regions and sweeps. The first
external approach built an overlap intermediate before doing centre filtering;
direct RING traversal removes that work, and the benchmarks showed the
difference. Keeping the kernel here also means users do not need a system
HEALPix installation.

Python handles inputs and results. Rust handles geometry, traversal, reductions,
and parallel work. NumPy is the only runtime dependency.

We measure the complete public call, including conversion, validation,
allocation, and result construction. Large-batch throughput is the priority;
single-call latency is a guardrail. A faster path is not useful if it changes
the result, makes installation fragile, or requires a confusing public switch.

## What stays out

Polypix does not provide orbit or sensor models, longitude/latitude or CRS
objects, concave geometry, holes, exact cell intersections, NESTED or
mixed-resolution results, MOCs, neighbours, map algebra, timestamps, arbitrary
reducers, GPU or distributed execution, or a generic grid interface.

Optional adapters and conversion recipes are welcome at the boundary. They must
not pull another project's object model or runtime dependencies into the core.

## Adding something new

A feature needs a real workload, one clear result, tests, documentation, and a
maintenance cost that fits the library. Kernel work should close a correctness
gap or remove a measured bottleneck or material intermediate. Convenience work
should repeatedly prevent user error or interoperability trouble. “Someone
might need it” is not enough.

Polypix is pre-1.0, so a clean breaking change is preferable to a permanent
compatibility alias. Before 1.0, the center-sampling contract, public API,
supported wheels, correctness tests, and performance checks must be settled.
