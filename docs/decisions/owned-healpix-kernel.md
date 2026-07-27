# Owned HEALPix Kernel Evaluation

Status: **prototype succeeded; develop the bounded RING-first kernel, but keep
CDS until the production gates pass**

## Decision

The direct RING scan prototype demonstrates that a Polypix-owned center-only
kernel can produce order-of-magnitude improvements on the primary dense
quadrilateral and strip workloads. This justifies developing the smallest
production kernel needed by the target API:

- fixed-resolution RING center coordinates;
- four boundary vertices;
- center-sampled coverage of validated convex spherical polygons;
- direct filtering of explicit candidate cells.

There will be no NESTED support, MOC API, neighbour operations, projections,
general polygon library, backend selector, or public algorithm controls.

CDS remains the production implementation until an owned spike passes the
correctness, packaging, and multi-platform gates below.

## Why Reconsider

The original CDS decision explicitly allowed reconsideration when performance
or dependency burden became material. Both now have evidence:

- The symmetric v0.2.1 comparison shows that the integrated Rust/CDS path is
  1.23× slower for 4,096 dense resolution-9 footprints and about 2× slower for
  one resolution-9 footprint on the measured Intel host. It leads other primary
  batch paths, so the result is mixed rather than a general regression.
- Polypix needs center membership, while CDS first constructs an exact
  polygon-overlap BMOC and Polypix then center-filters partial leaves. A
  center-specific traversal could avoid overlap work and the intermediate BMOC.
- `cdshealpix` 0.9.1 has no feature gates useful to Polypix. Its normal
  dependency subtree contains 53 uniquely named crates, including functionality
  for images, compression, dates, maps, and serialization. LTO keeps the wheel
  small, but source builds and supply-chain review still carry this breadth.

These facts justify a spike. They do not prove that a new kernel will be faster.

## Small Primitive Experiment

A clean internal NESTED-to-XYZ center implementation was temporarily substituted
for `cdshealpix::Layer::center`. It retained identical scorecard memberships and
passed the existing suite, but repeated sequential scorecard runs did not
demonstrate an improvement and several batch timings regressed. Host-frequency
variance made the precise size unsuitable as a claim; the absence of a
repeatable win was enough to remove the code.

This rejects incremental reimplementation of isolated mapping primitives as a
performance strategy. CDS already has optimized Morton decoding. Any credible
win must come from eliminating the overlap/BMOC/center-filter pipeline with a
center-specific traversal, not from accumulating replacement utilities one by
one.

## Prototype Outcome

The successful prototype did not traverse NESTED cells. It exploited the
geometry of the requested center rule directly:

1. bound each validated footprint across the `4 * nside - 1` HEALPix
   iso-latitude rings;
2. walk only the possible longitudes on each crossed ring;
3. test centers against the polygon half-spaces;
4. append standard RING IDs directly;
5. fuse fixed-quadrilateral preparation and reuse output storage;
6. distribute coarse independent chunks through Rayon.

On the Intel i7-1165G7 benchmark host, a representative 21-repeat
single-thread run measured 10.82x for 4,096 resolution-6 quadrilaterals, 16.70x
for the same batch at resolution 9, and 17.05x for 4,096 resolution-9 strip
quadrilaterals. Membership matched the CDS-backed kernel. Ragged 3--6 vertex
polygons reached 7.55x and one-footprint latency 6.15x, so a universal 10x claim
is not yet justified. Full details and the reproducible command live in
`spikes/NOTES.md`.

RING is part of the architectural result rather than a second supported
ordering. Direct ring spans are the source of the speedup, while conversion of
materialized results to NESTED consumes a meaningful share of high-output
calls. Supporting both would add code and choices without helping the primary
path.

## Admission Gates

The production CDS dependency is removed only if the owned kernel:

1. has zero membership differences against brute-force cell-center enumeration
   across all cells at tractable resolutions, the adversarial corpus, and a
   much larger fixed-seed randomized corpus;
2. agrees with independent HEALPix center and boundary fixtures through
   resolution 29, including faces, seams, and poles;
3. preserves the order-of-magnitude direction on the primary fixed-quad and
   strip public-call workloads and materially improves the documented ragged
   path, with no primary workload slower;
4. retains the improvement with automatic threading and improves, rather than
   further regresses, single-footprint latency;
5. demonstrates the same direction on x86-64 and ARM64 before release;
6. materially reduces the locked dependency graph and clean source-build cost;
7. remains a private, focused module whose maintenance burden is proportionate
   to the measured gain.

Performance thresholds are deliberately material. A marginal result does not
justify owning delicate HEALPix geometry.

## Current Recommendation

Absorb the ring geometry and fixed-capacity preparation into one private
production module. Add the native paired-edge input path before optimizing
secondary cases, then address ragged polygons and sparse candidates with
separate measured fast paths only if they stay small. Do not expose an
algorithm selector or preserve NESTED compatibility during `0.x`.

CDS remains the correctness reference and shipping implementation until the
admission gates pass. Ownership alone is still not the optimization: deleting
the overlap/BMOC pipeline in favor of direct center work is.
