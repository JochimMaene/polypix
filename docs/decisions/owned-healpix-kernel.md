# Owned HEALPix Kernel Evaluation

Status: **accepted; owned RING-first kernel replaces CDS**

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

The owned implementation is now the production implementation. Release remains
gated by the existing multi-platform wheel and smoke-test workflows.

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
is not yet justified. Commit `a8cd3cc` preserves the reproducible prototype
state used for these measurements.

RING is part of the architectural result rather than a second supported
ordering. Direct ring spans are the source of the speedup, while conversion of
materialized results to NESTED consumes a meaningful share of high-output
calls. Supporting both would add code and choices without helping the primary
path.

## Admission Gates And Evidence

The owned kernel was admitted after it:

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
5. delegates x86-64 and ARM64 release confirmation to the existing wheel smoke
   matrix before publication;
6. materially reduces the locked dependency graph and clean source-build cost;
7. remains a private, focused module whose maintenance burden is proportionate
   to the measured gain.

Performance thresholds are deliberately material. A marginal result does not
justify owning delicate HEALPix geometry.

## Result

The ring geometry, fixed-quadrilateral path, paired-edge strip path, ragged
polygon path, sparse candidate filtering, centers, and boundaries now live in
one private production module. Independent fixtures cover polar, equatorial,
seam, and resolution-29 geometry. CDS was removed from the locked graph. There
is no algorithm selector or NESTED compatibility layer.

Ownership alone is not the optimization: the improvement comes from deleting
the overlap/BMOC pipeline in favor of direct center work.

## Numerical Audit

The production geometry was compared directly with the official HEALPix C++
implementation and `cdshealpix` 0.9.1 after the owned kernel was completed.
The audit covered every cell through resolution 6 and 20,106 targeted cells at
resolution 29, including cap boundaries, transition rings, seams, and random
indices.

Centers and corners agreed with HEALPix C++ within `9.44e-16` and `7.78e-16`
respectively. Both implementations avoid polar cancellation by computing
`z = (1 - a) * (1 + a)` and the transverse component from `a` rather than
recovering it as `sqrt(1 - z*z)`. Polypix retains this formulation. Latitude
range pruning similarly computes a normalized edge's transverse component
directly with `hypot(x, y)`.

Polypix deliberately does not copy the floating-point ring decoder in
`cdshealpix::ring`. That decoder loses integer precision at resolution 26 and
above around polar-ring boundaries. At resolution 29, the last north-cap cell
differs from both Polypix and HEALPix C++ by about `3.83e-6` in one Cartesian
component. Polypix starts its square-root estimate in floating point but
corrects it to the exact integer result with overflow-safe integer arithmetic.
A resolution-29 fixture pins the affected transition.

The remaining CDS projection precautions—signed-zero preservation, inverse
projection clamping, and a pole guard—serve longitude/latitude projection APIs
that Polypix does not expose. Adding those paths would not improve the direct
RING-to-vector kernel.
