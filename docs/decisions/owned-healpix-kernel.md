# Owned HEALPix Kernel Evaluation

Status: **approve one bounded traversal spike; do not replace CDS yet**

## Decision

It is now reasonable to test a Polypix-owned HEALPix kernel, but not to start a
general HEALPix rewrite. The only admissible implementation is the smallest
internal kernel needed by the frozen public API:

- fixed-resolution NESTED center coordinates;
- four boundary vertices;
- center-sampled coverage of validated convex spherical polygons;
- direct filtering of explicit candidate cells.

There will be no RING support, MOC API, neighbour operations, projections,
general polygon library, backend selector, or public algorithm controls.

CDS remains the production implementation until an owned spike passes the
correctness and performance gates below. If it fails, delete the spike.

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

## Spike Shape

The spike should compete as one alternative implementation of `cover_polygon`
behind an internal compile-time switch or benchmark-only entry point. It should:

1. traverse the 12 NESTED base cells directly;
2. reject or accept subtrees using conservative spherical bounds;
3. descend ambiguous cells and test the requested-resolution center against the
   already prepared polygon half-spaces;
4. append raw NESTED ranges or leaves directly to the result, without building a
   general MOC;
5. reuse the existing outer Rayon batch parallelism and Python boundary.

The difficult requirement is conservative classification of curved HEALPix
cells at seams and poles. A fast algorithm that can miss a valid center is not a
candidate. If proving the bound requires recreating a general spherical
intersection library, stop: that would violate the narrow-kernel constraint.

## Admission Gates

The production CDS dependency is removed only if the owned kernel:

1. has zero membership differences against brute-force cell-center enumeration
   across all cells at tractable resolutions, the adversarial corpus, and a
   much larger fixed-seed randomized corpus;
2. agrees with independent HEALPix center and boundary fixtures through
   resolution 29, including faces, seams, and poles;
3. improves the geometric mean of the primary single-thread public-call
   workloads by at least 20% on the benchmark host, with no primary workload
   more than 5% slower;
4. retains the improvement with automatic threading and improves, rather than
   further regresses, single-footprint latency;
5. demonstrates the same direction on x86-64 and ARM64 before release;
6. materially reduces the locked dependency graph and clean source-build cost;
7. remains a private, focused module whose maintenance burden is proportionate
   to the measured gain.

Performance thresholds are deliberately material. A marginal result does not
justify owning delicate HEALPix geometry.

## Current Recommendation

Proceed with the bounded center-specific traversal spike. Do not incrementally
replace CDS utilities, expose another backend, or promise dependency removal.
The C++ comparison shows enough headroom to investigate, while the failed center
primitive shows that ownership alone is not an optimization.
