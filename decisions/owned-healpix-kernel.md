# Owned HEALPix Kernel

Status: **accepted; owned RING-first production kernel**

## Decision

Polypix adopted an owned, center-only HEALPix RING kernel after a direct scan
prototype met the project's internal admission gates for the primary dense
quadrilateral, ragged, and paired-edge sweep workloads. Its initial production surface
implemented:

- fixed-resolution RING center coordinates;
- four cell-corner vertices;
- center-sampled coverage of validated convex spherical polygons;
- direct filtering of explicit candidate cells.

Exact cap spans, fused cap counts, and segmented occupancy reduction were admitted
later by [Exact Caps and Segmented Occupancy Reduction](cap-and-occupancy-primitives.md).

NESTED support, MOCs, neighbour operations, projections, a general polygon
library, backend selection, and public algorithm controls remain outside the
project.

## Context

The unreleased CDS adapter prompted the kernel experiment for two reasons:

- Polypix needs center membership, while CDS first constructs an exact
  polygon-overlap BMOC and Polypix then center-filters partial leaves. A
  center-specific traversal could avoid overlap work and the intermediate BMOC.
- `cdshealpix` 0.9.1 has no feature gates useful to Polypix. Its normal
  dependency subtree contains 53 uniquely named crates, including functionality
  for images, compression, dates, maps, and serialization. LTO keeps the wheel
  small, but source builds and supply-chain review still carry this breadth.

## Rejected Alternatives

The briefly selected `cdshealpix` adapter was never released. It remained
correct by using CDS overlap results only as candidates and retaining Polypix's
own validation, side selection, center filtering, ordering, and threading.
The semantic mismatch and dependency breadth above motivated replacing it with
the direct RING traversal. Its thin-southern-footprint side-selection finding
remains a named regression test.

The BSD-3-Clause `healpix_bare` subset provides cell conversion but not polygon
coverage. It would leave Polypix responsible for traversal while adding a
runtime dependency, so it remains suitable only as a possible external oracle.

A clean internal NESTED-to-XYZ center implementation was temporarily substituted
for `cdshealpix::Layer::center`. It retained identical scorecard memberships and
passed the suite, but did not demonstrate a repeatable improvement and several
batch timings regressed. It was removed. The successful direction eliminated
the overlap/BMOC/center-filter pipeline rather than replacing isolated mapping
primitives.

## Implementation

The production traversal:

1. bound each validated footprint across the `4 * nside - 1` HEALPix
   iso-latitude rings;
2. walk only the possible longitudes on each crossed ring;
3. test centers against the polygon half-spaces;
4. append standard RING IDs directly;
5. fuse fixed-quadrilateral preparation and reuse output storage;
6. distribute coarse independent chunks through Rayon.

RING is part of the architectural result rather than a second supported
ordering. Direct ring spans are the source of the speedup, while conversion of
materialized results to NESTED consumes a meaningful share of high-output
calls. Supporting both would add code and choices without helping the primary
path.

Internal prototype comparisons were used to decide whether owning the kernel
earned its maintenance cost. They are deliberately not published as product
speed claims because the removed internal CDS adapter is not a reproducible
public alternative. Public comparative claims require the separate benchmark
repository described in the [project goal](../docs/project-goal.md).

The following component comparisons are retained as maintenance evidence, not
as cross-library claims. They use release builds and include public-call input
conversion. A future simplification should rerun the same A/B comparison
before removing a load-bearing path:

| Component | Internal A/B workload | July 2026 observation |
| --- | --- | --- |
| Fixed-quadrilateral preparation and predicate | 4,096 small quads, resolution 9, one thread | 3.33 ms enabled versus 4.57 ms through the generic polygon path: 1.37x. |
| Shared candidate-center cache | 4,096 overlapping small quads, two million resolution-12 candidates | Disabling the cache saved about 40 MiB but regressed the serial call by 11% and the automatic call by 1.77x. |
| Automatic scan estimator | 2,000 small quads at resolution 2, automatic versus forced serial | 1.97 ms versus 1.81 ms: about 9% dispatch overhead below the parallel crossover. |

These medians came from an eight-logical-CPU Intel development host. Exact
timings depend on host, toolchain, and the evolving kernel; the A/B workload
and ratio are the durable evidence. A future simplification should rerun the
comparison on its target branch rather than treating these numbers as a public
performance promise.

## Admission Evidence

The owned kernel was admitted after the evidence showed:

1. zero membership differences against brute-force cell-center enumeration
   across all cells at tractable resolutions, the adversarial corpus, and a
   much larger fixed-seed randomized corpus;
2. agreement with independent HEALPix center and corner fixtures through
   resolution 29, including faces, seams, and poles;
3. internal measurements meeting the predeclared gates for the primary
   fixed-quad, paired-edge sweep, and documented ragged public-call workloads;
4. retained improvements with automatic threading and improved, rather than
   further regressed, single-footprint latency;
5. an x86-64 and ARM64 wheel smoke matrix as a release gate;
6. a materially smaller locked dependency graph and clean source-build cost;
7. a private, focused kernel whose maintenance burden is proportionate
   to the measured gain.

Performance thresholds are deliberately material. A marginal result does not
justify owning delicate HEALPix geometry.

## Result

The private production kernel owns ring geometry, exact cap spans and counts,
the fixed-quadrilateral predicate, paired-edge sweep and ragged polygon paths,
z-indexed sparse candidate filtering, direction indexing, centers, and corners. Polygon coverage
uses a center-scan traversal; the four-edge predicate remains unrolled because
its measured gain is material. Independent fixtures cover polar, equatorial,
seam, and resolution-29 geometry. CDS was removed from the locked graph. There
is no algorithm selector or NESTED compatibility layer.

Ownership alone is not the optimization: the improvement comes from deleting
the overlap/BMOC pipeline in favor of direct center work.

## Implementation Provenance

The RING-to-face decoder was independently derived from the published HEALPix
RING numbering equations and twelve-face layout. It was not adapted from the
GPL-licensed HEALPix C++ `ring2xyf` implementation. The official implementation
was used only as an external numerical oracle after Polypix's decoder existed.

The subsequent face-coordinate transform is a small adaptation of
Astrometry.net's BSD-3-Clause HEALPix mapping and is identified in the source
and `THIRD_PARTY_NOTICES.md`. Keeping these origins explicit is part of the
Apache-2.0 release evidence.

Repository history through the 0.3 preparation branch was reviewed and contains
one contributor under two recorded email identities, both belonging to Jochim
Maene. No third-party contributor grant is required for the relicensing.

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

The repository also pins a broad audit against HEALPix C++ through
`healpy==1.19.0`. `tests/test_ring_geometry.py` selects 257 evenly spaced RING
indices at each of resolutions 0, 1, 3, 8, 16, and 29 and checks three
deterministic projections of their centers and corners. Targeted fixtures
retain near-machine-precision checks at transition rings through resolution 29,
while a separate scalar implementation checks every center through resolution
7. The broad and exact reference values can be regenerated with the checked-in
oracle script:

```bash
python tools/generate_ring_geometry_fixtures.py
```

The remaining CDS projection precautions—signed-zero preservation, inverse
projection clamping, and a pole guard—serve longitude/latitude projection APIs
that Polypix does not expose. Adding those paths would not improve the direct
RING-to-vector kernel.
