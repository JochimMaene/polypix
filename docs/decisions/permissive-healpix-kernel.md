# Permissive HEALPix Kernel Evaluation

Status: **superseded by the [owned RING kernel](owned-healpix-kernel.md)**

Date: 2026-07-26; superseded 2026-07-27

## Decision

Use [`cdshealpix` 0.9.1](https://docs.rs/cdshealpix/0.9.1/cdshealpix/)
as the first permissive replacement for `healpix_cxx`. Keep every CDS type and
semantic private to the native kernel.

This is not a drop-in substitution. CDS provides an exact polygon-*overlap*
query, whereas Polypix promises center-sampled convex coverage. The production
kernel must use the overlap result only as a hierarchical candidate traversal:

1. Validate and orient the convex, hemisphere-bounded XYZ footprint in
   Polypix.
2. Derive normalized inward great-circle edge normals.
3. Select the intended polygon side from those half-spaces. Do not delegate
   side selection to the CDS control-point heuristic.
4. Request the exact CDS polygon overlap BMOC.
5. Expand BMOC ranges flagged as fully covered directly.
6. For partial ranges, include only leaves whose CDS center satisfies every
   Polypix half-space, including the documented boundary tolerance.
7. Return raw fixed-resolution NESTED `u64` indices. Do not expose BMOC, MOC,
   flags, or mixed resolutions.

Adopt this path now rather than writing a new HEALPix traversal. The evaluation
used `healpix_cxx` temporarily as an oracle and benchmark baseline; the
integrated implementation removed it from the build and runtime chain.

## Why This Path

### License

The crate is offered under
[`Apache-2.0 OR MIT`](https://docs.rs/crate/cdshealpix/0.9.1/source/Cargo.toml.orig),
so Polypix can consume it under Apache-2.0. It has no native system-library
dependency. This removes the GPL and CFITSIO runtime chain from published
wheels.

The final distribution still needs a lockfile-wide license audit and generated
third-party notices. The spike verified the direct crate license, not every
transitive dependency's notice obligations.

### Required APIs Exist

The NESTED layer provides:

- `center(depth, hash) -> (lon, lat)` in radians;
- `vertices(depth, hash) -> [(lon, lat); 4]`;
- `polygon_coverage(..., exact_solution) -> BMOC`;
- flagged BMOC ranges and flat fixed-depth iteration.

The upstream documentation describes polygon coverage as a hierarchical view
of cells *overlapped* by a polygon and notes that exact mode needs more testing.
The source is explicit about the approximation used when exact mode is disabled:
[CDS polygon implementation](https://docs.rs/cdshealpix/0.9.1/src/cdshealpix/nested/mod.rs.html#4192-4310).
The BMOC API preserves whether a hierarchical cell is fully or partially
covered and can produce flagged fixed-depth ranges:
[BMOC documentation](https://docs.rs/cdshealpix/0.9.1/cdshealpix/nested/bmoc/struct.BMOC.html).

Therefore CDS supplies the difficult, HEALPix-aware overlap traversal while
Polypix retains authority over its much narrower center-in-convex-polygon
contract.

### Cell Geometry Matches

Across sampled cells at resolutions 0, 1, 5, 12, and 29:

- maximum center-vector component difference from `healpix_cxx`:
  `1.11e-16`;
- maximum corner-vector component difference after ordering normalization:
  `8.33e-16`.

CDS returns corners in south, east, north, west order. The current
`healpix_cxx` wrapper returns north, west, south, east, so the compatibility
mapping is:

```text
CDS indices [2, 3, 0, 1]
```

The public goal currently promises four corners, not a named starting corner.
Even so, retaining this order costs essentially nothing and avoids an
unnecessary behavioral break.

### Correctness Spike

The evaluation spike was intentionally discarded after its regression cases
and acceptance checks moved into the production tests and scorecard.

The exact-overlap plus flagged-range center filter matched an independent
fixed-resolution brute-force center oracle for:

- ordinary, antimeridian, north-pole, south-pole, thin, reversed-orientation,
  and exact-center-on-boundary cases;
- 200 deterministic randomized small convex quadrilaterals at resolutions 5
  and 7.

Six representative cases also matched `healpix_cxx` membership exactly after
removing Polypix's old packed-ID prefix.

One important failure was found and fixed in the adapter design:
`ContainsSouthPoleMethod::ControlPointIn` chose the wrong candidate region for
a thin randomized footprint near longitude -45 degrees and latitude
-35.7 degrees, yielding an empty result. The safe adapter:

- computes desired south-pole membership from Polypix's convex half-spaces;
- reproduces CDS's default side choice;
- uses the optimized default query only when both choices agree;
- otherwise calls the explicit `ContainsSouthPole` or
  `DoNotContainsSouthPole` variant.

This policy passed the randomized corpus. Production tests must retain the
regression case.

Exact and approximate CDS overlap happened to yield the same center samples in
this bounded corpus. Production must nevertheless request exact mode: an
overlap false negative cannot be repaired by the later center filter, and
upstream documents the approximate cell-edge assumption.

## Bounded Performance Evidence

Machine: Intel i7-1165G7, 4 cores/8 threads, WSL2 Linux. Figures are best of
five and in microseconds.

| Batch | Resolution | CDS adapted, exact | Current public `healpix_cxx`, 1 thread |
| ---: | ---: | ---: | ---: |
| 256 | 6 | 1,998 | 3,764 |
| 256 | 8 | 7,434 | 6,878 |
| 4,096 | 6 | 29,689 | 59,434 |
| 4,096 | 8 | 117,385 | 109,522 |

These are feasibility numbers, not publishable comparisons. The CDS timing is
pure Rust and excludes Python conversion and result construction; the current
timing includes the public Python call. The current default-parallel
implementation completed the 4,096-item workloads in 18,517 and 29,810
microseconds, so an unthreaded CDS integration would fail the product goal.

The table does **not** show that CDS or Rust is generally faster than
`healpix_cxx` or C++. Results change with resolution: CDS led the resolution-6
spike, while `healpix_cxx` led the resolution-8 spike. The production decision
is about the complete Polypix design—licensing, packaging, correctness, outer
batch parallelism, and end-to-end throughput—not an inherent language or
kernel-speed advantage. A focused permissive C++ or Polypix-owned kernel remains
a valid replacement if it wins the same public-call scorecard materially.

The evidence supports integration, not a performance-leadership claim:

- CDS is competitive single-threaded on this workload.
- CDS polygon coverage itself is sequential.
- Polypix must parallelize independent footprints at the outer batch layer,
  preserve segment order, and benchmark the completed public call.
- Candidate-cell coverage should bypass BMOC construction and test the
  precomputed candidate centers directly against each footprint.

### Symmetric public-call comparison

After integration, `benchmarks/legacy_cpp_baseline.py` was run unchanged against
released v0.2.1 at commit `20d2df6` and Rust/CDS at commit `a746b1a`. These
figures are medians of seven complete public calls on the same Intel i7-1165G7
WSL2 host. Every shared workload produced the same normalized membership.

| Standard workload | v0.2.1 C++ auto | Rust/CDS auto | Rust/CDS speedup |
| --- | ---: | ---: | ---: |
| 4,096 dense quads, resolution 6 | 18.956 ms | 7.908 ms | 2.40× |
| 4,096 dense quads, resolution 9 | 34.422 ms | 42.318 ms | 0.81× |
| 4,096 strip intervals, resolution 9 | 27.012 ms | 22.225 ms | 1.22× |
| 512 footprints × 8,192 candidates, resolution 12 | 7.452 ms | 3.517 ms | 2.12× |
| one footprint, resolution 9 | 0.050 ms | 0.092 ms | 0.54× |

With both implementations restricted to one thread, the respective Rust/CDS
speedups were 2.36×, 0.82×, 1.03×, 1.79×, and 0.41×. This confirms a workload
and resolution-dependent result. The integrated design leads several primary
batch paths, while C++ remains faster at dense resolution 9 and for single-call
latency on this host.

The crate warns that `target-cpu=native` BMI2 selection can regress badly on
some AMD Ryzen processors. Published portable wheels should not use
`target-cpu=native`; architecture-specific tuning must be justified by the
cross-machine benchmark matrix:
[upstream performance warning](https://github.com/cds-astro/cds-healpix-rust#warning).

## Build And Portability

The crate declares Rust 1.81 and is pure Rust. A clean cached release build of
the spike took 23.2 seconds. The stripped LTO spike executable was 484 KiB and
linked only the normal C runtime, `libm`, and `libgcc_s` on Linux.

The main build-cost concern is source breadth: version 0.9.1 has no useful
feature gates and resolved 81 packages for the spike, including map, image,
compression, date, and serialization dependencies unrelated to Polypix. This
does not create Python runtime dependencies, and LTO strips unused code, but it
increases supply-chain review and source-build cost.

Do not fork or vendor a reduced CDS subset now. First measure production wheel
size, clean CI build time, and audit burden. If those become material, ask
upstream for feature gating before maintaining a Polypix fork.

Rust and PyO3/maturin are credible for Linux x86-64/ARM64, macOS x86-64/ARM64,
and Windows x86-64, but this spike ran only on Linux x86-64. The agreed wheel
matrix remains an acceptance gate, not an inferred guarantee.

## Rejected Alternatives

### `healpix_bare`

The official BSD-3-Clause
[`healpix_bare`](https://healpix.sourceforge.io/html/intro_HEALPix_Software_Package.htm)
provides pixel-index/angle/vector conversion and RING/NESTED conversion. It
does not provide polygon coverage. Using it would still require Polypix to
design and maintain the hardest traversal and curved-cell intersection logic.
It adds no material capability over the working CDS path and is therefore not
the selected runtime.

It may be useful later as an independent mapping oracle, but that does not
justify a runtime dependency.

### A New Custom Traversal

A minimal custom hierarchical traversal sounds small but must conservatively
classify curved HEALPix cell boundaries against great-circle polygon edges,
handle base-cell seams and poles, prove no center false negatives, and remain
fast through resolution 29. The CDS source already contains this specialized
logic.

Writing it now would duplicate risk and maintenance without evidence that CDS
fails the production acceptance gates. Reconsider only if the integrated CDS
path cannot meet correctness, performance, wheel, or dependency-burden
requirements.

The post-integration evidence triggered that reconsideration. See
[`owned-healpix-kernel.md`](owned-healpix-kernel.md) for the approved bounded
spike and its admission gates. CDS remains the production kernel meanwhile.

## Production Acceptance Gates

The integration is accepted for publication when:

1. Port the fixed and randomized spike cases into permanent tests, including
   the control-point regression.
2. Compare against an independent brute-force center oracle at tractable
   resolutions and the temporary `healpix_cxx` oracle at higher resolutions.
3. Specify and freeze the floating-point boundary tolerance. The spike's
   `1e-14` closed-boundary tolerance intentionally differs from the current
   backend on rare near-edge centers.
4. Add deterministic outer-batch threading and explicit `threads` control.
5. Benchmark complete Python calls for dense, ragged, strip, sparse-candidate,
   polar, antimeridian, thin, and boundary-heavy workloads.
6. Build and import-test every supported wheel.
7. Audit all locked licenses and regenerate third-party notices.
8. Confirm that wheel size and clean build time remain proportionate; seek CDS
   feature gates only if measurements justify it.

The `0.3.0` integration passes the local correctness, threading, benchmark,
wheel-content, and license gates and removes the C++/CFITSIO build chain. The
hosted platform matrix remains a release gate: a release is not published if
any supported wheel fails to build, install, or run the test suite.

If a release gate fails, keep the Python contract and evaluate a narrowly
scoped custom traversal using the same corpus; do not add backend selection to
the public API.
