# Rejected Permissive HEALPix Backends

Status: **superseded by the [owned RING kernel](owned-healpix-kernel.md)**

Date: 2026-07-26; superseded 2026-07-27

## Decision History

Polypix briefly selected `cdshealpix` 0.9.1 as a permissively licensed
replacement for the former GPL HEALPix C++ dependency. CDS is available under
Apache-2.0 or MIT and removed the C++/CFITSIO packaging chain, but its general
polygon-overlap BMOC pipeline was broader than Polypix's center-only contract.

The adapter remained private and used CDS overlap results only as candidates.
Polypix retained convexity validation, side selection, the closed-boundary
tolerance, center filtering, batch ordering, and threading. This architecture
was correct but was not the final performance direction.

## Evidence That Triggered Replacement

End-to-end measurements were workload-dependent. Against released v0.2.1 on
the same Intel host, the Rust/CDS implementation was 2.40x faster for 4,096
resolution-6 quadrilaterals but 1.23x slower for the same batch at resolution
9 and about 2x slower for one resolution-9 footprint. It also pulled 53
uniquely named crates into the normal dependency subtree because CDS had no
useful feature gates.

Those results justified a bounded owned-kernel spike. The direct RING scan then
removed overlap classification and BMOC materialization, producing the
order-of-magnitude primary-workload gains recorded in the
[accepted decision](owned-healpix-kernel.md). CDS is no longer a runtime,
build, or test dependency.

## Durable Regression Evidence

The CDS evaluation exposed a control-point side-selection defect for a thin
southern footprint. That geometry remains a named regression test even though
the backend was removed. Independent RING-center oracles, polar and seam cases,
wide randomized footprints, and the HEALPix C++ geometry audit now own the
correctness evidence.

## Other Rejected Option: `healpix_bare`

The official `healpix_bare` subset is BSD-3-Clause and provides cell
conversion, but not polygon coverage. It would still leave Polypix responsible
for the center traversal while adding another runtime dependency. Once the
owned traversal proved both faster and correct, `healpix_bare` no longer
earned a production role. It remains a possible external oracle, not an
implementation dependency.

## Licensing Boundary

The owned RING-to-face decoder was independently derived from the published
HEALPix equations. The GPL HEALPix C++ implementation was used only as an
external oracle. The face-coordinate mapping adapted from Astrometry.net is
BSD-3-Clause and is recorded in `THIRD_PARTY_NOTICES.md`.
