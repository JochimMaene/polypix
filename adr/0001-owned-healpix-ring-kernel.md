# ADR-0001: Own the HEALPix RING kernel

- Status: Accepted
- Date: 2026-07-27

## Context

Polypix takes regions that have already been worked out and maps them to
HEALPix cell centres. We first tried doing that through CDS. CDS built an
overlap/BMOC result, after which Polypix filtered it down to centre membership.

It did more work than Polypix needed. In the cases that drove the project—many
small polygons, sweeps, and ragged inputs—the extra overlap step was too much.
The dependency tree was much larger than the job called for as well.

Performance was the main reason to own the kernel. Direct RING traversal skips
the overlap result and centre filtering, and it was faster in the batch
benchmarks that mattered here.

Licensing and provenance mattered too. Polypix is distributed under Apache-2.0,
so keeping the production kernel focused leaves fewer source and license notices
to review.

## Decision

Polypix owns the native, center-only HEALPix RING kernel. It walks the relevant
rings, tests cell centres against validated geometry, and emits standard RING
indices directly. It handles caps, convex spherical polygons, paired-edge
sweeps, explicit candidate cells, and the public centre and corner transforms.

RING is the one supported ordering. We are not adding NESTED, MOCs, neighbour
operations, projections, a general polygon library, backend selection, or public
algorithm controls.

## Consequences

The direct traversal is faster in the batch cases that led to this decision, and
the package no longer needs a system HEALPix library at runtime. In return, we
own the geometry code, its fixtures, and its portability tests. We also have to
keep the source and license notices accurate. A replacement would need to match
the semantics, provenance, and relevant benchmarks, not merely provide
functions with similar names.

## Alternatives considered

- **External HEALPix adapter:** it did overlap work before centre filtering and
  brought too many dependencies and another licensing/provenance boundary to
  audit.
- **Mapping-only dependency:** Polypix would still own the coverage traversal,
  so this would add a dependency without removing the hard part.
- **RING and NESTED:** converting results adds cost and does not help the main
  workloads.

## Evidence

The centre and corner fixtures, brute-force membership checks, and the regular
test suite cover the supported resolutions. Benchmarks of complete public calls
showed a useful gain on the target batches. Those results justify this kernel;
they are not a claim that it wins every comparison with every HEALPix library.
