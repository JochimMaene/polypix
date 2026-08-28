# ADR-0005: Add exact overlap coverage

- Status: Accepted
- Date: 2026-08-27

## Context

Center sampling is the right default for coverage maps, but it can omit a cell
whose area is clipped by a region and can return nothing for a sub-cell region.
Conservative spatial indexes, whole-cell footprint exchange, and narrow swaths
need every touched cell instead.

The four HEALPix corners are not an exact cell polygon: its edges are curved.
Expanding center coverage by one neighbor ring is also neither exact nor enough
to seed a region that contains no center.

## Decision

`cover_cap()`, `cover_polygon()`, and `cover_sweep()` accept
`mode="center" | "overlap"`. Center remains the default. Overlap returns every
cell whose true HEALPix area intersects the region, including tangency, and
resolves floating-point boundary ties toward inclusion.

The owned RING kernel remains dependency-free. It reuses the center scan's
one-ring latitude guard, pads each per-ring longitude interval by one cell on
either side, evaluates HEALPix edges through the analytical face-coordinate
map, and tests cap distance or great-circle intersections on those curves. The
center path remains separate and unchanged.

Reducers keep binary membership: even a tiny intersection contributes one
count or the region's whole weight. `Coverage` does not store the mode because
it stores membership rather than geometry provenance.

## Consequences

Sub-cell regions no longer disappear in overlap mode, and touched-cell results
can be used as conservative spatial indexes. Adjacent regions may share cells,
and the returned whole-cell area can overstate the region near its boundary.

Exact curved-boundary tests cost more than center predicates, and the per-cell
test carries no edge pruning: a cell is tested against every footprint edge, so
one cell costs time linear in the vertex count where the center path bins edges
by height. Overlap coverage is consequently for coarse footprints, which the
performance guide states with measured figures. Conservative pruning around
curved edges can only lose cells when it is wrong, so it is deferred to a change
that can design and prove it rather than bundled with the predicate itself.

The first release reuses ordinary coverage reduction for overlap caps rather
than adding a fused special case. Full-containment, bounding-box,
fractional-area, NESTED, and MOC APIs remain out of scope.

## Alternatives considered

- **Corner polygons:** short, but incorrectly replaces curved HEALPix edges
  with great-circle arcs.
- **Neighbor expansion:** admits false positives and cannot recover an empty
  center result.
- **External overlap backend:** restores the dependency, build, provenance, and
  batch-performance costs rejected by ADR-0001.
