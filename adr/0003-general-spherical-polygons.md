# ADR-0003: Cover general spherical polygons

- Status: Accepted; geographic input amended by ADR-0004
- Date: 2026-08-25

## Context

Area-of-interest masks such as Europe are commonly concave, multipart, and may
contain holes. Requiring every caller to split such a region into convex pieces
duplicates difficult spherical geometry work and makes shared-edge cells and
reducers easy to get wrong.

## Decision

`cover_polygon()` accepts simple convex or concave XYZ boundaries. `Polygon`
groups an outer boundary with holes, and `MultiPolygon` groups components into
one result segment. Each component fits inside an open hemisphere and uses the
shorter great-circle arc between adjacent vertices.

The native kernel scans conservative HEALPix RING bounds and checks cell centers
against the outer boundary and holes. Convex components reuse the existing
half-space path. Multipart hits are sorted and deduplicated before reductions.
Polypix still does not accept longitude/latitude, GeoJSON, CRS objects, or GIS
operations. ADR-0004 later adds a narrow geographic input adapter without
changing the native geometry model or adding CRS operations.

## Consequences

Callers can build reusable candidate-cell AOIs without supplying their own
triangulation. General boundary checks and validation grow with boundary detail,
so the initial target is simplified AOIs with hundreds rather than tens of
thousands of vertices. A boundary index belongs here only after a representative
benchmark shows it is needed.
