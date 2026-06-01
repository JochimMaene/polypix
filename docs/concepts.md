# Concepts

## HEALPix Resolution

Polypix uses HEALPix NESTED ordering. The public API calls the HEALPix order a
`resolution`.

For a resolution `r`:

- `nside = 2 ** r`,
- the sphere contains `12 * nside ** 2` cells,
- child cells are obtained by increasing the resolution.

## Packed Cell IDs

Polypix returns packed `uint64` cell IDs instead of bare HEALPix pixel indices.
The packed ID stores both:

- the HEALPix resolution,
- the NESTED pixel index at that resolution.

The packing format is intentionally not part of the public API beyond treating
the returned IDs as stable opaque tokens. Use `center()` and `boundary()` when
you need geometry for returned IDs.

## Center-In-Polygon Coverage

Polypix includes a cell if the cell center lies inside the polygon. It does not
attempt to return every cell that touches or overlaps the polygon boundary.

This makes the rule deterministic and useful for workloads that need a compact
representative cover rather than a conservative overlap cover.

## Convex Spherical Polygons

Input polygons are interpreted on the unit sphere. Edges are great-circle
segments between consecutive vertices.

Longitude/latitude vertices are passed as `(longitude, latitude)` pairs in
degrees. Unit-vector vertices are passed as normalized `(x, y, z)` coordinates.

Polypix normalizes polygon orientation internally and rejects:

- polygons with fewer than three unique vertices,
- duplicate vertices,
- degenerate edges,
- non-convex polygons,
- non-finite coordinates.

A repeated final vertex is accepted as a closed ring marker and is not counted as
a duplicate.

## Batch Layout

`cover()` accepts `Polygon` or `MultiPolygon` objects. `Polygon` represents one
convex spherical polygon. `MultiPolygon` represents a batch of polygons using a
flat vertex array plus offsets before calling C++.

Use `Polygon.from_lonlat(...)` or `Polygon.from_xyz(...)` for a single polygon.
Use `MultiPolygon.from_lonlat(...)` or `MultiPolygon.from_xyz(...)` for dense or
ragged batches.

For `N` polygons, the internal offsets array has length `N + 1`:

```text
polygon i vertices = vertices[offsets[i] : offsets[i + 1]]
```

For `Polygon`, `cover()` returns one cell ID array. For `MultiPolygon`, it
returns a flat cell ID array plus a count array with one covered-cell count per
polygon.
