# Concepts

## HEALPix Resolution

Polypix uses HEALPix NESTED ordering. The public API calls the HEALPix order a
`resolution`.

For a resolution `r`:

- `nside = 2 ** r`,
- the sphere contains `12 * nside ** 2` cells,
- increasing the resolution subdivides each cell into four children.

| Resolution | `nside` | Cells on the sphere |
| ---: | ---: | ---: |
| 0 | 1 | 12 |
| 1 | 2 | 48 |
| 2 | 4 | 192 |
| 8 | 256 | 786,432 |
| 12 | 4,096 | 201,326,592 |

Polypix accepts resolutions from 0 through 29.

## Packed Cell IDs

Polypix returns packed `uint64` cell IDs instead of bare HEALPix pixel indices.
Each packed ID contains both:

- the HEALPix resolution,
- the NESTED pixel index at that resolution.

This allows one array to contain cells from multiple resolutions. The packing
format is not part of the public API. Treat returned values as stable opaque
tokens and use `center()` or `boundary()` to recover longitude/latitude
geometry.

## Center-In-Polygon Coverage

Polypix includes a HEALPix cell if the cell center lies inside the polygon.
Cells that touch the polygon boundary but have centers outside the polygon are
not included.

This is useful when you need a compact representative cover. It is not a
conservative overlap cover, and it should not be used as a substitute for
full polygon intersection.

## Spherical Polygons

Input polygons are interpreted on the unit sphere. Edges are great-circle
segments between consecutive vertices.

Longitude/latitude vertices are passed as `(longitude, latitude)` pairs in
degrees. Longitudes may wrap around the antimeridian; latitudes must be between
`-90` and `90` degrees. Unit-vector vertices are passed as normalized
`(x, y, z)` coordinates.

Polypix normalizes polygon orientation internally and rejects invalid geometry:

- fewer than three unique vertices,
- duplicate vertices,
- degenerate edges,
- non-convex polygons,
- non-finite coordinates,
- unit vectors that are not normalized.

A repeated final vertex is accepted as a closed-ring marker and is removed
before coverage is computed.

## Batch Layout

`Polygon` represents one polygon. `MultiPolygon` represents a batch of polygons
using a flat vertex array and offsets before calling the native implementation.

Use `Polygon.from_lonlat(...)` or `Polygon.from_xyz(...)` for one polygon. Use
`MultiPolygon.from_lonlat(...)` or `MultiPolygon.from_xyz(...)` for dense or
ragged batches.

For `N` polygons, the internal offsets array has length `N + 1`:

```text
polygon i vertices = vertices[offsets[i] : offsets[i + 1]]
```

For a `Polygon`, `cover()` returns one `uint64` cell ID array. For a
`MultiPolygon`, it returns a flat `uint64` cell ID array plus a `counts` array
with one covered-cell count per input polygon.

## Parallel Execution

For sufficiently large batches, Polypix parallelizes coverage across polygons.
The default worker count is based on hardware concurrency and the requested
resolution. Set `POLYPIX_NUM_THREADS` to a positive integer to choose a fixed
worker count:

```bash
POLYPIX_NUM_THREADS=4 python script.py
```

Threading changes throughput, not the returned cell IDs.
