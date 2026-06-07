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
tokens and use `centers()` or `boundaries()` to recover longitude/latitude
geometry.

## Intended Geometry

Polypix is designed for coverage simulation outputs that can be represented as
convex footprints on the unit sphere. Examples include sensor footprints, beam
contours, access regions, and swath strips from satellite, aerial, astronomy,
or other spherical-domain simulations.

Inputs are not projected planar geometry. They are normalized `(x, y, z)` unit
vectors from the sphere center to the footprint vertices, all in one common
frame.

Polypix starts after footprint generation. Orbit propagation, attitude
modeling, sensor projection, and beam-shape modeling belong upstream.

## Center-In-Footprint Coverage

Polypix includes a HEALPix cell if the cell center lies inside the footprint.
Cells that touch the footprint boundary but have centers outside the footprint
are not included.

This is useful when you need a compact representative cover. It is not a
conservative overlap cover, and it should not be used as a substitute for
full footprint intersection.

## Spherical Footprints

Input footprints are interpreted on the unit sphere. Edges are great-circle
segments between consecutive vertices.

Footprint edges may cross the antimeridian because the geometry is evaluated on
the sphere, not in a planar longitude/latitude coordinate system.

Polypix normalizes footprint orientation internally and rejects invalid
geometry:

- fewer than three unique vertices,
- duplicate vertices,
- degenerate edges,
- non-convex footprints,
- non-finite coordinates,
- unit vectors that are not normalized.

A repeated final vertex is accepted as a closed-ring marker and is removed
before coverage is computed.

## Coverage Results

`cover_footprint()` accepts one footprint with shape `(vertices, 3)` or a dense
batch with shape `(footprints, vertices, 3)`. `cover_swath()` accepts two edge
arrays with shape `(samples, 3)` and covers each consecutive interval as one
quadrilateral.

Both functions return `Coverage`. For `N` covered footprints or swath intervals,
the output offsets array has length `N + 1`:

```text
covered cells for item i = cell_ids[offsets[i] : offsets[i + 1]]
```

`Coverage.counts` is derived from the offsets and contains one covered-cell
count per input footprint or swath interval.

## Parallel Execution

For sufficiently large batches, Polypix parallelizes coverage across footprints.
The default worker count is based on hardware concurrency and the requested
resolution. Set `POLYPIX_NUM_THREADS` to a positive integer to choose a fixed
worker count:

```bash
POLYPIX_NUM_THREADS=4 python script.py
```

Threading changes throughput, not the returned cell IDs.
