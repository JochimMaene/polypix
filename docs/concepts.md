# Concepts

## HEALPix Resolution And IDs

Polypix uses fixed-resolution HEALPix NESTED ordering. The public API calls the
HEALPix order a `resolution`:

```text
nside = 2 ** resolution
cell_count = 12 * 4 ** resolution
```

| Resolution | `nside` | Cells on the sphere |
| ---: | ---: | ---: |
| 0 | 1 | 12 |
| 1 | 2 | 48 |
| 2 | 4 | 192 |
| 8 | 256 | 786,432 |
| 12 | 4,096 | 201,326,592 |

Polypix accepts resolutions from 0 through 29. Cell values are ordinary NESTED
pixel indices between zero and `cell_count - 1`; they are not packed tokens and
do not encode a resolution. One result contains one resolution.

RING ordering, mixed-resolution cells, MOCs, neighbors, hierarchy traversal,
and map operations are deliberately outside this focused library.

## Body-Centered Geometry

Inputs and geometry outputs use body-centered `(x, y, z)` vectors on the unit
sphere. Input magnitude is ignored and normalized robustly. The coordinate
frame can represent Earth or another sphere, but Polypix does not assign WGS84,
geodetic, ellipsoid, or CRS semantics.

Satellite and sensor coverage are leading uses. Footprint generation—such as
orbit propagation, attitude, sensor projection, or ellipsoid intersection—
belongs upstream.

## Center-Sampled Coverage

A cell is covered when its center lies inside a footprint or on its boundary.
This is a representative center sample, not conservative intersection,
full-containment, or fractional-area coverage.

Inputs are convex spherical polygons whose edges follow shorter great-circle
arcs. Longitude wraparound and poles do not need special treatment because the
kernel operates on three-dimensional vectors.

Either vertex orientation and one repeated closing vertex are accepted.
Degenerate, duplicate, antipodal, self-intersecting, and non-convex geometry is
rejected.

## Batches And Segments

`cover_footprint()` accepts one footprint, a dense batch, or a ragged sequence.
`cover_strip()` turns consecutive pairs from two sampled edges into independent
convex quadrilaterals.

Both return one `Coverage` with flat cells and offsets:

```text
cells for item i = cells[offsets[i] : offsets[i + 1]]
```

This representation avoids one Python object per footprint while keeping input
item boundaries. `counts` is derived from adjacent offsets.

## Candidate Cells

Pass `candidate_cells` when only a sparse existing set matters:

```python
coverage = px.cover_strip(
    left_edge_xyz,
    right_edge_xyz,
    resolution=12,
    candidate_cells=aoi_cells,
)
```

Candidates are standard NESTED indices at the requested resolution and have set
semantics. The native kernel tests their centers directly; it does not first
materialize complete global coverage.

## Parallel Execution

Large batches can run across native worker threads while the Python GIL is
released:

```python
sequential = px.cover_footprint(batch, resolution=9, threads=1)
automatic = px.cover_footprint(batch, resolution=9)
```

`threads=None` selects the automatic policy. A positive integer selects a
per-call worker count. Results are identical across thread settings.
