# Concepts

## HEALPix Resolution And IDs

Polypix uses fixed-resolution HEALPix RING ordering. The public API calls the
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

Polypix accepts resolutions from 0 through 29. Cell values are ordinary RING
pixel indices between zero and `cell_count - 1`; they are not packed tokens and
do not encode a resolution. One result contains one resolution.

NESTED ordering, mixed-resolution cells, MOCs, neighbors, hierarchy traversal,
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

Inputs are convex spherical polygons contained in an open hemisphere. Every
pair of adjacent vertices is joined by the unique shorter great-circle arc.
Those rules determine the represented region: for example, longitudes
`-179°` and `179°` are two degrees apart across the antimeridian, not 358
degrees apart. A hemisphere or larger region cannot be represented. Polypix
rejects detectable ambiguity such as antipodal adjacent vertices or an
exact-hemisphere boundary, but it cannot infer that a caller intended the
other side of an otherwise valid minor-arc polygon.

Longitude wraparound and poles need no special coordinate treatment because
the kernel operates on three-dimensional vectors.

Either vertex orientation and one repeated closing vertex are accepted.
Degenerate, duplicate, antipodal, self-intersecting, and non-convex geometry is
rejected.

## Batches And Segments

`cover_footprint()` accepts one footprint, a dense batch, or a ragged sequence.
`cover_strip()` turns consecutive pairs from two sampled edges into independent
convex quadrilaterals.

Strip samples use the same shorter-great-circle rule. Sampling is therefore
part of the input contract: upstream code must sample densely enough that each
consecutive arc represents the physical boundary. Steps approaching 180
degrees bow strongly on the sphere, a step beyond 180 degrees selects the
opposite shorter arc, and an exactly ambiguous segment is rejected. Polypix
cannot distinguish intentional minor-arc geometry from an undersampled
trajectory.

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

Candidates are standard RING indices at the requested resolution and have set
semantics. The native kernel tests their centers directly; it does not first
materialize complete global coverage. Coverage uses a `1e-14` dot-product
boundary tolerance. Candidate filtering and complete scans may evaluate a
center through mathematically equivalent floating-point paths, so only centers
inside that tolerance band can be strategy-sensitive.

Complete scans use one conservative longitude bound for each footprint. This
is fast for the primary workload of small footprints and short strip segments,
but work follows the spherical bounding box rather than output size. Large
diagonal or pole-containing footprints can therefore cost substantially more
per returned cell. Per-ring edge intersections remain deliberately deferred
until such footprints become a measured primary workload.

## Parallel Execution

Large batches can run across native worker threads while the Python GIL is
released:

```python
sequential = px.cover_footprint(batch, resolution=9, threads=1)
automatic = px.cover_footprint(batch, resolution=9)
```

`threads=None` selects the automatic policy. A positive integer sets the
reusable worker-pool maximum, capped by the host. Calls below the measured
parallel crossover remain sequential without initializing a pool. Results are
identical across thread settings on the same build and platform.
