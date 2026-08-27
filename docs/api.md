# API reference


```{eval-rst}
.. currentmodule:: polypix

.. autosummary::
   :nosignatures:

   cover_polygon
   Polygon
   MultiPolygon
   cover_cap
   cover_sweep
   cell_at
   cell_centers
   cell_corners
   cell_count
   cell_neighbors
   revisit
   Coverage
   RevisitStats
   Count
   Sum
```

## Covering regions

```{eval-rst}
.. autofunction:: cover_polygon

.. autoclass:: Polygon

.. autoclass:: MultiPolygon
   :members: __len__, __iter__

.. autofunction:: cover_cap

.. autofunction:: cover_sweep
```

## Working with cells

```{eval-rst}
.. autofunction:: cell_at

.. autofunction:: cell_centers

.. autofunction:: cell_corners

.. autofunction:: cell_count

.. autofunction:: cell_neighbors
```

## Reducing and summarizing

```{eval-rst}
.. autoclass:: Count

.. autoclass:: Sum

.. autofunction:: revisit
```

## Results

```{eval-rst}
.. autoclass:: Coverage
   :members: from_arrays, reduce, __len__, __getitem__
   :special-members: __len__, __getitem__

.. autoclass:: RevisitStats
   :members: __len__
   :special-members: __len__
```

(geometry-contract)=
## Geometry contract

A polygon is given as vertices in boundary order. It may be convex or concave,
but each component must fit inside an open hemisphere. Adjacent vertices are
joined by the shorter of the two great-circle arcs between them, so longitudes
-179° and 179° are two degrees apart, and nothing as large as a hemisphere can
be described this way.

Accepted:

- either vertex orientation;
- a repeated closing vertex;
- redundant vertices on an existing edge, within floating-point precision;
- explicit holes and multipart unions through `Polygon` and `MultiPolygon`;
- a cell center lying exactly on an edge, which counts as covered.

Rejected:

- fewer than three distinct vertices;
- duplicate, antipodal, or non-finite vertices;
- degenerate edges and self-intersections;
- holes outside, touching, or crossing their outer ring or another hole;
- exact-hemisphere boundaries, and anything else detectably ambiguous.

Detectably is doing real work in that last item. Vertices that are individually
valid but were meant to describe the other side of the sphere look like any other
polygon from here.

### Geographic interface inputs

`cover_polygon()` accepts a mapping directly or an object whose
`__geo_interface__` property returns one. The mapping may describe a `Polygon`,
`MultiPolygon`, or one `Feature` containing either. The first polygon ring is
the outer boundary and the rest are holes; each multipolygon component becomes
one `Polygon` in a single union region.

Positions are `(longitude, latitude)` or `(longitude, latitude, altitude)` in
decimal degrees, interpreted directly as angles on a unit sphere. The datum and
frame belong to the caller, and altitude is ignored. Longitudes must lie in
`[-180, 180]` and latitudes in `[-90, 90]`; Polypix neither reads nor transforms
a CRS. Empty polygonal geometry and a Feature with null geometry produce one
empty segment. Feature properties, IDs, bounding boxes, and foreign members are
ignored.

Points, lines, geometry collections, and feature collections are rejected.
Homogeneous sequences form batches, but geographic mappings, Cartesian arrays,
and Polypix geometry objects cannot be mixed in one batch. These inputs import
vertices and ring topology, not GeoJSON edge interpolation: the shorter
great-circle rule above remains authoritative. Reproject, repair, simplify, or
densify with a GIS library before the call when needed.

A whole-world longitude/latitude bounding box is not a spherical polygon:
`-180` and `180` are the same meridian, and every longitude meets at each pole.

### Numerical limits

Footprints below roughly 1e-8 radians across may be rejected. Where that starts depends on how the vertices are laid out, and at the same scale concavity becomes indistinguishable from a collinear edge.

Validation compares vertex pairs and tests every edge against every vertex, so
its cost grows with the square of the vertex count. Hand a densely sampled
boundary to `cover_sweep()` in short segments instead of passing one polygon with
hundreds of vertices.
