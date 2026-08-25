# API reference


```{eval-rst}
.. currentmodule:: polypix

.. autosummary::
   :nosignatures:

   cover_convex_polygon
   cover_cap
   cover_sweep
   cell_at
   cell_centers
   cell_corners
   cell_count
   revisit
   Coverage
   RevisitStats
   Count
   Sum
```

## Covering regions

```{eval-rst}
.. autofunction:: cover_convex_polygon

.. autofunction:: cover_cap

.. autofunction:: cover_sweep
```

## Working with cells

```{eval-rst}
.. autofunction:: cell_at

.. autofunction:: cell_centers

.. autofunction:: cell_corners

.. autofunction:: cell_count
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

A polygon is given as vertices in boundary order. It must be convex and fit
inside an open hemisphere. Adjacent vertices are joined by the shorter of the two
great-circle arcs between them, so longitudes -179° and 179° are two degrees
apart, and nothing as large as a hemisphere can be described this way.

Accepted:

- either vertex orientation;
- a repeated closing vertex;
- redundant vertices on an existing edge, within floating-point precision;
- a cell center lying exactly on an edge, which counts as covered.

Rejected:

- fewer than three distinct vertices;
- duplicate, antipodal, or non-finite vertices;
- degenerate edges, self-intersections, and concave geometry;
- exact-hemisphere boundaries, and anything else detectably ambiguous.

Detectably is doing real work in that last item. Vertices that are individually
valid but were meant to describe the other side of the sphere look like any other
polygon from here.

### Numerical limits

Footprints below roughly 1e-8 radians across may be rejected. Where that starts depends on how the vertices are laid out, and at the same scale concavity becomes indistinguishable from a collinear edge.

Validation compares vertex pairs and tests every edge against every vertex, so
its cost grows with the square of the vertex count. Hand a densely sampled
boundary to `cover_sweep()` in short segments instead of passing one polygon with
hundreds of vertices.
