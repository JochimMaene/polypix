# ADR-0004: Accept polygonal Python geo-interface input

- Status: Accepted
- Date: 2026-08-27
- Amends: ADR-0003

## Context

Python GIS geometry objects commonly expose a `__geo_interface__` mapping.
Polypix already represents concave polygons, holes, and multipart unions, but
callers have to unpack those mappings and convert longitude/latitude vertices to
Cartesian directions themselves.

The interface does not reliably carry a CRS, and its planar edge interpolation
does not match Polypix's shorter great-circle edges. Depending on a GIS runtime
would not resolve either ambiguity and would break the NumPy-only runtime model.

## Decision

`cover_polygon()` accepts mappings and objects exposing `__geo_interface__` for
`Polygon`, `MultiPolygon`, and one polygonal `Feature`. Coordinates are
longitude and latitude in decimal degrees, used directly as angles on a unit
sphere in the caller's datum and frame. Optional altitude is ignored. Coordinates
are converted to Cartesian directions in the Python boundary; the existing
spherical validation, open-hemisphere restriction, and great-circle edges then
apply unchanged.

Empty polygonal geometry and a Feature with null geometry produce one empty
region. Metadata is ignored. Non-polygonal geometry, geometry collections,
feature collections, and mixed-representation batches are rejected. Projected
coordinates are outside the contract; range validation catches obvious cases,
but Polypix does not inspect CRS metadata, reproject, repair, simplify, densify,
or expose `__geo_interface__` from its arbitrary-frame XYZ geometry classes.

## Consequences

Shapely, GeoPandas, Fiona, and similar objects can reach polygon coverage
without becoming runtime dependencies. Callers remain responsible for
reprojection and for approximating planar source edges when great-circle edges
are not suitable. Detailed boundaries retain the existing quadratic validation
cost and should be simplified upstream.
