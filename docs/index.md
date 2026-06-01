# Polypix

Polypix computes HEALPix cells whose **cell centers** fall inside convex
spherical polygons.

It is designed for batch workloads:

- polygon vertices are passed through explicit `Polygon` or `MultiPolygon` objects,
- expensive spherical geometry and HEALPix work runs in C++,
- Python exposes a small API with NumPy results.

## What It Returns

For one polygon, `cover()` returns one array:

- `cell_ids`: `uint64` array of covered cells.

For a batch, `cover()` returns two arrays:

- `cell_ids`: flat `uint64` array of all covered cells,
- `counts`: integer array with one covered-cell count per polygon.

## Single Polygon

```python
import numpy as np
import polypix as px

polygon = np.array(
    [
        [-5.0, -5.0],
        [12.0, -4.0],
        [10.0, 9.0],
        [-6.0, 7.0],
    ],
    dtype=np.float64,
)

cell_ids = px.cover(px.Polygon.from_lonlat(polygon), resolution=2)
print(cell_ids)
```

## Multiple Polygons

```python
polygons = np.stack([polygon_a, polygon_b])

cell_ids, counts = px.cover(px.MultiPolygon.from_xyz(polygons), resolution=2)

cell_ids_by_polygon = np.split(cell_ids, np.cumsum(counts[:-1]))
first_polygon_cell_ids = cell_ids_by_polygon[0]
```

## Unit Vectors

```python
polygon = np.array(
    [
        [0.99240388, -0.08682409, -0.08715574],
        [0.97687319, 0.20763806, -0.06975647],
        [0.97014738, 0.17108787, 0.15643447],
        [0.98768834, -0.10452846, 0.12186934],
    ],
    dtype=np.float64,
)

cell_ids = px.cover(px.Polygon.from_xyz(polygon), resolution=2)
```

## Scope

Polypix assumes convex spherical polygons already provided in query order. The
public API is optimized for throughput rather than defensive input repair.
`Polygon` and `MultiPolygon` follow familiar geospatial naming, but they are not
general planar Shapely geometries: Polypix does not support holes or non-convex
polygons.
