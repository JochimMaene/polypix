# API

Import Polypix as:

```python
import polypix as px
```

## cover

```python
cover(polygon, resolution) -> np.ndarray
cover(multi_polygon, resolution) -> tuple[np.ndarray, np.ndarray]
```

Return packed Polypix cell IDs for convex spherical polygons.

`cover()` accepts only `Polygon` and `MultiPolygon` objects. This keeps
coordinate-system choices explicit:

```python
polygon = px.Polygon.from_lonlat(vertices)
cell_ids = px.cover(polygon, resolution=8)
```

`Polygon.from_lonlat(...)` expects vertices as `(longitude, latitude)` pairs in
degrees. `Polygon.from_xyz(...)` expects normalized unit vectors as `(x, y, z)`.
A repeated final vertex is accepted and removed.

For batches, use `MultiPolygon`:

```python
polygons = px.MultiPolygon.from_lonlat([polygon_a, polygon_b])
cell_ids, counts = px.cover(polygons, resolution=8)
```

`MultiPolygon` accepts:

- dense batches: `(polygons, vertices, 2)` or `(polygons, vertices, 3)` arrays,
- ragged batches: lists of `(vertices, 2)` or `(vertices, 3)` arrays.
- explicit flat arrays with offsets through `MultiPolygon(vertices, offsets, coordinates)`.

The batch return value is:

- `cell_ids`: flat `uint64` array of all covered cells,
- `counts`: integer array with one covered-cell count per polygon.

Split the flat cells by polygon with:

```python
cell_ids_by_polygon = np.split(cell_ids, np.cumsum(counts[:-1]))
```

For an empty batch, both arrays are empty.

## center

```python
center(cell_ids) -> tuple[float, float] | np.ndarray
```

Return cell centers as `(longitude, latitude)` in degrees. A scalar cell ID
returns one tuple; a one-dimensional array returns an `(n, 2)` array. The input
may contain cells from different resolutions.

## boundary

```python
boundary(cell_ids) -> np.ndarray
```

Return cell boundaries as longitude/latitude degrees. A scalar cell ID returns
an array with shape `(4, 2)`; a one-dimensional array returns `(n, 4, 2)`.
