# API Reference

Import Polypix as:

```python
import polypix as px
```

The public API is intentionally small:

- `Polygon`
- `MultiPolygon`
- `cover`
- `center`
- `boundary`

## Polygon

```python
px.Polygon.from_lonlat(vertices)
px.Polygon.from_xyz(vertices)
```

Represents one convex spherical polygon.

`from_lonlat(...)` accepts an array-like object with shape `(vertices, 2)`.
Each row is `(longitude, latitude)` in degrees.

`from_xyz(...)` accepts an array-like object with shape `(vertices, 3)`. Each
row is a normalized unit vector `(x, y, z)`.

Example:

```python
vertices = np.array(
    [
        [-5.0, -5.0],
        [12.0, -4.0],
        [10.0, 9.0],
        [-6.0, 7.0],
    ],
    dtype=np.float64,
)

polygon = px.Polygon.from_lonlat(vertices)
```

The constructor stores vertices as contiguous `float64` arrays. It validates
array shape and coordinate-system names immediately. Geometry validation happens
when the polygon is passed to `cover()`.

## MultiPolygon

```python
px.MultiPolygon.from_lonlat(polygons)
px.MultiPolygon.from_xyz(polygons)
px.MultiPolygon(vertices, offsets, coordinates)
```

Represents a batch of convex spherical polygons.

The `from_lonlat(...)` and `from_xyz(...)` constructors accept either dense or
ragged input:

- a dense array with shape `(polygons, vertices, 2)` or `(polygons, vertices, 3)`,
- a ragged list of arrays with shape `(vertices, 2)` or `(vertices, 3)`,
- for a single polygon, an array with shape `(vertices, 2)` or `(vertices, 3)`.

Use the direct constructor when you already have flat vertices and offsets:

```python
vertices = np.array(
    [
        [-5.0, -5.0],
        [12.0, -4.0],
        [10.0, 9.0],
        [-6.0, 7.0],
        [20.0, -10.0],
        [33.0, -10.0],
        [33.0, 0.0],
        [20.0, 0.0],
    ],
    dtype=np.float64,
)

batch = px.MultiPolygon(vertices, offsets=[0, 4, 8], coordinates="lonlat")
```

`offsets` must be a one-dimensional integer array that starts at `0`, is
nondecreasing, and ends at the total vertex count.

`len(batch)` returns the number of polygons.

## cover

```python
px.cover(polygons, resolution)
```

Returns packed Polypix cell IDs for one polygon or a batch of polygons.

Parameters:

- `polygons`: a `Polygon` or `MultiPolygon`.
- `resolution`: integer HEALPix resolution from 0 through 29.

Returns:

- for `Polygon`, a one-dimensional `np.ndarray` with dtype `uint64`;
- for `MultiPolygon`, `(cell_ids, counts)`, where `cell_ids` is a flat
  one-dimensional `uint64` array and `counts` has one integer count per input
  polygon.

Example:

```python
cell_ids = px.cover(polygon, resolution=8)
```

Batch example:

```python
batch = px.MultiPolygon.from_lonlat(
    [
        np.array(
            [
                [-5.0, -5.0],
                [12.0, -4.0],
                [10.0, 9.0],
                [-6.0, 7.0],
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                [20.0, -10.0],
                [33.0, -10.0],
                [33.0, 0.0],
                [20.0, 0.0],
            ],
            dtype=np.float64,
        ),
    ]
)
cell_ids, counts = px.cover(batch, resolution=8)
cells_by_polygon = np.split(cell_ids, np.cumsum(counts[:-1]))
```

`cover()` raises `TypeError` when called with raw arrays instead of `Polygon` or
`MultiPolygon` objects. It raises `ValueError` for invalid resolutions or
invalid polygon geometry.

## center

```python
px.center(cell_ids)
```

Returns HEALPix cell centers as `(longitude, latitude)` in degrees.

Parameters:

- `cell_ids`: a scalar packed cell ID or a one-dimensional sequence of packed
  cell IDs.

Returns:

- for a scalar input, a `(longitude, latitude)` tuple;
- for a one-dimensional input, an array with shape `(n, 2)` and dtype
  `float64`.

Example:

```python
lon, lat = px.center(int(cell_ids[0]))
centers = px.center(cell_ids)
```

The input may contain cells from different resolutions.

## boundary

```python
px.boundary(cell_ids)
```

Returns HEALPix cell boundaries as longitude/latitude coordinates in degrees.

Parameters:

- `cell_ids`: a scalar packed cell ID or a one-dimensional sequence of packed
  cell IDs.

Returns:

- for a scalar input, an array with shape `(4, 2)`;
- for a one-dimensional input, an array with shape `(n, 4, 2)`.

Example:

```python
outline = px.boundary(int(cell_ids[0]))
outlines = px.boundary(cell_ids[:10])
```

The four returned vertices are the HEALPix cell corners. The boundary is not
closed by repeating the first vertex.

## Exceptions And Validation

Polypix validates Python array shape before entering the native coverage code.
The native code validates spherical geometry.

Common validation errors include:

- `resolution` is not an integer or is outside `0..29`;
- longitude/latitude arrays do not have shape `(vertices, 2)`;
- unit-vector arrays do not have shape `(vertices, 3)`;
- unit vectors are not finite or not normalized;
- a polygon has fewer than three unique vertices;
- a polygon has duplicate vertices, degenerate edges, or is non-convex;
- packed cell IDs passed to `center()` or `boundary()` are invalid.
