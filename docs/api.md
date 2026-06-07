# API Reference

Import Polypix as:

```python
import polypix as px
```

The public API is intentionally small:

- `Coverage`
- `cover_footprint`
- `cover_swath`
- `centers`
- `boundaries`

## Coverage

```python
px.Coverage(cell_ids, offsets)
```

Coverage results are returned by `cover_footprint()` and `cover_swath()`.

Attributes:

- `cell_ids`: a one-dimensional `np.ndarray` with dtype `uint64`.
- `offsets`: a one-dimensional `np.ndarray` with dtype `uint64`.
- `counts`: a derived one-dimensional `np.ndarray` with dtype `intp`.

For `N` input footprints, `offsets` has length `N + 1`. The covered cells for
footprint `i` are:

```python
coverage.cell_ids[coverage.offsets[i] : coverage.offsets[i + 1]]
```

For a single footprint, `offsets` is `[0, len(cell_ids)]`.

## cover_footprint

```python
px.cover_footprint(footprints_xyz, resolution)
```

Returns packed Polypix cell IDs for one convex spherical footprint or a dense
batch of footprints. Footprints are normalized `(x, y, z)` unit vectors in a
common frame.

Parameters:

- `footprints_xyz`: unit-vector vertices with shape `(vertices, 3)` or
  `(footprints, vertices, 3)`.
- `resolution`: integer HEALPix resolution from 0 through 29.

Returns:

- `Coverage`.

Example:

```python
import math

import numpy as np
import polypix as px


def lonlat_to_xyz(lon_deg, lat_deg):
    lon = math.radians(lon_deg)
    lat = math.radians(lat_deg)
    cos_lat = math.cos(lat)
    return cos_lat * math.cos(lon), cos_lat * math.sin(lon), math.sin(lat)


footprint = np.asarray(
    [
        lonlat_to_xyz(-5.0, -5.0),
        lonlat_to_xyz(12.0, -4.0),
        lonlat_to_xyz(10.0, 9.0),
        lonlat_to_xyz(-6.0, 7.0),
    ],
    dtype=np.float64,
)

coverage = px.cover_footprint(footprint, resolution=8)
```

Batch example:

```python
footprint_a = np.asarray(
    [
        lonlat_to_xyz(-5.0, -5.0),
        lonlat_to_xyz(12.0, -4.0),
        lonlat_to_xyz(10.0, 9.0),
        lonlat_to_xyz(-6.0, 7.0),
    ],
    dtype=np.float64,
)
footprint_b = np.asarray(
    [
        lonlat_to_xyz(20.0, -10.0),
        lonlat_to_xyz(33.0, -10.0),
        lonlat_to_xyz(33.0, 0.0),
        lonlat_to_xyz(20.0, 0.0),
    ],
    dtype=np.float64,
)

coverage = px.cover_footprint(np.stack([footprint_a, footprint_b]), resolution=8)
cells_by_footprint = [
    coverage.cell_ids[start:stop]
    for start, stop in zip(coverage.offsets[:-1], coverage.offsets[1:])
]
```

## cover_swath

```python
px.cover_swath(left_edge_xyz, right_edge_xyz, resolution)
```

Returns packed Polypix cell IDs for consecutive swath intervals.

Parameters:

- `left_edge_xyz`: unit-vector left edge samples with shape `(samples, 3)`.
- `right_edge_xyz`: unit-vector right edge samples with shape `(samples, 3)`.
- `resolution`: integer HEALPix resolution from 0 through 29.

Returns:

- `Coverage`.

`left_edge_xyz` and `right_edge_xyz` must contain the same number of samples,
and at least two samples. Each consecutive interval is covered as one
quadrilateral:

```text
[left[i], right[i], right[i + 1], left[i + 1]]
```

Example:

```python
coverage = px.cover_swath(left_edge_xyz, right_edge_xyz, resolution=8)
cells_by_interval = [
    coverage.cell_ids[start:stop]
    for start, stop in zip(coverage.offsets[:-1], coverage.offsets[1:])
]
```

## centers

```python
px.centers(cell_ids)
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
lon, lat = px.centers(int(coverage.cell_ids[0]))
center_lonlat = px.centers(coverage.cell_ids)
```

The input may contain cells from different resolutions.

## boundaries

```python
px.boundaries(cell_ids)
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
outline = px.boundaries(int(coverage.cell_ids[0]))
outlines = px.boundaries(coverage.cell_ids[:10])
```

The four returned vertices are the HEALPix cell corners. The boundary is not
closed by repeating the first vertex.

## Exceptions And Validation

Polypix validates Python array shape before entering the native coverage code.
The native code validates spherical geometry.

Common validation errors include:

- `resolution` is not an integer or is outside `0..29`;
- footprint arrays do not have shape `(vertices, 3)` or `(footprints, vertices, 3)`;
- swath edge arrays do not have shape `(samples, 3)`;
- swath edge arrays have different lengths or fewer than two samples;
- unit vectors are not finite or not normalized;
- a footprint has fewer than three unique vertices;
- a footprint has duplicate vertices, degenerate edges, or is non-convex;
- packed cell IDs passed to `centers()` or `boundaries()` are invalid.
