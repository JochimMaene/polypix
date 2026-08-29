# Getting started

Polypix turns directions and spherical regions into [HEALPix](https://healpix.sourceforge.io/) cell IDs. Install it with:

```bash
pip install polypix
```

NumPy is the only runtime dependency. See [Installation](install.md) if you
need to build from source.

## Directions in, cell IDs out

Numeric inputs are Cartesian directions shaped like `(..., 3)`. They don't have
to be unit length; Polypix ignores their magnitude. The coordinate frame is
also yours to choose. Polypix doesn't attach a frame or transform coordinates.

For example, these helpers convert between longitude/latitude in degrees and
Cartesian directions:

```python
import numpy as np
import polypix as px

def unit_vector(lon_deg, lat_deg):
    lon, lat = np.radians(lon_deg), np.radians(lat_deg)
    return np.stack((
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ), axis=-1)

def to_lonlat(vectors):
    lon = np.degrees(np.arctan2(vectors[..., 1], vectors[..., 0]))
    lat = np.degrees(np.arcsin(np.clip(vectors[..., 2], -1, 1)))
    return np.stack((lon, lat), axis=-1)
```

`cover_polygon()` also accepts GeoJSON-like mappings and objects with a
`__geo_interface__`. Those coordinates are longitude and latitude in decimal
degrees. Reproject them yourself first; the interface carries no reliable CRS.

## Cover circles

Pass a center direction and an angular radius in radians to `cover_cap()`:

```python
centers = unit_vector([-7.5, 8.0], [3.0, -4.0])
radii = np.radians([6.0, 4.0])

coverage = px.cover_cap(centers, radii, resolution=8)
```

```{figure} assets/generated/cover-cap.svg
:alt: Two spherical caps and the grid cells each one covers.
:width: 100%
:align: center

Two spherical caps and the cells `cover_cap()` returns.
```

The result contains one segment per input circle. `len(coverage)` is the number
of input regions, and `coverage[i]` is the cell-ID array for region `i`.

```python
len(coverage)       # 2
coverage[0]         # cells covered by the first circle
coverage.offsets    # boundaries of all segments in coverage.cells
```

## Cover polygons

Give `cover_polygon()` the vertices in boundary order. Consecutive vertices are
joined by the shorter great-circle arc:

```python
scene = unit_vector(
    [-9.0, 7.0, 11.0, -2.0],
    [-6.0, -8.0, 4.0, 8.0],
)
coverage = px.cover_polygon(scene, resolution=8)
```

```{figure} assets/generated/cover-convex-polygon.svg
:alt: A convex polygon and the grid cells it covers.
:width: 100%
:align: center

A polygon footprint and the cells it covers.
```

For several polygons, pass a sequence or a dense array shaped
`(regions, vertices, 3)`. A `Polygon` can include holes, and a `MultiPolygon`
can contain separate components. Each input region still produces one segment.

If the region is a long, thin swath, use `cover_sweep()` instead. Give it the
sampled left and right edges; each pair of consecutive samples becomes one
quadrilateral:

```python
track_lon = np.linspace(-13.0, 13.0, 7)
track_lat = 5 * np.sin(np.radians(track_lon * 7))

left = unit_vector(track_lon, track_lat + 3.2)
right = unit_vector(track_lon, track_lat - 3.2)
swath = px.cover_sweep(left, right, resolution=8)
```

```{figure} assets/generated/cover-sweep.svg
:alt: Two sampled edges, the quadrilaterals between consecutive samples, and the cells they cover.
:width: 100%
:align: center

Each pair of samples becomes one quadrilateral and one result segment.
```

Sample densely enough that the great-circle arcs between samples follow the
swath you mean.

## Center or overlap?

By default, a cell is selected when its center is inside the region:

```python
center_coverage = px.cover_cap(center, radius, resolution=8)
```

This is fast, but a region smaller than a cell can return no cells. Use
`mode="overlap"` when every touched cell must be included:

```python
overlap_coverage = px.cover_cap(
    center,
    radius,
    resolution=8,
    mode="overlap",
)
```

Overlap mode costs more, especially for detailed polygons. The same option is
available on `cover_polygon()` and `cover_sweep()`. See [How it works](concepts.md)
for the exact rules.

## Assign points to cells

For pointings, sample points, or a ground track, use `cell_at()`:

```python
points = unit_vector([-11.0, -3.0, 5.0, 12.0], [6.0, -7.0, 2.0, -9.0])
cells = px.cell_at(points, resolution=8)
```

`cell_centers()` gives the representative direction for those cells. It does
not recover the original directions:

```python
centers = px.cell_centers(cells, resolution=8)
lonlat = to_lonlat(centers)
```

```{figure} assets/generated/cell-at.svg
:alt: Four directions, each snapped to the grid cell containing it, with an arrow to that cell's center.
:width: 100%
:align: center

`cell_at()` returns the cell; `cell_centers()` returns its representative center.
```

Use `cell_neighbors()` when you need the cells touching a set of cells at an
edge or corner. The input cells themselves aren't included:

```python
neighbors = px.cell_neighbors(cells, resolution=8)
```

## Count or sum over cells

`Coverage` keeps the cell IDs in one flat `cells` array and uses `offsets` to
separate the input regions. Reduce it when you want one value per cell:

```python
counts = coverage.reduce(px.Count())
```

Because HEALPix cells have equal area, these counts can be compared directly.
For weights or durations, use `Sum`:

```python
exposure = coverage.reduce(px.Sum(values))
```

If you only need counts, ask for them during covering. This avoids creating the
individual region-cell hits:

```python
counts = px.cover_cap(
    centers,
    radii,
    resolution=8,
    reduce=px.Count(),
)
```

```{figure} assets/generated/earth-observation-count.png
:alt: Global map of overflight counts, banded by latitude, from three Sentinel-2 spacecraft over 14 days.
:width: 100%
:align: center

Cell counts can be plotted directly as a global map.
```

For a small set of cells at a high resolution, add `candidate_cells`:

```python
site_counts = px.cover_cap(
    centers,
    radii,
    resolution=20,
    candidate_cells=site_cells,
    reduce=px.Count(),
)
```

See [Performance](performance.md) for resolution, memory, batching, and thread
choices.

## Revisit statistics

When coverage segments are consecutive time bins, `revisit()` summarizes when
each cell was occupied:

```python
stats = px.revisit(swath)
```

The bins are ordinal. Map `stats.first_start`, `stats.last_stop`, and the gap
fields to your own timestamps. Matching segment indices must describe matching
time boundaries; Polypix has no clock to check that for you.