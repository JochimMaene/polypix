# Getting started

Every block on this page runs. Start a Python session, paste the setup below,
and each following block continues from the one before it.

```bash
pip install polypix
```

NumPy is the only runtime dependency, and there are wheels for CPython 3.12+ on
Linux, macOS, and Windows. [Installation](install.md) covers source builds.

## What you call

Four common geometry operations cover most uses:

| You have | Call | You get back |
| --- | --- | --- |
| Visibility circles, elevation-mask footprints, instantaneous fields of view | `cover_cap()` | the cells inside each circle |
| Scenes, frames, convex sensor footprints | `cover_convex_polygon()` | the cells inside each polygon |
| The swath a sensor paints as it moves | `cover_sweep()` | the cells under each interval of the swath |
| Individual pointings, ground tracks, sample points | `cell_at()` | the one cell each direction falls in |

All four take batches. Angular arguments, where present, are in **radians**.

## Setup

Everything you pass in is a Cartesian direction `(x, y, z)`. Magnitude is
ignored, so position vectors work just as well as unit vectors, and the frame is
whichever one you are already working in: Earth-fixed for a ground footprint,
celestial for a sky survey. Polypix never labels or transforms it.

If your data is longitude and latitude, convert it once on the way in, and
convert cell centers back on the way out. These two helpers are all you need,
and the rest of the page uses them throughout:

```{doctest}
>>> import numpy as np
>>> import polypix as px

>>> def unit_vector(lon_deg, lat_deg):
...     """Cartesian directions for longitudes and latitudes in degrees."""
...     lon, lat = np.radians(lon_deg), np.radians(lat_deg)
...     return np.stack(
...         [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
...         axis=-1,
...     )

>>> def to_lonlat(vectors):
...     """Degrees longitude and latitude for unit vectors."""
...     longitude = np.degrees(np.arctan2(vectors[..., 1], vectors[..., 0]))
...     latitude = np.degrees(np.arcsin(np.clip(vectors[..., 2], -1.0, 1.0)))
...     return np.stack((longitude, latitude), axis=-1)
```

[Direction geometry](concepts.md#direction-geometry) explains why these
conversions stay outside Polypix.

The examples below all work at resolution 4, where a cell is about 400 km
across. That is coarse enough that you can count the cells in the pictures and
check them against the numbers.

## Cover visibility circles

A spherical cap is a center direction plus an angular radius in radians. This is
the shape of a ground station's view of a satellite above an elevation mask, of
a satellite's own service circle, and of any instantaneous circular field of
view.

Here are two caps of different sizes, given in degrees and converted:

```{doctest}
>>> cap_lon, cap_lat = [-7.5, 8.0], [3.0, -4.0]
>>> cap_radius_deg = [6.0, 4.0]

>>> cap_coverage = px.cover_cap(
...     unit_vector(cap_lon, cap_lat), np.radians(cap_radius_deg), resolution=4
... )
>>> cap_coverage.segment_sizes
array([9, 4])
```

Both caps came back in one result. Indexing it returns a read-only view for one
region:

```{doctest}
>>> cap_coverage[0]
array([1374, 1375, 1438, 1439, 1502, 1503, 1566, 1567, 1631])
```

Every region operation follows the same rule: a cell is selected when its center
lies inside the region.

```{figure} assets/generated/cover-cap.svg
:alt: Two spherical caps and the grid cells each one covers.
:width: 100%
:align: center

The two caps above, and the cells `cover_cap()` returned for them.
```

## Cover scenes and sensor footprints

A convex polygon is the shape of an imaging scene, a detector frame projected on
the ground, or any convex area of interest. Give the vertices in boundary order;
adjacent vertices are joined by the shorter great-circle arc:

```{doctest}
>>> scene_lon = [-9.0, 7.0, 11.0, -2.0]
>>> scene_lat = [-6.0, -8.0, 4.0, 8.0]

>>> scene_coverage = px.cover_convex_polygon(
...     unit_vector(scene_lon, scene_lat), resolution=4
... )
>>> scene_coverage.segment_sizes
array([16])
>>> scene_coverage[0][:6]
array([1312, 1376, 1377, 1439, 1440, 1441])
```

One polygon produces one segment in the result.

```{figure} assets/generated/cover-convex-polygon.svg
:alt: A convex polygon and the grid cells it covers.
:width: 100%
:align: center

A four-sided footprint and the cells it covers.
```

If every polygon has the same vertex count, a dense `(regions, vertices, 3)`
array is the fastest thing to hand over. For a ragged batch, pass a sequence of
arrays instead.

## Cover a swath

A moving sensor sweeps out a swath. Sample its left and right edges at whatever
cadence your propagator gives you, and `cover_sweep()` treats every consecutive
pair of samples as one quadrilateral of swath. Seven samples along a track, with
edges 3.2° either side of it:

```{doctest}
>>> track_lon = np.linspace(-13.0, 13.0, 7)
>>> track_lat = 5.0 * np.sin(np.radians(track_lon * 7.0))

>>> left_edge = unit_vector(track_lon, track_lat + 3.2)
>>> right_edge = unit_vector(track_lon, track_lat - 3.2)
>>> swath_coverage = px.cover_sweep(left_edge, right_edge, resolution=4)
>>> len(swath_coverage)
6
>>> swath_coverage.segment_sizes
array([2, 2, 4, 4, 2, 2])
```

Seven samples give six intervals, with one result segment per interval:

```{doctest}
>>> swath_coverage[3]
array([1376, 1440, 1504, 1568])
```

```{figure} assets/generated/cover-sweep.svg
:alt: Two sampled edges, the quadrilaterals between consecutive samples, and the cells they cover.
:width: 100%
:align: center

Each pair of consecutive samples becomes one quadrilateral, and each quadrilateral is its own segment in the result.
```

Polypix joins adjacent samples with minor great-circle arcs. Sample densely
enough that those arcs represent the intended swath edge.

## Assign pointings to cells

When your input is points rather than regions — a ground track, a set of target
coordinates, individual pointings — use `cell_at()`:

```{doctest}
>>> point_lon = [-11.0, -3.0, 5.0, 12.0]
>>> point_lat = [6.0, -7.0, 2.0, -9.0]

>>> point_cells = px.cell_at(unit_vector(point_lon, point_lat), resolution=4)
>>> point_cells
array([1374, 1695, 1441, 1762])
```

`cell_centers()` goes back the other way, but watch what "back" means. These are the
cells' own centers, not the directions you started with:

```{doctest}
>>> point_centers = px.cell_centers(point_cells, resolution=4)
>>> np.round(to_lonlat(point_centers), 2)
array([[-11.25,   7.18],
       [ -2.81,  -4.78],
       [  5.62,   2.39],
       [ 14.06,  -9.59]])
```

Only cell centers round-trip exactly:

```{doctest}
>>> px.cell_at(point_centers, resolution=4)
array([1374, 1695, 1441, 1762])
```

```{figure} assets/generated/cell-at.svg
:alt: Four directions, each snapped to the grid cell containing it, with an arrow to that cell's center.
:width: 100%
:align: center

`cell_at()` gives you the cell a direction falls in. `cell_centers()` then gives that cell's center, which is the arrow head, not where you started.
```

## Turn cells into a map

Cell IDs are integers in `[0, cell_count(resolution))`. Reduce the segmented
membership from any geometry into a global map with one operation:

```{doctest}
>>> resolution = 4
>>> coverage = cap_coverage
>>> hits = coverage.reduce(px.Count())
>>> hits.shape
(3072,)
```

`hits[i]` is the number of regions that selected cell `i`. Convert occupied
cell centers to longitude and latitude before plotting:

```{doctest}
>>> occupied = np.flatnonzero(hits)
>>> lonlat = to_lonlat(px.cell_centers(occupied, resolution))
>>> lonlat.shape == (occupied.size, 2)
True
```

```{figure} assets/generated/earth-observation-count.png
:alt: Global map of observation counts, banded by latitude, from ten satellites over ten days.
:width: 100%
:align: center

At mission scale, the same reduction counts 144,000 swath intervals from ten
satellites. [How often does a satellite fly
over?](examples/earth-observation-constellation.md) builds the analysis end to
end.
```

Equal-area cells make these counts directly comparable without area weighting.

Use `coverage.reduce(px.Sum(values))` when each segment contributes an
exposure, duration, probability, or capacity instead of one count. The native
reducer reads `Coverage.offsets` directly rather than repeating one value per
hit.

## Preserve occupied-bin runs

For revisit analysis, keep complete ordinal runs and apply time afterward:

```{doctest}
>>> runs = px.occupancy(swath_coverage)
>>> bool(np.all(runs.starts < runs.stops))
True
>>> time_edges_s = np.arange(swath_coverage.segment_count + 1) * 60
>>> run_starts_s = time_edges_s[runs.starts]
>>> run_stops_s = time_edges_s[runs.stops]
```

With multiple aligned source coverages, pass the sequence together. Matching
segment indices must have identical time boundaries, and consecutive bins must
be temporally adjacent. Set `minimum_sources=2` for simultaneous two-entry
occupancy; source uniqueness is your responsibility. Polypix leaves maximum,
percentile, leading/trailing, and cyclic gap definitions to the analysis layer
because they are policy choices, not geometry. For sweeps, the result is an
occupied-bin approximation whose physical boundary precision is limited by the
sampling cadence.

When runs are only a step on the way to per-cell revisit numbers, skip
building them:

```{doctest}
>>> stats = px.occupancy(swath_coverage, reduce=px.Stats())
>>> observed_once = stats.run_counts == 1
>>> bool(np.all(stats.maximum_internal_gap_steps[observed_once] == 0))
True
>>> worst_gap_s = stats.maximum_internal_gap_steps * 60
>>> bool(np.all(stats.first_start < stats.last_stop))
True
```

Every field describes the same thresholded axis, so `run_counts`,
`internal_gap_steps_sum`, and `maximum_internal_gap_steps` can be combined
directly. `first_start` and `last_stop` let you add horizon-edge gaps yourself.

## Count overlaps without building membership

If you only need cap counts per cell, ask for them directly and Polypix never
builds the region–cell pairs:

```{doctest}
>>> counts = px.cover_cap(
...     unit_vector(cap_lon, cap_lat),
...     np.radians(cap_radius_deg),
...     resolution=4,
...     reduce=px.Count(),
... )
>>> counts.shape
(3072,)
```

At high resolution, query only the cells you need:

```{doctest}
>>> site_cells = point_cells[:2]
>>> site_counts = px.cover_cap(
...     unit_vector(cap_lon, cap_lat),
...     np.radians(cap_radius_deg),
...     resolution=4,
...     reduce=px.Count(cells=site_cells),
... )
>>> site_counts.shape
(2,)
```

## When a region comes back empty

A cell is covered when its **center** falls inside your region. A region smaller
or thinner than a cell can therefore return an empty segment. Raise `resolution`
until cells are smaller than the smallest feature you need to resolve.
[Resolutions](resolutions.md#picking-one) has the sizes, and [center-sampled
coverage](concepts.md#center-sampled-coverage) shows exactly what the rule
includes and excludes.

## Where to go next

- [How it works](concepts.md) explains resolution, center sampling, segmented
  results, occupancy runs, and handing cell IDs to other HEALPix libraries.
- [Performance and memory](performance.md) covers sizing results, sparse
  queries, batching, and threads.
- [API reference](api.md) is the complete call contract.
