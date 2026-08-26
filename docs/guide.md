# Getting started

This page is one long session. Start a Python interpreter, paste in the setup
below, and each block that follows continues where the previous one left off.

```bash
pip install polypix
```

NumPy is the only runtime dependency, and there are wheels for CPython 3.12 and
newer on Linux, macOS, and Windows. [Installation](install.md) covers source
builds.

## What you call

Four operations cover most of what people do with the library:

| You have | Call | You get back |
| --- | --- | --- |
| Visibility circles, elevation-mask footprints, instantaneous fields of view | `cover_cap()` | the cells inside each circle |
| Scenes, frames, and areas of interest | `cover_polygon()` | the cells inside each polygonal region |
| The swath a sensor paints as it moves | `cover_sweep()` | the cells under each interval of the swath |
| Individual pointings, ground tracks, sample points | `cell_at()` | the one cell each direction falls in |

All four take batches, and every angular argument is in radians.

## Setup

Everything you pass in is a Cartesian direction `(x, y, z)`. Magnitudes are
ignored, so position vectors work as well as unit vectors, and the frame is
whichever one you are already working in: Earth-fixed for a ground footprint,
celestial for a sky survey. Polypix neither labels the frame nor transforms
between frames.

If your data is longitude and latitude, convert it once on the way in and
convert cell centers back on the way out. The two helpers below are all that
takes, and the rest of this page uses them throughout:

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

The examples all work at resolution 4, where a cell is about 400 km across. That
is coarse enough that you can count the cells in each picture and check them
against the numbers in the output.

## Cover visibility circles

A spherical cap is a center direction plus an angular radius in radians. It is
the shape of a ground station's view of a satellite above an elevation mask, of a
satellite's own service circle, and of any circular instantaneous field of view.

Here are two caps of different sizes, given in degrees and converted on the way
in:

```{doctest}
>>> cap_lon, cap_lat = [-7.5, 8.0], [3.0, -4.0]
>>> cap_radius_deg = [6.0, 4.0]

>>> cap_coverage = px.cover_cap(
...     unit_vector(cap_lon, cap_lat), np.radians(cap_radius_deg), resolution=4
... )
>>> len(cap_coverage)
2
>>> np.diff(cap_coverage.offsets)
array([9, 4])
```

Both caps came back in a single result. A result keeps every cell in one flat
`cells` array, with `offsets` recording where each input's cells begin and end.
That is why `len()` counts inputs and not cells, and why
`np.diff(offsets)` tells you how many cells each input covered. Indexing gives
you a read-only view of one of them:

```{doctest}
>>> cap_coverage[0]
array([1374, 1375, 1438, 1439, 1502, 1503, 1566, 1567, 1631])
```

All of the region operations follow the same rule: a cell is selected when its
center lies inside the region.

```{figure} assets/generated/cover-cap.svg
:alt: Two spherical caps and the grid cells each one covers.
:width: 100%
:align: center

The two caps above, and the cells `cover_cap()` returned for them.
```

## Cover scenes and sensor footprints

A polygon is the shape of an imaging scene, a detector frame projected on
the ground, or an area of interest. Give the vertices in boundary order.
Adjacent vertices are joined by the shorter of the two great-circle arcs between
them:

```{doctest}
>>> scene_lon = [-9.0, 7.0, 11.0, -2.0]
>>> scene_lat = [-6.0, -8.0, 4.0, 8.0]

>>> scene_coverage = px.cover_polygon(
...     unit_vector(scene_lon, scene_lat), resolution=4
... )
>>> np.diff(scene_coverage.offsets)
array([16])
>>> scene_coverage[0][:6]
array([1312, 1376, 1377, 1439, 1440, 1441])
```

One polygon produces one segment in the result.

Concave arrays work directly. Use `Polygon(outer, *holes)` when a component has
holes and `MultiPolygon(*polygons)` when one region has separate components.
The whole multipart region still produces one result segment, with overlaps
removed before reducers are applied.

That segment can be reused as an area-of-interest restriction:

```python
europe_cells = px.cover_polygon(europe, resolution=8).cells
coverage = px.cover_polygon(
    scenes, resolution=8, candidate_cells=europe_cells
)
```

```{figure} assets/generated/cover-convex-polygon.svg
:alt: A convex polygon and the grid cells it covers.
:width: 100%
:align: center

A four-sided footprint and the cells it covers.
```

If all of your polygons have the same vertex count, a dense
`(regions, vertices, 3)` array is the cheapest thing to hand over. If they do
not, pass a sequence of arrays instead.

## Cover a swath

A moving sensor sweeps out a swath. Sample its left and right edges at whatever
cadence your propagator gives you, and `cover_sweep()` reads every consecutive
pair of samples as one quadrilateral of swath. Here are seven samples along a
track, with edges 3.2° either side of it:

```{doctest}
>>> track_lon = np.linspace(-13.0, 13.0, 7)
>>> track_lat = 5.0 * np.sin(np.radians(track_lon * 7.0))

>>> left_edge = unit_vector(track_lon, track_lat + 3.2)
>>> right_edge = unit_vector(track_lon, track_lat - 3.2)
>>> swath_coverage = px.cover_sweep(left_edge, right_edge, resolution=4)
>>> len(swath_coverage)
6
>>> np.diff(swath_coverage.offsets)
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

Adjacent samples are joined with the shorter great-circle arc, so sample
densely enough that those arcs really do describe the swath edge you had in
mind.

## Assign pointings to cells

When your input is points rather than regions (a ground track, a set of target
coordinates, individual pointings) use `cell_at()`:

```{doctest}
>>> point_lon = [-11.0, -3.0, 5.0, 12.0]
>>> point_lat = [6.0, -7.0, 2.0, -9.0]

>>> point_cells = px.cell_at(unit_vector(point_lon, point_lat), resolution=4)
>>> point_cells
array([1374, 1695, 1441, 1762])
```

`cell_centers()` goes back the other way, though it is worth being careful about
what "back" means here. What you get are the cells' own centers, not the
directions you started from:

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

Cell IDs are integers in `[0, cell_count(resolution))`, so a global map is just
an array of that length. One operation reduces the segmented membership from any
geometry into one:

```{doctest}
>>> resolution = 4
>>> coverage = cap_coverage
>>> hits = coverage.reduce(px.Count())
>>> hits.shape
(3072,)
```

Now `hits[i]` is the number of regions that selected cell `i`. Convert the
occupied cell centers to longitude and latitude before plotting:

```{doctest}
>>> occupied = np.flatnonzero(hits)
>>> lonlat = to_lonlat(px.cell_centers(occupied, resolution))
>>> lonlat.shape == (occupied.size, 2)
True
```

```{figure} assets/generated/earth-observation-count.png
:alt: Global map of overflight counts, banded by latitude, from three Sentinel-2 spacecraft over 14 days.
:width: 100%
:align: center

At mission scale, the same reduction counts 60,480 swath intervals from the
three Sentinel-2 spacecraft. [What is Sentinel-2's revisit
time?](examples/earth-observation-constellation.md) builds the analysis end to
end.
```

Since the cells are equal in area, these counts can be compared as they are,
with no area weighting anywhere.

When a segment contributes an exposure, a duration, a probability, or a capacity
rather than a single count, use `coverage.reduce(px.Sum(values))` instead. The
native reducer reads `Coverage.offsets` directly, so it never has to repeat one
value per hit.

## Summarize revisit over a timeline

When a coverage's segments happen to be consecutive time bins, `revisit()` reads
them as a timeline and returns per-cell statistics. The bins are ordinal, so you
apply your own time base afterwards:

```{doctest}
>>> stats = px.revisit(swath_coverage)
>>> observed_once = stats.run_counts == 1
>>> bool(np.all(stats.maximum_internal_gap_steps[observed_once] == 0))
True
>>> worst_gap_s = stats.maximum_internal_gap_steps * 60
>>> bool(np.all(stats.first_start < stats.last_stop))
True
```

Every field describes the same thresholded axis, so `run_counts`,
`internal_gap_steps_sum`, and `maximum_internal_gap_steps` can be combined
directly. `first_start` and `last_stop` bound the window a cell was observed in,
which is what you need to add the gaps at the ends of the horizon yourself.

With several aligned timelines, pass the whole sequence at once. Matching segment
indices have to describe identical time boundaries, and consecutive bins have to
be adjacent in time; `minimum_sources=2` then asks for cells covered by two
sources at once. Whether your sources are really distinct is something we cannot
check, so that part is yours. Percentiles, leading and trailing gaps, and cyclic
gap definitions are policy and not geometry, so they are left to the analysis
layer as well. For a sampled sweep, remember that these are occupied bins rather
than exact access events, and that the boundaries are only as precise as the
sampling cadence.

## Count overlaps without building membership

If cap counts per cell are all you need, ask for them directly and the
region-cell pairs are never built at all:

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
...     candidate_cells=site_cells,
...     reduce=px.Count(),
... )
>>> site_counts.shape
(2,)
```

## When a region comes back empty

A cell is covered when its center falls inside your region, so a region smaller
or thinner than a cell can quite legitimately come back empty. The fix is to
raise `resolution` until the cells are smaller than the smallest feature you care
about. [Resolutions](resolutions.md#picking-one) lists the sizes, and
[center-sampled coverage](concepts.md#center-sampled-coverage) shows exactly what
the rule includes and leaves out.

## Where to go next

- [How it works](concepts.md) explains resolution, center sampling, segmented
  results, occupancy runs, and how to hand cell IDs to other HEALPix libraries.
- [Performance and memory](performance.md) covers sizing results, sparse
  queries, batching, and threads.
- [API reference](api.md) has the exact signatures and validation rules.
