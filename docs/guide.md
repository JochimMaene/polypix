# Getting started

Install Polypix from PyPI:

```bash
python -m pip install polypix
```

NumPy is its only runtime dependency. Pre-built wheels are available for
CPython 3.12 and newer on Linux, macOS, and Windows. See [Installation](install.md)
for source builds and platform details.

## Cover spherical regions

Polypix works with Cartesian directions. Vector magnitude is ignored, so both
unit vectors and ordinary position vectors are accepted.

An exact spherical cap is a center direction plus an angular radius in radians:

```python
import numpy as np
import polypix as px

centers = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)
radii = np.deg2rad([5.0, 8.0])

coverage = px.cover_cap(centers, radii, resolution=8)
```

The result keeps the batch in two arrays:

```pycon
>>> coverage.counts
array([1502, 3824])
>>> coverage[0].shape
(1502,)
```

`coverage[i]` is a read-only NumPy view containing the RING cell IDs for input
region `i`. All region operations use the same rule: a cell is selected when
its center lies inside the region.

For a convex polygon, supply its vertices in boundary order. Edges follow the
shorter great-circle arc between adjacent vertices:

```python
footprint = np.array(
    [
        [1.0, -0.12, -0.08],
        [1.0,  0.12, -0.08],
        [1.0,  0.12,  0.08],
        [1.0, -0.12,  0.08],
    ]
)

coverage = px.cover_footprint(footprint, resolution=8)
```

A dense `(regions, vertices, 3)` array is the fastest input form when every
polygon has the same number of vertices. A Python sequence of arrays also works
for a ragged batch.

## Assign directions to cells

Use `cell_at()` when the input is a set of points rather than regions:

```python
directions = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [-2.0, 1.0, 0.5],
    ]
)

cells = px.cell_at(directions, resolution=8)
cell_centers = px.centers(cells, resolution=8)
```

`cell_centers` contains the grid representatives, not the original directions.
The exact center round trip is `cell_at(centers(cells, r), r) == cells`.

## Cover a sampled sweep

Two aligned boundary curves define one quadrilateral between every adjacent
sample pair:

```python
left = np.array([[1.0, -0.1, -0.1], [1.0, -0.1, 0.1], [1.0, -0.1, 0.3]])
right = np.array([[1.0, 0.1, -0.1], [1.0, 0.1, 0.1], [1.0, 0.1, 0.3]])

swept = px.cover_sweep(left, right, resolution=8)
assert len(swept) == 2
```

This is useful for a moving footprint whose left and right edges have already
been sampled upstream. Sampling controls the geometry: Polypix joins adjacent
points with minor great-circle arcs.

## Count overlaps without materializing them

If the question is “how many caps contain each cell?”, use the fused count
operation:

```python
counts = px.count_caps_per_cell(centers, radii, resolution=8)
```

The dense result is indexed by RING cell ID. It avoids constructing a
`Coverage` containing the same cell once for every cap that covers it. At high
resolution, query a small list of cells instead:

```python
site_counts = px.count_caps_per_cell(
    centers,
    radii,
    resolution=16,
    cells=site_cells,
)
```

## Where to go next

- [Concepts](concepts.md) explains resolution, center sampling, segmented
  results, and occupancy summaries.
- [Performance and memory](performance.md) covers result sizing, sparse
  queries, batching, and threads.
- [API reference](api.md) is the complete call contract.
- [Interoperability](interoperability.md) covers handoff to other HEALPix and
  astronomy packages.

```{toctree}
:hidden:
:maxdepth: 1

install
```
