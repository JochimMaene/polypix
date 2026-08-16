# Getting started

Install Polypix from PyPI:

```bash
pip install polypix
```

NumPy is the only runtime dependency, and there are wheels for CPython 3.12+ on
Linux, macOS, and Windows. [Installation](install.md) covers source builds.

## Cover spherical regions

Everything you pass in is a Cartesian direction. Magnitude is ignored, so
position vectors work just as well as unit vectors.

A spherical cap is a center direction plus an angular radius in radians:

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

Both regions come back in one result, kept apart by offsets:

```pycon
>>> coverage.counts
array([1502, 3824])
>>> coverage[0].shape
(1502,)
```

`coverage[i]` is a read-only view of the RING cell IDs for region `i`. Every
region operation follows the same rule: a cell is selected when its center lies
inside the region.

For a convex polygon, give the vertices in boundary order. Adjacent vertices are
joined by the shorter great-circle arc:

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

If every polygon has the same vertex count, a dense `(regions, vertices, 3)`
array is the fastest thing to hand over. For a ragged batch, pass a sequence of
arrays instead.

## Assign directions to cells

When your input is points rather than regions, use `cell_at()`:

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

Note that `cell_centers` holds grid representatives, not the directions you
started with. Only cell centers round-trip exactly:
`cell_at(centers(cells, r), r) == cells`.

## Cover a sampled sweep

Two aligned boundary curves define one quadrilateral between every adjacent
sample pair:

```python
left = np.array([[1.0, -0.1, -0.1], [1.0, -0.1, 0.1], [1.0, -0.1, 0.3]])
right = np.array([[1.0, 0.1, -0.1], [1.0, 0.1, 0.1], [1.0, 0.1, 0.3]])

swept = px.cover_sweep(left, right, resolution=8)
assert len(swept) == 2
```

This is what you want for a moving footprint whose edges you have already
sampled. Your sampling *is* the geometry here — Polypix joins adjacent points
with minor great-circle arcs and cannot tell a sparse sample from a deliberate
one.

## Count overlaps without building them

If the question is really "how many caps contain each cell?", skip the
membership step:

```python
counts = px.count_caps_per_cell(centers, radii, resolution=8)
```

The result is indexed by RING cell ID, and you never build a `Coverage`
holding the same cell once per covering cap. At high resolution, ask about a
short list of cells instead:

```python
site_counts = px.count_caps_per_cell(
    centers,
    radii,
    resolution=16,
    cells=site_cells,
)
```

## Where to go next

- [User guide](concepts.md) — resolution, center sampling, segmented results,
  occupancy summaries.
- [Performance and memory](performance.md) — sizing results, sparse queries,
  batching, threads.
- [API reference](api.md) — the complete call contract.
- [Interoperability](interoperability.md) — handing data to other HEALPix and
  astronomy packages.

```{toctree}
:hidden:
:maxdepth: 1

install
```
