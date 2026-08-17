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

Polypix wants Cartesian directions, so a small helper keeps the examples in
degrees:

```{literalinclude} ../examples/docs_diagrams.py
:language: python
:caption: examples/docs_diagrams.py
:start-after: "--8<-- [start:unit-vector]"
:end-before: "--8<-- [end:unit-vector]"
```

```{literalinclude} ../examples/docs_diagrams.py
:language: python
:caption: examples/docs_diagrams.py
:start-after: "--8<-- [start:cover-cap]"
:end-before: "--8<-- [end:cover-cap]"
```

Both caps come back in one result, kept apart by offsets. `coverage.counts` is
how many cells each one got, and the `assert` above is not decoration: these
snippets are included straight from a file the test suite runs, so the numbers
in them cannot quietly go stale.

`coverage[i]` is a read-only view of the RING cell IDs for region `i`. Every
region operation follows the same rule: a cell is selected when its center lies
inside the region.

```{figure} assets/generated/cover-cap.svg
:alt: Two spherical caps and the grid cells each one covers.
:width: 100%
:align: center

The two caps above, and the cells `cover_cap()` returned for them.
```

For a convex polygon, give the vertices in boundary order. Adjacent vertices are
joined by the shorter great-circle arc:

```{literalinclude} ../examples/docs_diagrams.py
:language: python
:caption: examples/docs_diagrams.py
:start-after: "--8<-- [start:cover-footprint]"
:end-before: "--8<-- [end:cover-footprint]"
```

```{figure} assets/generated/cover-footprint.svg
:alt: A convex polygon and the grid cells it covers.
:width: 100%
:align: center

A four-sided footprint and the cells it covers.
```

If every polygon has the same vertex count, a dense `(regions, vertices, 3)`
array is the fastest thing to hand over. For a ragged batch, pass a sequence of
arrays instead.

## Assign directions to cells

When your input is points rather than regions, use `cell_at()`:

```{literalinclude} ../examples/docs_diagrams.py
:language: python
:caption: examples/docs_diagrams.py
:start-after: "--8<-- [start:cell-at]"
:end-before: "--8<-- [end:cell-at]"
```

```{figure} assets/generated/cell-at.svg
:alt: Four directions, each snapped to the grid cell containing it, with an arrow to that cell's centre.
:width: 100%
:align: center

`cell_at()` gives you the cell a direction falls in. `centers()` then gives that cell's centre, which is the arrow head, not where you started.
```

Note that `cell_centers` holds grid representatives, not the directions you
started with. Only cell centers round-trip exactly:
`cell_at(centers(cells, r), r) == cells`.

## Cover a sampled sweep

Two aligned boundary curves define one quadrilateral between every adjacent
sample pair:

```{literalinclude} ../examples/docs_diagrams.py
:language: python
:caption: examples/docs_diagrams.py
:start-after: "--8<-- [start:cover-sweep]"
:end-before: "--8<-- [end:cover-sweep]"
```

```{figure} assets/generated/cover-sweep.svg
:alt: Two sampled edges, the quadrilaterals between consecutive samples, and the cells they cover.
:width: 100%
:align: center

Each pair of consecutive samples becomes one quadrilateral, and each quadrilateral is its own segment in the result.
```

This is what you want for a moving footprint whose edges you have already
sampled. Your sampling *is* the geometry here. Polypix joins adjacent points
with minor great-circle arcs, and it cannot tell a sparse sample from a
deliberate one.

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

- [User guide](concepts.md) explains resolution, center sampling, segmented
  results, and occupancy summaries.
- [Performance and memory](performance.md) covers sizing results, sparse
  queries, batching, and threads.
- [API reference](api.md) is the complete call contract.
- [Interoperability](interoperability.md) covers handing data to other HEALPix
  and astronomy packages.

```{toctree}
:hidden:
:maxdepth: 1

install
```
