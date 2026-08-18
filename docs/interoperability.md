# Interoperability

Polypix exchanges two things with the rest of the ecosystem: `(N, 3)` NumPy
direction arrays and fixed-resolution HEALPix RING IDs. There is no frame object
model to adopt and no Astropy or geospatial runtime to install.

## Cell IDs

```text
nside      = 2 ** resolution
cell_count = 12 * 4 ** resolution
```

Resolutions 0 through 29 give you the power-of-two subset of RING grids. Cells
come back as `uint64`, with the resolution on the result object rather than
baked into each ID. NESTED ordering and mixed-resolution representations are not
here.

Every valid cell ID fits in signed `int64`, so when an API insists on signed:

```python
signed_cells = coverage.cells.astype(np.int64, copy=False)
```

## Directions

Geometry arrives as `(N, 3)` vectors in whatever frame you are working in.
Magnitude is ignored. If another library hands you component-major `(3, N)`,
move the axis yourself:

```python
vectors_n3 = np.moveaxis(vectors_3n, 0, -1)
cells = px.cell_at(vectors_n3, resolution=8)
```

Polypix attaches no unit, frame, body, datum, CRS, or epoch to these arrays.

## Who does what

```text
upstream:   state + physical constraints -> directions, caps, footprint edges
Polypix:    resolved regions -> center membership
downstream: maps, units, statistics, MOCs, plotting, storage
```

Astropy, Skyfield, Orekit, SPICE wrappers, and your own sensor models resolve
physical state into direction-space geometry. Minimum elevation, off-nadir
limits, attitude, ellipsoid intersection, refraction, terrain: all of that is
upstream. Pass Polypix the cap or footprint that falls out of them.

## Other HEALPix libraries

Standard RING IDs go straight to healpy, astropy-healpix, or cdshealpix for
everything Polypix leaves out: ordering conversion, neighbors, interpolation,
resampling, harmonics, file formats.

Two things to watch when you hand data over:

- `corners()` returns four corner vectors, and HEALPix cell edges are curved.
  Those four points are not a sampled boundary, so do not round-trip them as an
  exact great-circle polygon.
- A MOC represents whole cells by area. Converting center-selected cells into a
  MOC changes what the result means. It does not retroactively turn your query
  into an intersection query.

## Bringing segmented data in

`Coverage.from_arrays()` takes a flat RING array plus offsets from another
system. It copies and validates them, checking cell ranges and within-segment
uniqueness. Polypix's own calls skip that copy because they already own and
trust their output.
