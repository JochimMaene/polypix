# Interoperability

Polypix uses ordinary NumPy arrays and standard fixed-resolution HEALPix RING
IDs as its ecosystem boundary. It deliberately does not introduce frame-aware
objects or require another astronomy or geospatial package at runtime.

## Grid conventions

```text
nside      = 2 ** resolution
cell_count = 12 * 4 ** resolution
```

Polypix accepts resolutions 0 through 29 and therefore the power-of-two subset
of HEALPix RING grids. Cell arrays use `uint64`; the resolution is carried by
the result object, not encoded in each ID. NESTED ordering and mixed-resolution
representations remain outside the library.

Every valid Polypix cell ID fits signed `int64`. Convert deliberately when a
downstream API requires signed IDs:

```python
signed_cells = coverage.cells.astype(np.int64, copy=False)
```

## Cartesian directions

Geometry arrives as `(N, 3)` Cartesian vectors in one caller-defined frame.
Magnitude is ignored. If another library returns component-major `(3, N)`
vectors, move the component axis explicitly:

```python
vectors_n3 = np.moveaxis(vectors_3n, 0, -1)
cells = px.cell_at(vectors_n3, resolution=8)
```

Polypix attaches no unit, celestial frame, body, datum, CRS, or epoch to these
arrays. Coordinate transformations belong upstream.

## Geometry producers upstream

Astropy, Skyfield, Orekit, SPICE wrappers, propagation libraries, and custom
sensor models can resolve physical state into direction-space geometry. The
boundary is:

```text
upstream:  state + physical constraints -> directions, caps, or footprint edges
Polypix:   resolved spherical regions -> fixed-resolution center membership
downstream: maps, units, statistics, MOCs, plotting, or persistence
```

Minimum elevation, off-nadir limits, attitude, ellipsoid intersection,
refraction, and terrain therefore remain upstream. Pass their resulting cap or
footprint geometry to Polypix.

## Other HEALPix tools

Standard RING IDs can be handed to healpy, astropy-healpix, cdshealpix, or
another HEALPix implementation for functionality Polypix intentionally omits:
ordering conversion, neighbors, interpolation, map resampling, harmonics, and
file formats.

`corners()` returns only four HEALPix corner vectors. HEALPix edges are curved,
so those points are not a sampled boundary and must not be treated as an exact
great-circle polygon for round-tripping.

Turning center-selected fixed-resolution cells into a MOC changes the
representation to whole-cell area. It does not retroactively give the original
operation conservative intersection semantics.

## Imported segmented membership

Use `Coverage.from_arrays()` when another system already has one flat RING
array plus segment offsets. The factory copies and validates the arrays,
including cell ranges and within-segment uniqueness. Native Polypix producers
avoid that copy because they already own and validate their output.
