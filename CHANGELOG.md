# Changelog

User-facing changes, newest first, grouped as breaking changes, features,
fixes, performance, and documentation. Internal refactors, tests, and CI are
omitted.

## Unreleased

### Features

- Python 3.15 is supported.

### Fixes

- A polygon list entry with the wrong width is named in the error.

### Performance

- Coverage over `candidate_cells` is up to 20 times as fast when the selection
  is much wider than each polygon or sweep.
- Overlap coverage is 1.4 to 2.7 times as fast, and up to 2 times as fast
  again over `candidate_cells` that span many longitudes, such as a land mask.
- Polygon lists convert up to 1.8 times as fast.
- Large threaded results, about a million cells and up, are assembled in
  parallel.
- `Count` and `Sum` over sparse `candidate_cells` at high resolutions are up to
  3 times as fast.
- Cap counts at resolution 9 and above are up to 1.7 times as fast.
- Dense cap counts no longer skip threading when the output grid outgrows the
  number of cap rings: about 3 times as fast at resolutions 8 and 9 on eight
  logical CPUs.
- `revisit()` calls with few hits no longer clear a whole grid first, which
  made them up to 50 times slower at resolutions 5 to 8.

## [1.0.0] — 2026-11-04

First stable release, unchanged since 1.0.0-rc.1. Polypix now follows semantic
versioning; the public API is `polypix.__all__`.

### Breaking changes

- `resolution` is keyword-only on every covering and cell function:
  `px.cover_cap(centers, radii, resolution=6)`.
- Renamed the `RevisitStats` fields:

  | Before | After |
  | --- | --- |
  | `run_counts` | `visit_counts` |
  | `internal_gap_steps_sum` | `gap_steps_total` |
  | `maximum_internal_gap_steps` | `gap_steps_max` |
  | `first_start` | `first_bin` |
  | `last_stop` | `stop_bin` |

- Renamed `Coverage.reduce(cells=...)` to `candidate_cells=`.
- `Sum(values)` copies and validates its values when created; `Sum.values` is a
  read-only `float64` array.
- A sweep segment too thin to cover raises `ValueError`, as in
  `cover_polygon()`.

### Fixes

- Small or slightly concave polygons, and polygons with nearly duplicate
  vertices, no longer miss interior cells.
- Very sharp polygon and sweep corners no longer include cells past the corner.
- `mode="overlap"` caps with a radius close to pi no longer include cells
  around the opposite point, and run much faster there.
- Boundary cells are treated the same by full scans, `candidate_cells=`, and
  `reduce=`.
- Running out of memory raises `MemoryError` instead of crashing Python.
- Processes forked after Polypix started threads no longer hang; they run
  single-threaded, as do calls when threads cannot start (previously
  `PanicException`).
- `RevisitStats` arrays and `Sum` values stay read-only after copying or
  pickling.
- `timedelta64` arrays are rejected as coordinates and `Sum` values.
- Clearer errors for cell IDs of 2**63 and above, and for cap counts too large
  for memory.

### Performance

- Faster overlap coverage.
- Cap counts over selected cells are up to 1.8 times as fast, regional counts
  1.5 times.
- `threads=N` never starts more than N threads.

### Documentation

- Documented the versioning policy, behavior under `fork()` and Ctrl-C, and the
  accepted input shapes.

## [0.5.1] — 2026-09-24

### Fixes

- Full scans and `candidate_cells=` agree near short polygon edges.
- `Coverage` arrays stay validated and read-only after copying or unpickling.
- Fixed `mode="center"` coverage and counts for caps covering almost the whole
  sphere.

Known limitation, fixed in 1.0.0: `mode="overlap"` treats cap radii within
about 2.1e-8 radians of pi as the full sphere, which can add cells at
resolutions 26 and above.

### Performance

- Faster cap counts, with less memory across the full grid.
- Faster convex polygon, sweep, ring-scan, and overlap coverage.
- Faster sparse `revisit(minimum_sources=1)`.

## [0.5.0] — 2026-09-05

### Breaking changes

- Renamed the first `cover_polygon()` parameter from `polygons_xyz` to
  `geometry`.

### Features

- `mode="overlap"` on `cover_cap()`, `cover_polygon()`, and `cover_sweep()`
  covers every cell a region touches.
- `cover_polygon()` accepts GeoJSON-like mappings and `__geo_interface__`
  objects, including holes, multipart geometries, and single Features.
- Added `cell_neighbors()`.

### Fixes

- Overlap coverage stays conservative near short edges at resolutions 28 and
  29.
- Very large `threads` values no longer overflow.

### Performance

- Faster selected-cell cap counts for large batches, and faster dense
  `revisit(minimum_sources=1)`.

## [0.4.0] — 2026-08-26

### Breaking changes

- Renamed `cover_footprint()` to `cover_polygon()`, `cover_strip()` to
  `cover_sweep()`, `centers()` to `cell_centers()`, and `boundaries()` to
  `cell_corners()`.
- Removed `Coverage.counts`; use `np.diff(coverage.offsets)` instead.
- Standardized cell IDs, offsets, and revisit window bounds on signed `int64`.

### Features

- Added concave polygon coverage. `Polygon(outer, *holes)` supports holes and
  `MultiPolygon(*polygons)` combines separate components into one region.
- Added exact spherical-cap coverage with `cover_cap()`.
- Added `Count` and `Sum` reducers to every covering function and to
  `Coverage.reduce()`.
- Added `revisit()` for per-cell run counts, internal gap statistics, and
  observed time bounds across one or more sources.
- Added `cell_at()` for converting direction vectors to HEALPix RING cells.
- Added `Coverage.from_arrays()`, segment indexing, `len()`, and `cell_count()`.

### Fixes

- Included cells when a polygon or sweep ends exactly on the prime meridian.
- Made malformed offsets, mutated coverage arrays, and invalid cell values
  raise Python exceptions instead of reaching native panics or wrong results.
- Made empty and one-sample sweeps consistently return empty coverage.

### Performance

- A 128-edge concave polygon covers about 45 times as fast.
- Small `candidate_cells` reductions are up to hundreds of times as fast.
- Less conversion and validation overhead for ragged batches, `Coverage`
  objects, integer arrays, and reused polygons.
- Cap counts and selected reductions stay serial when threads would be slower.

### Documentation

- Added runnable Starlink and Earth-observation examples.

## [0.3.0] — 2026-07-28

Polypix's own Rust kernel replaces the GPL-backed backend, and the license
changes to Apache-2.0. Earlier releases keep their original license.

### Breaking changes

| 0.2 API | 0.3 API |
| --- | --- |
| `cover_swath(...)` | `cover_strip(...)` |
| `cell_ids` | `cells` |
| `allowed_cell_ids=` | `candidate_cells=` |

- Removed packed NESTED cell IDs, backend selection, and the `polypix.bench`
  command. Public cell IDs became fixed-resolution HEALPix RING indices.

### Features

- Optional native threading and Windows wheels.

### Fixes

- Stricter geometry, candidate-cell, and array validation.

### Performance

- Large calls run in parallel automatically; small ones stay serial.

## [0.2.1] — 2026-07-24

### Features

- Added `allowed_cell_ids=` to restrict footprint and swath coverage to a known
  set of cells.

## [0.2.0] — 2026-06-07

### Breaking changes

- Replaced `Polygon`, `MultiPolygon`, and `cover()` with direct XYZ-array input
  to `cover_footprint()`.
- Renamed `center()` and `boundary()` to their plural forms and changed batched
  coverage to return a segmented `Coverage` object.

### Features

- Added `cover_swath()` for coverage between sampled left and right edges.
- Added dense footprint batches and per-input offsets in `Coverage`.

## [0.1.0] — 2026-06-04

### Features

- Initial release with center-sampled coverage for convex spherical polygons.
- Accepted longitude/latitude or Cartesian vertices, including polygon batches.
- Returned packed HEALPix NESTED cell IDs with center and boundary helpers.
- Published Python 3.12 wheels for Linux and macOS.

[Unreleased]: https://github.com/JochimMaene/polypix/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/JochimMaene/polypix/compare/v0.5.1...v1.0.0
[0.5.1]: https://github.com/JochimMaene/polypix/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/JochimMaene/polypix/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/JochimMaene/polypix/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/JochimMaene/polypix/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/JochimMaene/polypix/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/JochimMaene/polypix/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/JochimMaene/polypix/releases/tag/v0.1.0
