# Changelog

Notable user-facing changes are listed here, newest first. Sections follow
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
categories: breaking changes first, then features, fixes, performance, and
documentation. Internal refactors, tests, and CI changes are omitted.

## Unreleased

### Breaking changes

- Renamed the first `cover_polygon()` parameter from `polygons_xyz` to
  `geometry`.

### Features

- Accepted polygonal `__geo_interface__` objects and GeoJSON-like mappings in
  `cover_polygon()`, including holes, multipart unions, and single Features.
- Added `cell_neighbors()` for batched immediate HEALPix neighbor lookup.

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

- Reduced a 128-edge concave polygon benchmark at resolution 9 from about
  313 ms to 7 ms while keeping convex performance unchanged.
- Made small `candidate_cells` reductions up to hundreds of times faster by
  testing only the requested cells when that is cheaper than scanning the grid.
- Reduced conversion and validation overhead for ragged batches, existing
  `Coverage` objects, integer arrays, and repeated polygon use.
- Avoided parallel cap-count and selected-reduction strategies when their
  memory or merge cost would be slower than the sequential path.

### Documentation

- Added executable Starlink and Earth-observation examples, including a
  resolution-9 analysis restricted by a concave Germany area of interest.

## [0.3.0] — 2026-07-28

Version 0.3 replaced the GPL-backed native backend with a Polypix-owned Rust
kernel and changed the project license to Apache-2.0. Earlier releases remain
available under their original license terms.

### Breaking changes

| 0.2 API | 0.3 API |
| --- | --- |
| `cover_swath(...)` | `cover_strip(...)` |
| `cell_ids` | `cells` |
| `allowed_cell_ids=` | `candidate_cells=` |

- Removed packed NESTED cell IDs, backend selection, and the `polypix.bench`
  command. Public cell IDs became fixed-resolution HEALPix RING indices.

### Features

- Added a Polypix-owned Rust kernel for fixed-resolution HEALPix RING coverage.
- Added optional bounded native threading and Windows wheels.

### Fixes

- Tightened geometry, candidate-cell, and array validation around the new
  native kernel.

### Performance

- Added automatic parallelism for large calls while retaining a low-latency
  sequential path for small inputs.

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

[Unreleased]: https://github.com/JochimMaene/polypix/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/JochimMaene/polypix/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/JochimMaene/polypix/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/JochimMaene/polypix/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/JochimMaene/polypix/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/JochimMaene/polypix/releases/tag/v0.1.0
