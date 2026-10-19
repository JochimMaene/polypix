# Changelog

Notable user-facing changes are listed here, newest first. Sections follow
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
categories: breaking changes first, then features, fixes, performance, and
documentation. Internal refactors, tests, and CI changes are omitted.

## Unreleased

## [1.0.0-rc.1] — 2026-10-19

First release candidate for 1.0. From 1.0 on, Polypix follows semantic
versioning, and the public API is everything in `polypix.__all__`.

### Breaking changes

- `resolution` is keyword-only on `cover_cap()`, `cover_polygon()`,
  `cover_sweep()`, `cell_at()`, `cell_centers()`, `cell_corners()`, and
  `cell_neighbors()`, so two numbers can no longer be swapped by accident.
- Renamed the `RevisitStats` fields:

  | Before | After |
  | --- | --- |
  | `run_counts` | `visit_counts` |
  | `internal_gap_steps_sum` | `gap_steps_total` |
  | `maximum_internal_gap_steps` | `gap_steps_max` |
  | `first_start` | `first_bin` |
  | `last_stop` | `stop_bin` |

- Renamed `Coverage.reduce(cells=...)` to `candidate_cells=`, matching the
  covering functions.
- `Sum(values)` copies and checks its values when you create it. Errors are
  raised there, and `Sum.values` is a read-only `float64` array.
- A sweep segment too thin to cover now raises `ValueError`, as the same shape
  already did in `cover_polygon()`.

#### Migrating

```python
# resolution is keyword-only
px.cover_cap(centers, radii, 6)               # before
px.cover_cap(centers, radii, resolution=6)    # after
px.cell_centers(cells, 10)                    # before
px.cell_centers(cells, resolution=10)         # after

# RevisitStats field names (meanings unchanged)
stats.run_counts                              # before
stats.visit_counts                            # after

# Coverage.reduce keyword
coverage.reduce(px.Count(), cells=wanted)             # before
coverage.reduce(px.Count(), candidate_cells=wanted)   # after

# Sum keeps its own copy: finish editing the values first
weights[0] = 5.0
reducer = px.Sum(weights)                     # invalid values fail here
```

### Fixes

- Small or slightly concave polygons, and polygons with nearly duplicate
  vertices, no longer miss cells inside them.
- Polygons and sweeps with a very sharp corner no longer include cells beyond
  that corner.
- `mode="overlap"` caps with a radius close to pi no longer include cells around
  the opposite point, and run much faster there. This removes the known
  limitation listed under 0.5.1.
- Cells exactly on a region's boundary are now treated the same whether you
  scan the whole grid, pass `candidate_cells=`, or count with `reduce=`.
- Large inputs that run out of memory raise `MemoryError` instead of crashing
  the Python process.
- Processes forked after Polypix started worker threads, such as
  `multiprocessing` workers on Linux, no longer hang. They run single-threaded.
- A process that briefly cannot start threads now runs calls single-threaded,
  instead of raising `PanicException` from then on.
- `RevisitStats` arrays and `Sum` values stay read-only after copying or
  pickling.
- `timedelta64` arrays are rejected as coordinates and `Sum` values, instead of
  being read as plain numbers.
- Clearer errors for cell IDs of 2**63 and above, and for cap counts too large
  for memory.

### Performance

- Overlap coverage is faster.
- Cap counts are faster: up to 1.8 times as fast for counts over selected
  cells, and 1.5 times for regional counts.
- `threads=N` no longer starts more threads than requested.

### Documentation

- The README states the versioning policy.
- The guides describe behavior under `fork()` and Ctrl-C, list the accepted
  input shapes, and give the correct resolutions for the case studies.

## [0.5.1] — 2026-09-24

### Fixes

- Fixed coverage near short polygon edges, so full scans and
  `candidate_cells=` return consistent results.
- Kept `Coverage` arrays validated and read-only after copying or unpickling.
- Fixed `mode="center"` coverage and counts for caps covering almost the
  entire sphere.

Known limitation: `mode="overlap"` still treats cap radii within roughly
2.1e-8 radians of pi as a full sphere. At resolutions 26 and above, this can
include cells outside the cap.

### Performance

- Sped up cap counts and reduced memory use when counting across the full grid.
- Sped up convex polygon and sweep coverage, ring scans, and overlap coverage.
- Sped up sparse `revisit(minimum_sources=1)` calculations.

## [0.5.0] — 2026-09-05

### Breaking changes

- Renamed the first `cover_polygon()` parameter from `polygons_xyz` to
  `geometry`.

### Features

- Added exact whole-cell intersection coverage through `mode="overlap"` on
  `cover_cap()`, `cover_polygon()`, and `cover_sweep()`.
- Accepted polygonal `__geo_interface__` objects and GeoJSON-like mappings in
  `cover_polygon()`, including holes, multipart unions, and single Features.
- Added `cell_neighbors()` for batched immediate HEALPix neighbor lookup.

### Fixes

- Stabilized short spherical-polygon edges so overlap coverage remains
  conservative at resolutions 28 and 29.
- Capped oversized positive `threads` values at the native integer limit
  before applying the existing host thread limit.

### Performance

- Accelerated selected-cell cap counts for large cap batches and dense
  `revisit(minimum_sources=1)` workloads whose state fits in cache.

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

[Unreleased]: https://github.com/JochimMaene/polypix/compare/v1.0.0-rc.1...HEAD
[1.0.0-rc.1]: https://github.com/JochimMaene/polypix/compare/v0.5.1...v1.0.0-rc.1
[0.5.1]: https://github.com/JochimMaene/polypix/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/JochimMaene/polypix/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/JochimMaene/polypix/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/JochimMaene/polypix/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/JochimMaene/polypix/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/JochimMaene/polypix/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/JochimMaene/polypix/releases/tag/v0.1.0
