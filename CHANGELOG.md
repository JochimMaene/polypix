# Changelog

## Unreleased

### Added

- Added scale-invariant batch direction-to-RING indexing through `cell_at()`,
  including automatic parallelism for large inputs.
- Added exact spherical-cap coverage through `cover_cap()`.
- Added `Count`, `Sum`, `Runs`, and `Stats` reducers, accepted as `into=` by
  every covering call and by `occupancy()`, and by `Coverage.reduce()` for
  already-materialized membership. A reducer names the result; Polypix fuses
  the accumulation into the geometry kernel where that is faster.
- Added `occupancy(sources, *, minimum_sources=1, into=None)`, returning
  lossless half-open runs by default or, with `into=Stats()`, per-cell run
  counts, complete internal gap sums and maxima, and observed window bounds
  computed in a single pass without materializing runs.
- Added packed ragged convex polygons through `vertex_offsets=`,
  `Coverage.segment_indices()`, `Coverage.filter_hits()`, and
  `cell_count()`.

### Changed

- Replaced `count_caps_per_cell()`, `count_coverage_per_cell()`,
  `sum_coverage_per_cell()`, `occupancy_runs()`, and `occupancy_stats()` with
  the `into=` reducer form, removing the geometry-specific reducer asymmetry
  and the two-verb occupancy split. The fused cap kernel is retained behind
  `cover_cap(..., into=Count())`.
- Skipped the redundant per-hit revalidation of an already-validated,
  read-only `Coverage` inside `occupancy_runs()` and the reductions, cutting
  two of four full passes over the hits.
- Served queried `count_coverage_per_cell()` and `sum_coverage_per_cell()`
  results from a dense scratch grid up to resolution 8, instead of one hash
  probe per hit. Sparse higher-resolution queries keep the hash path and its
  flat memory profile.
- Made `Coverage` a validated, read-only segmented interchange type with
  `from_arrays()`, segment indexing, `len()`, and zero-copy native results.
- Standardized public cell IDs, offsets, segment indices, and occupancy-run
  indices on signed `int64`; renamed the canonical per-segment size property to
  `segment_sizes`.
- Renamed the paired-edge operation from `cover_strip()` to `cover_sweep()` and
  the canonical convex-region operation to `cover_convex_polygon()`. Renamed
  cell transforms to `cell_centers()` and `cell_corners()`.
- Made zero- and one-sample sweeps consistently return empty segmented
  coverage.
- Replaced the mixed source-local/source-unioned occupancy summary with complete
  `OccupancyRuns`.
- Cached scan-ring geometry at common resolutions, reused sweep sample
  normalization, and removed repeated polygon-longitude transforms.
- Reworked the executable Starlink example around exact caps and the
  Earth-observation example around native ordinal occupancy runs.

### Removed

- Removed the pre-1.0 `cover_footprint()`, `centers()`, and `corners()` names;
  use `cover_convex_polygon()`, `cell_centers()`, and `cell_corners()`.
- Removed the ambiguous `Coverage.counts` property; use `segment_sizes`.
- Removed the lossy `summarize_occupancy()` and `OccupancySummary`; use
  `occupancy_runs()` and calculate the required gap policy downstream.

## 0.3.0 — 2026-07-28

Version 0.3 is a deliberate pre-1.0 cleanup. It replaces the native backend,
adopts a permissive license, and makes breaking API changes without carrying
deprecated aliases.

### Migration from 0.2

| 0.2 API | 0.3 API |
| --- | --- |
| `cover_swath(...)` | `cover_strip(...)` |
| `cell_ids` result field | `cells` |
| `allowed_cell_ids=` | `candidate_cells=` |

The old `polypix.bench` command was removed. Focused regression benchmarks
remain in this repository; cross-library comparisons are being prepared in a
separate repository.

### Changed

- Replaced the previous GPL-backed coverage implementation with a Polypix-owned
  Rust kernel that emits center-sampled HEALPix RING indices directly.
- Standardized the public surface on fixed-resolution RING indices, NumPy
  arrays, segmented `Coverage` results, and optional bounded native threading.
- Added automatic native parallelism for large `centers()` and `boundaries()`
  arrays while retaining the sequential latency path for smaller inputs.
- Changed the project license to Apache-2.0. Releases through 0.2.1 remain
  available under the license terms with which they were originally published.
- Removed NESTED compatibility and backend-selection machinery. Breaking
  changes are intentional while Polypix remains pre-1.0.
