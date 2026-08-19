# Changelog

## Unreleased

### Added

- Added scale-invariant batch direction-to-RING indexing through `cell_at()`,
  including automatic parallelism for large inputs.
- Added exact spherical-cap coverage through `cover_cap()`.
- Added `Count`, `Sum`, and `Stats` reducers, accepted as `into=` by every
  covering call and by `occupancy()`, and by `Coverage.reduce()` for cell lists
  that are already built. A reducer names the result; Polypix fuses the
  accumulation into the geometry kernel where that is faster. Omitting `into=`
  keeps the full result: a `Coverage`, or every occupancy run.
- Added `occupancy(sources, *, minimum_sources=1, into=None)`, returning
  lossless half-open runs by default or, with `into=Stats()`, per-cell run
  counts, complete internal gap sums and maxima, and observed window bounds
  computed in a single pass without building the runs.
- Added packed ragged convex polygons through `vertex_offsets=`,
  `Coverage.segment_indices()`, `Coverage.filter_hits()`, and
  `cell_count()`.

### Changed

- Replaced `count_caps_per_cell()`, `count_coverage_per_cell()`,
  `sum_coverage_per_cell()`, `occupancy_runs()`, and `occupancy_stats()` with
  the `into=` reducer form, removing the geometry-specific reducer asymmetry
  and the two-verb occupancy split. The fused cap kernel is retained behind
  `cover_cap(..., into=Count())`.
- Chose between the fused cap kernel and covering-then-reducing for
  `cover_cap(..., into=Count(cells=...))` by comparing their estimated costs.
  A fused selected count tests every cap against every requested cell, so it is
  kept for small requests and for caps too large to store; large requests
  now cover once and reduce. Always fusing was measured at up to 47x the cost.
  The comparison lives in the kernel, next to the code whose cost it weighs,
  and declines before preparing anything so that falling back stays cheap.
- Added typed `@overload` signatures to `cover_convex_polygon()`,
  `cover_cap()`, `cover_sweep()`, and `occupancy()`, so an `into=` call site
  resolves to `Coverage`, `OccupancyRuns`, `OccupancyStats`, or the accumulated
  array under `mypy --strict` instead of `Any`.
- Narrowed the per-cell occupancy statistics accumulator from 32 to 24 bytes,
  which the existing segment-count bound already permits, recovering 9 to 16
  percent on `occupancy(..., into=Stats())` across the dense and sparse paths.
- Budgeted the dense occupancy state array against the accumulator actually
  allocated, so `into=Stats()` no longer admits a grid sized for the smaller
  run accumulator, and rejected segment and source counts that would truncate
  through `u32` rather than returning a silently wrong result.
- Skipped the redundant per-hit revalidation of an already-validated,
  read-only `Coverage` inside `occupancy_runs()` and the reductions, cutting
  two of four full passes over the hits.
- Served queried `Count` and `Sum` reductions from a dense scratch grid up to
  resolution 8, instead of one hash probe per hit, but only when the coverage
  and query together touch enough of that grid to amortize zeroing it. Sparse
  higher-resolution queries, and small queries against a large grid, keep the
  hash path and its flat memory profile.
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
