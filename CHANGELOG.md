# Changelog

## Unreleased

### Added

- Added scale-invariant batch direction-to-RING indexing through `cell_at()`,
  including automatic parallelism for large inputs.
- Added exact spherical-cap coverage through `cover_cap()`.
- Added fused dense or queried per-cell cap counts through
  `count_caps_per_cell()`.
- Added sparse source-run and merged-gap reduction through
  `summarize_occupancy()` and `OccupancySummary`.

### Changed

- Made `Coverage` a validated, read-only segmented interchange type with
  `from_arrays()`, segment indexing, `len()`, and zero-copy native results.
- Renamed the paired-edge operation from `cover_strip()` to `cover_sweep()` and
  the four-point cell transform from `boundaries()` to `corners()` while the
  project remains pre-1.0.
- Cached scan-ring geometry at common resolutions, reused sweep sample
  normalization, and removed repeated polygon-longitude transforms.
- Reworked the executable Starlink example around exact caps and the
  Earth-observation example around native segmented occupancy reduction.

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
