# Changelog

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
