# Changelog

## Unreleased

### Added

- Replaced the convex-only polygon entry point with `cover_polygon()`. Concave
  arrays now work directly; `Polygon(outer, *holes)` and
  `MultiPolygon(*polygons)` add holes and multipart regions without adding a
  GIS or CRS object model. Convex arrays retain the existing native fast path.
- Added scale-invariant batch direction-to-RING indexing through `cell_at()`,
  including automatic parallelism for large inputs.
- Added exact spherical-cap coverage through `cover_cap()`.
- Added `Count` and `Sum` reducers, accepted as `reduce=` by every covering
  call and by `Coverage.reduce()` for cell lists that are already built. A
  reducer names the accumulation; Polypix fuses it into the geometry kernel
  where that is faster. Omitting `reduce=` keeps the full `Coverage`.
- Added `revisit(timelines, *, minimum_sources=1)`, returning per-cell run
  counts, complete internal gap sums and maxima, and observed window bounds
  computed in a single pass without ever building the runs. The result carries
  those six arrays and nothing else: every one of them needs that pass, and
  anything omitted is either one NumPy expression away or was an argument the
  caller supplied.
- Added ragged convex polygon batches, passed as a sequence of
  `(vertices, 3)` arrays, and `cell_count()`.

### Changed

- Replaced the angle sum used for every concave polygon cell with a flat
  crossing check, and grouped detailed boundaries by height so most cells only
  inspect nearby edges. A 128-edge concave benchmark at resolution 9 fell from
  313 to about 7 milliseconds; an 8-edge notched polygon fell from 5.5 to 0.9
  milliseconds. The matched convex benchmarks stayed within run-to-run noise.
  Raw batches also handle concave items in the native pass now, instead of
  abandoning that pass and retrying the whole batch through the general path.
- Validated a ragged polygon batch in one pass rather than one call per
  polygon. The shapes already had to be read to choose between the dense and
  ragged paths, and the offsets follow from them, so converting and checking
  entry by entry was pure duplication: one `concatenate` and one whole-buffer
  check now replace a per-polygon Python call that cost about five
  microseconds each. A 200,000-polygon ragged batch fell from 432 to 271
  milliseconds and a 10,000-polygon one from 20.4 to 14.1, with no regression
  on small batches. The offending entry is still named exactly, by rerunning
  the per-entry path once something has already failed.
- Answered a reduction over a small cell selection by testing those cells
  instead of covering everything and gathering. Under a reducer,
  `candidate_cells` names the only cells the result depends on, so the
  selection is itself a candidate set. Below a size bound the covering calls
  hand it to the kernel as one, which measured 220x faster for polygons at
  resolution 10, 870x for sums at resolution 11 with a third of the peak
  memory, and 30x for a sweep - all bitwise-identical results. Larger
  selections keep scanning once and gathering, which stays faster for them.
  Without a reducer `candidate_cells` is unchanged: the caller's restriction on
  the coverage itself, not a hint.
- Validated the offsets every segmented cell array carries in one shared place,
  so the revisit entry point checks them too. They index
  `cells[offsets[i]..offsets[i + 1]]` directly, but only checked that the
  offsets were nonempty and agreed on a segment count. Offsets no `Coverage`
  produced - reachable through the native functions, which take any arrays -
  therefore panicked, or, when the first offset was not zero, silently dropped
  the leading hits of every segment.
- Named the entry at fault when a ragged batch mixed vertex *widths*. The
  dense and ragged paths were chosen by comparing vertex counts alone, so
  `[(3, 3), (3, 2)]` looked uniform, reached `np.asarray` as a ragged nested
  sequence, and surfaced its message - "setting an array element with a
  sequence" - instead of naming `polygons_xyz[1]`. The choice now compares
  whole shapes.
- Rejected a `Coverage` whose cells were mutated after construction with a
  `ValueError` instead of a Rust panic. The reductions do not rescan validated
  hits, but a result array owns its data, so Python can reset its read-only
  flag and write to it. The dense accumulators now take the bounds check they
  already performed as a result rather than a panic, and the map-keyed ones,
  which have no such bound, compare against the grid explicitly.
- Upgraded to `pyo3` 0.29.2 and `numpy` 0.29.0. The new `as_slice()` requires
  correct alignment as well as contiguity, which fixes a slice taken over a
  misaligned pointer. A contiguous array can still be misaligned, and neither
  `asarray()` nor `ascontiguousarray()` repairs that, so the conversion helpers
  force the copy and such inputs keep working; the native message now names
  alignment. The declared `rust-version` stays 1.87, which the crate needs for
  `is_multiple_of`, and is above the 1.83 both dependencies require.

- Replaced `count_caps_per_cell()`, `count_coverage_per_cell()`, and
  `sum_coverage_per_cell()` with the `reduce=` reducer form, removing the
  geometry-specific reducer asymmetry. The fused cap kernel is retained behind
  `cover_cap(..., reduce=Count())`.
- Chose between the fused cap kernel and covering-then-reducing for
  `cover_cap(..., candidate_cells=..., reduce=Count())` by comparing their
  estimated costs.
  A fused selected count tests every cap against every requested cell, so it is
  kept for small requests and for caps too large to store; large requests
  now cover once and reduce. Always fusing was measured at up to 47x the cost.
  The comparison lives in the kernel, next to the code whose cost it weighs,
  and declines before preparing anything so that falling back stays cheap.
- Added typed `@overload` signatures to `cover_polygon()`,
  `cover_cap()`, and `cover_sweep()`, so a `reduce=` call site resolves to
  `Coverage` or the accumulated array under `mypy --strict` instead of `Any`.
- Narrowed the per-cell revisit statistics accumulator from 32 to 24 bytes,
  which the existing segment-count bound already permits, recovering 9 to 16
  percent on `revisit()` across the dense and sparse paths.
- Budgeted the dense revisit state array against the accumulator actually
  allocated, and rejected segment and source counts that would truncate through
  `u32` rather than returning a silently wrong result.
- Skipped the redundant per-hit revalidation of an already-validated,
  read-only `Coverage` inside `revisit()` and the reductions, cutting
  two of four full passes over the hits.
- Removed two redundant validation passes when importing an integer array,
  which the move to signed public indices had put on the path of feeding any
  Polypix result back into the next call. The int64 range check now runs only
  for unsigned and object inputs, where it is not already implied, and the
  non-negative check is a reduction rather than a comparison that also
  materializes a boolean temporary. The remaining scan is deferred to the
  native cell-range validation that every cell argument already performs: a
  reinterpreted negative index exceeds every resolution's cell count, so
  `validate_cell_range()` now distinguishes it and reports the same message the
  Python scan did. Offset arrays, which are bounds-checked rather than
  range-checked, keep their scan. `Coverage.from_arrays()` on a 921,600-hit
  coverage went from 2.80 ms to 1.56 ms, against 2.29 ms for the equivalent
  unsigned arrays.
- Served queried `Count` and `Sum` reductions from a dense scratch grid up to
  resolution 8, instead of one hash probe per hit, but only when the coverage
  and query together touch enough of that grid to amortize zeroing it. Sparse
  higher-resolution queries, and small queries against a large grid, keep the
  hash path and its flat memory profile.
- Made `Coverage` a validated, read-only segmented interchange type with
  `from_arrays()`, segment indexing, `len()`, and zero-copy native results.
- Standardized public cell IDs, offsets, and revisit window bounds on signed
  `int64`.
- Renamed the paired-edge operation from `cover_strip()` to `cover_sweep()` and
  the canonical convex-region operation to `cover_polygon()`. Renamed
  cell transforms to `cell_centers()` and `cell_corners()`.
- Made zero- and one-sample sweeps consistently return empty segmented
  coverage.
- Replaced the mixed source-local/source-unioned occupancy summary with
  thresholded per-cell revisit statistics.
- Cached scan-ring geometry at common resolutions, reused sweep sample
  normalization, and removed repeated polygon-longitude transforms.
- Reworked the executable Starlink example around exact caps and the
  Earth-observation example around native ordinal revisit statistics.

### Fixed

- Selected reductions over large footprints no longer scan the whole grid when
  testing the selection is cheaper. The choice was made in Python from the
  selection size alone, which cannot see the scan cost, so it inverted for large
  footprints: 400 quads of 1.5 degrees at resolution 11 with 50,000 selected
  cells took 69 ms instead of 11 ms. The kernel decides now, using the scan
  estimate it already computes. Results were identical either way.

- Declined to parallelize a dense cap count when its per-worker buffers cost
  more than the scan they divide. `cover_cap(..., reduce=Count())` gives each
  worker a grid-sized accumulator and merges them by addition, and that buffer
  spans the whole grid however few caps a worker's chunk holds - so unlike
  coverage chunking, more workers neither shrinks it nor amortizes the merge
  pass over it. A few thousand modest caps at resolution 9 were measured up to
  2x slower with threads than without. The existing memory budget only ruled
  out the largest grids; the decision now also weighs the scan against the
  merge, and falls back to one sequential buffer when it does not clear both.
- Covered the cells a footprint's longitude span ends on. `cover_sweep()` and
  `cover_polygon()` derive one longitude interval per footprint; when
  that span ended exactly on the prime meridian the interval was treated as
  unwrapped, so the cells at longitude zero - offset 0 of every unshifted ring -
  were never scanned. A footprint with a vertex on the meridian silently lost up
  to one cell per ring it crossed. Selecting `candidate_cells` was unaffected,
  because that path tests each cell directly.

### Removed

- Removed the pre-1.0 `cover_footprint()`, `centers()`, and `corners()` names;
  use `cover_polygon()`, `cell_centers()`, and `cell_corners()`.
- Removed the lossy `summarize_occupancy()` and `OccupancySummary`; use
  `revisit()` and calculate the required gap policy downstream.
- Removed the derivable `Coverage` members `counts`, `segment_sizes`,
  `segment_indices()`, `segment_count`, and `filter_hits()`. None is cheaper
  inside Polypix than outside it: use `np.diff(coverage.offsets)`,
  `np.repeat(np.arange(len(coverage)), np.diff(coverage.offsets))`, and
  `len(coverage)`. `len()` and segment indexing remain.
- Removed `cover_polygon(..., vertex_offsets=...)`. A caller holding a
  packed buffer passes a sequence of slices into it and pays one `concatenate`,
  which measured within noise of building the offsets by hand.

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
