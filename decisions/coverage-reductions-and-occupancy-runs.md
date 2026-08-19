# Coverage Reductions and Ordinal Occupancy Runs

Status: **accepted; pre-1.0 API consolidation**

## Decision

Polypix uses `Coverage` as the geometry-neutral boundary for per-cell
accumulation:

```python
count_coverage_per_cell(coverage, *, cells=None) -> ndarray[int64]
sum_coverage_per_cell(coverage, values, *, cells=None) -> ndarray[float64]
```

Both operations are native. With `cells=None` they return a dense fixed-grid
array. With `cells=` they preserve query order and duplicates and do not
allocate by global cell count. `values` is a scalar or one finite `float64`
value per coverage segment. Floating addition is deterministic in segment and
hit order; non-finite inputs and overflow are rejected.

The cap-only fused operation remains:

```python
count_caps_per_cell(centers_xyz, radii_rad, resolution, *, cells=None,
                    threads=None) -> ndarray[int64]
```

It is not the template for a Cartesian product of geometry-specific reducers.
It remains because the cap kernel can accumulate private RING spans without
ever materializing cap-cell membership, and measurements show that this is a
material end-to-end advantage. Convex polygons and sweeps use the generic
`Coverage` reducers unless a count-only workload later proves that membership
materialization is its dominant bottleneck.

For aligned ordered coverage, Polypix exposes the lossless ordinal primitive:

```python
occupancy_runs(sources, *, minimum_sources=1) -> OccupancyRuns
```

Each `Coverage` sequence entry counts as one source. For every observed cell, the result
contains maximal half-open `[start, stop)` intervals where at least
`minimum_sources` source entries cover the cell. Its cell-major representation
is `cells`, `offsets`, `starts`, and `stops`. All indices are signed `int64`.
The result also retains `minimum_sources` and `source_count` so its threshold
provenance is not lost when it crosses an analysis boundary.

Source uniqueness, identical bin boundaries, and temporal adjacency are caller
obligations; equal resolution and segment count alone cannot prove them. A time
discontinuity must split the analysis or be represented by an empty separator
bin so a run cannot cross it.

`occupancy_runs()` does not own time. A caller can map run boundaries through
constant or arbitrary time-edge arrays and can choose complete, leading,
trailing, finite-horizon, or cyclic gaps. It also leaves mean, maximum, median,
frequency, minimum-duration, and short-gap-merging policies downstream.
Thresholding intentionally loses source attribution; a workflow needing
observer identity or routing calls the operation per source or retains the
original `Coverage` values. For sweep-derived inputs the runs describe sampled
occupied bins, not exact continuous access events.

The older `summarize_occupancy()` and `OccupancySummary` are removed.
Their result mixes source-local `run_counts` with source-unioned gap fields and
retains only sum and count of complete gaps. That cannot answer common maximum
revisit questions and makes the aggregation axis easy to misread.

## Naming and interchange consistency

The same consolidation adopts these canonical names and representations:

- `cover_convex_polygon()` names the accepted geometry precisely; the former
  `cover_footprint()` spelling is removed;
- `cell_centers()` and `cell_corners()` make the cell transform axis explicit;
  the former `centers()` and `corners()` spellings are removed;
- `Coverage.segment_sizes` replaces the ambiguous name `counts`;
- `Coverage.segment_indices()` provides the common flat-hit-to-segment mapping;
- `Coverage.filter_hits()` preserves segmentation after downstream predicates;
- `cover_convex_polygon(..., vertex_offsets=...)` accepts an already-packed
  ragged batch without a Python sequence of arrays;
- `cell_count(resolution)` exposes the fixed-grid size;
- public cells, offsets, segment indices, and run indices are non-negative
  signed `int64`, avoiding casts and mixed signed/unsigned promotion in NumPy.

`cover_sweep()` remains a first-class convenience. Paired left/right edges are
the natural output of scan geometry and the operation avoids Python-side
quadrilateral assembly. The name distinguishes a moving region from a HEALPix
latitude strip. Zero or one paired sample now consistently yields zero output
segments, as other empty batches do.

## Why the reducers are not geometry-specific

Adding `count_convex_polygons_per_cell()`, `count_sweeps_per_cell()`, weighted
variants, selected variants, and future geometry variants would grow a sparse
operation-by-geometry matrix into the public API. The shared `Coverage` seam
instead gives every geometry the same count and constant-weight sum semantics,
and lets callers cover once and perform several reductions.

This design does not claim that materialization is always free. A
geometry-specific fused operation is still admissible when it avoids a measured
large intermediate and materially changes the complete workflow. Exact cap
counting meets that test. API symmetry alone does not.

## Performance evidence

On the development host, representative release-mode probes gave these
maintenance measurements:

- for about 1.63 million cap-coverage hits, native generic counting took about
  2.48 ms versus 3.36 ms for `numpy.bincount`; native weighted summation took
  about 2.98 ms versus 10.08 ms for `repeat(values)` plus `bincount`;
- for a synthetic 6.4-million-hit coverage, native count and sum took about
  9.66 ms and 14.78 ms, versus 18.59 ms and 22.34 ms for the corresponding
  NumPy expressions;
- on the same 1.63-million-hit cap workload, fused cap counting took about
  4.58 ms for the complete operation. Materializing cap coverage took about
  4.93 ms before the additional 2.48 ms generic count, so retaining the fused
  path remained materially faster and avoided the membership allocation;
- on the ten-source Earth-observation workload, 9.28 million hits produced
  9.20 million lossless runs in about 0.31 s. Direct two-pass materialization
  replaced a global tuple sort that initially took about 3.09 s; measured peak
  reducer memory above the prepared inputs fell from roughly 353 MiB to
  143 MiB, which is close to the unavoidable 141 MiB result itself.

The native generic reducers avoid per-hit segment-index and weight temporaries.
The queried path uses storage proportional to the requested cells rather than
the resolution-sized grid, including at resolution 29. None of these additions
changes the cap traversal or explicit coverage hot paths.

These values are regression-design evidence from one machine, not public
cross-machine performance claims. Repository benchmarks guard both the fused
cap path and the generic reducers.

## Correctness evidence

Generic reducers are checked against independent NumPy oracles for dense and
queried outputs, duplicate queries, empty segments, scalar and per-segment
weights, malformed data, non-finite values, overflow, and sparse resolution-29
queries. Fused cap counts remain checked against materialized exact cap coverage.

Occupancy runs are checked against a randomized boolean-state oracle and fixed
cases for source switches, simultaneous coverage, thresholds, gaps, empty
segments, impossible thresholds, malformed sources, and high-resolution sparse
state. Runs are sorted by cell, chronological within a cell, maximal, and
half-open.

## Consequences

The typical generic analysis flow is now:

```text
resolved cap, polygon, or sweep geometry
    -> Coverage
    -> count/sum maps and/or ordinal occupancy runs
    -> caller-owned time mapping, statistics, constraints, and visualization
```

Users doing only cap counts retain the faster fused path. Users doing polygon
or sweep analysis get one consistent reduction API and can reuse materialized
coverage. Revisit libraries receive complete ordinal runs instead of one
irreversible mean-gap summary.

The design still excludes arbitrary reducer callbacks, timestamps, cadence,
physical access rules, general map algebra, and a promise of conservative cell
intersection coverage.
