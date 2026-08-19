# Coverage Reductions and Ordinal Occupancy Runs

Status: **accepted; pre-1.0 API consolidation**

## Decision

Polypix names the *result* a caller wants with a reducer, and every producing
operation accepts one through `into=`:

```python
Count(cells=None)                 Sum(values, cells=None)
Runs()                            Stats()

cover_convex_polygon(..., into=None)   cover_cap(..., into=None)
cover_sweep(..., into=None)            occupancy(sources, ..., into=None)
Coverage.reduce(reducer)
```

`into=None` returns the materialized `Coverage` or `OccupancyRuns`. A reducer
returns its accumulated array or statistics instead. `Coverage.reduce()` applies
the same vocabulary to already-materialized membership, so covering once and
reducing several times stays a single concept.

A reducer is a request, not a promise about the algorithm. Polypix fuses the
accumulation into the geometry kernel where measurement shows that wins and
materializes membership otherwise; the result is identical either way, so
fusing an additional pair later is an invisible optimization rather than an API
change.

Reductions remain native. With `cells=None` they return a dense fixed-grid
array. With `cells=` they preserve query order and duplicates and do not
allocate by global cell count. `values` is a scalar or one finite `float64`
value per coverage segment. Floating addition is deterministic in segment and
hit order; non-finite inputs and overflow are rejected.

The fused cap kernel remains, but it is no longer a separate public verb.
`cover_cap(..., into=Count())` accumulates private RING spans without ever
materializing cap-cell membership, and measurements show that this is a
material end-to-end advantage. The former `count_caps_per_cell()` spelling is
removed: it made a measured implementation detail look like an API asymmetry,
and invited a Cartesian product of geometry-specific reducer names that this
decision explicitly rejects. Convex polygons and sweeps accept the same
reducers and materialize first until a workload proves fusing them pays.

For aligned ordered coverage, Polypix exposes one operation with two results:

```python
occupancy(sources, *, minimum_sources=1, into=None) -> OccupancyRuns | OccupancyStats
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

`into=Stats()` returns `OccupancyStats`: per observed cell, `run_counts`,
`internal_gap_steps_sum`, `maximum_internal_gap_steps`, and the observed window
bounds `first_start` and `last_stop`, with `internal_gap_counts` derived.
Working state is one accumulator per observed cell rather than one entry per
run, and the pass is single. Every field describes the same thresholded,
source-unioned axis.

The older `summarize_occupancy()` and `OccupancySummary` are removed.
Their result mixes source-local `run_counts` with source-unioned gap fields and
retains only sum and count of complete gaps. That cannot answer common maximum
revisit questions and makes the aggregation axis easy to misread.
`Stats()` is not a reinstatement: it reports one axis and answers maximum
revisit directly, and it reports window bounds rather than choosing a leading,
trailing, or cyclic gap policy.

The governing principle for both halves of the API is:

> Materialize lossless runs when run boundaries are the required result; fuse
> per-cell statistics when runs are only an intermediate.

The same rule explains the fused cap kernel. It is not a cap-specific
exception but the geometry-side instance of one idea: do not build a large
intermediate to produce a small answer.

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
- `occupancy()` names the analysis and `into=` names the result, so the
  occupancy pair reads the same way as the covering calls; the former
  `occupancy_runs()` and `occupancy_stats()` spellings are removed;
- public cells, offsets, segment indices, and run indices are non-negative
  signed `int64`, avoiding casts and mixed signed/unsigned promotion in NumPy.

`cover_sweep()` remains a first-class convenience. Paired left/right edges are
the natural output of scan geometry and the operation avoids Python-side
quadrilateral assembly. The name distinguishes a moving region from a HEALPix
latitude strip. Zero or one paired sample now consistently yields zero output
segments, as other empty batches do.

## Superseded: a conditional return type

`cap-and-occupancy-primitives.md` rejected `cover_cap(..., output="counts")`
because it "would make the return type unstable; a separate verb is clearer."
`into=` is that shape and supersedes that rejection.

Two things changed. The separate verbs were tried, and they produced exactly
the asymmetry this decision is fixing: one geometry had a reducer and the others
did not, on grounds no caller could see. And the return type is not in practice
unstable: it is a function of a literal reducer at the call site, statically
knowable through typed overloads, and the reducer names the result in the call
itself. A caller reading `cover_cap(..., into=Count())` is not in doubt about
what comes back.

The earlier objection remains correct about the general case. A boolean or
string flag that toggles a return type is worse than two verbs. A reducer value
that *is* the requested result is not that pattern.

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

That last measurement priced the run implementation against a worse run
implementation, and not against the aggregated result it replaced. Doing so
exposes the cost the first version of this decision did not carry. On the
Earth-observation example, the same workload end to end:

| | reduce phase | example total | peak RSS |
| --- | --- | --- | --- |
| `summarize_occupancy()` | 68 ms | 217 ms | 132 MiB |
| `occupancy()` runs, reduced in NumPy | 311 ms | 476 ms | 413 MiB |
| `occupancy(into=Stats())` | 52 ms | 208 ms | 132 MiB |

The middle row is not an implementation defect. A cell here is observed briefly
and revisited hours later, so 9.28 million hits yield 9.20 million runs: the
representation compresses nothing on this shape of workload, and 147 MiB of run
boundaries exist only to be reduced away. The fused pass restores the memory
profile and is faster than the removed summary, which is why lossless runs stay
the default only where boundaries are the answer.

Retaining the fused cap kernel is measured against the reducer path it would
otherwise use: for 10,771 caps, `cover_cap(..., into=Count())` took about
9.7 ms and 1.8 MiB at resolution 6 and about 39 ms and 13 MiB at resolution 8,
against about 12.5 ms and 14 MiB, and about 127 ms and 212 MiB, for covering
then reducing. The advantage compounds with resolution, which is why the kernel
survives even though its public verb does not.

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
    -> Coverage                      (into=None)
    -> count/sum maps                (into=Count()/Sum(), or Coverage.reduce)
    -> ordinal runs or per-cell occupancy statistics  (occupancy, into=)
    -> caller-owned time mapping, statistics, constraints, and visualization
```

Users doing only cap counts retain the faster fused path, now spelled like
every other reduction. Users doing polygon or sweep analysis get the same
reducers and can reuse materialized coverage through `Coverage.reduce()`.
Revisit libraries receive complete ordinal runs instead of one irreversible
mean-gap summary, and workflows needing only per-cell revisit numbers get them
without paying for boundaries they discard.

The design still excludes arbitrary reducer callbacks, timestamps, cadence,
physical access rules, general map algebra, and a promise of conservative cell
intersection coverage.

## Rejected alternatives

- Separate reduction verbs per geometry. Tried, and `count_caps_per_cell()` was
  the result: a measured implementation detail promoted to an API asymmetry
  that no caller could predict, with `count_convex_polygons_per_cell()` and
  friends waiting behind it.
- Keeping `occupancy_runs()` and `occupancy_stats()` as separate names. That is
  the same asymmetry in a different place: two verbs for one analysis, differing
  only in whether an intermediate is materialized.
- A lazy or deferred evaluation layer over `Coverage`. It cannot fuse the
  downstream NumPy that consumes a result, so it would not have removed the
  Earth-observation cost, and it would cost `Coverage` its defining property as
  a concrete zero-copy interchange value for a pipeline only two nodes deep.
- A shared kernel sink abstraction, so every geometry could fuse every reducer.
  Admissible later and invisible to the API by construction, but unjustified
  now: only the cap kernel has a measured fusing advantage.
