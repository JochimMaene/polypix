# Coverage Reductions and Ordinal Occupancy Runs

Status: **accepted; pre-1.0 API consolidation**

## Decision

Polypix names the *result* a caller wants with a reducer, and every producing
operation accepts one through `into=`:

```python
Count(cells=None)                 Sum(values, cells=None)
Stats()

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
array. With `cells=` they preserve query order and duplicates and return a
result sized by the request rather than by the grid; a small grid may still use
a bounded dense scratch array internally when the call touches enough of it.
`values` is a scalar or one finite `float64` value per coverage segment.
Floating addition is deterministic in segment and hit order; non-finite inputs
and overflow are rejected.

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

`occupancy()` does not own time. A caller can map run boundaries through
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
  replaced a global tuple sort on the dense path, which initially took about
  3.09 s; the sparse path still sorts its runs, because it cannot count them
  per cell without a second pass over expensive probes. Measured peak
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

Those figures are the resolution-6 dense-state path. They are not the whole
picture, and the table above is the workload most favourable to `Stats()`. The
dense state array is sized by the grid rather than by the observed cells, so its
zeroing cost grows while the hit-bound work does not. At the largest grid the
24-byte accumulator admits — resolution 9, where 75 MiB of state is zeroed for
about 216,000 observed cells — `Stats()` is *slower* than the operation it
replaces: 33 to 37 ns per hit against 26 to 28 ns for `summarize_occupancy()`,
measured over four input sizes at 1.4 million hits, or about 1.25x. It still
holds peak RSS to 143 MiB against 207 MiB. Resolutions 6 and 11 both favour
`Stats()` by roughly 1.8x, so the regression is confined to the upper end of the
dense band and is accepted for the memory profile it buys.

Above resolution 9 both reducers fall back to hash-keyed state, and there
`Stats()` stays slower than the default run materialization — about 115 ms
against 87 ms for 725,000 runs at resolution 11 — while still avoiding the run
allocation. Narrowing every
statistics counter to 32 bits, which the segment-count bound already permits,
took the accumulator from 32 to 24 bytes and recovered 9 to 16 percent of that
gap across resolutions 9 to 12 and on the dense path. The remainder is the cost
of a larger per-cell accumulator in a hash map, and is accepted.

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

That advantage does not extend to a selected query, and the first version of
this decision routed one there anyway. A fused selected count decodes each
requested cell's centre and tests it against every cap, so its cost is
`cells * caps` rather than the hit count. For 10,771 caps at resolution 8 it
took about 457 ms for 100,000 requested cells against about 9.7 ms for covering
once and reducing, and about 810 ms against 25 ms at 200,000 — 47x and 32x for
the only spelling the API offers. `cover_cap()` therefore estimates both costs
and picks the cheaper. The estimate uses measured native throughput: about
10 ns to emit a coverage hit, about 10 ns to gather a requested cell, and about
43 ns per requested cell plus 0.48 ns per cap test to fuse. Across 240 shapes
spanning resolutions 5 to 10, one to 10,771 caps, three radii, and requests
from 10 to 100,000 cells, the worst remaining mis-pick costs about 1.3x.

The fused selected path is kept rather than deleted because it is the only one
that answers a query the coverage cannot express: a whole-sphere cap at
resolution 29 covers 1.4e19 cells, so covering first is impossible while four
requested cells resolve in about 150 microseconds. The estimate selects it there
automatically, since the covering cost it is compared against is astronomical.

The dense scratch grid for selected reductions had the same shape of defect.
Choosing it on grid size alone meant a four-cell query at resolution 8 zeroed
6 MiB for 79 probes, taking about 161 microseconds against about 11
microseconds for the hash path one resolution higher — where the grid is four
times larger. Requiring the coverage and query together to touch at least a
thirty-second of the grid, as `access.rs` already required of its dense
occupancy state, brought that case to about 9 microseconds while leaving large
queries on the dense path.

The native generic reducers avoid per-hit segment-index and weight temporaries.
The queried *result* is proportional to the requested cells rather than the
resolution-sized grid, including at resolution 29; its working storage is the
bounded dense scratch grid or a hash table, whichever the work estimate above
selects. None of these additions
changes the cap traversal or explicit coverage hot paths.

Moving public indices to signed `int64` does change one path that no kernel
touches: importing an array back into Polypix. A signed array takes the
non-negative branch of index validation, which an unsigned array skipped, and
that branch now runs on every array Polypix itself returns. Two passes were
avoidable and are gone. The int64 range check is dead for a signed input, since
a value that is not negative is already inside int64, so it runs only for
unsigned and object inputs. The non-negative check is a reduction rather than
`any(array < 0)`, which read the array once instead of also materializing a
boolean temporary of equal length. Re-importing the 921,600-hit
Earth-observation coverage through `Coverage.from_arrays()` went from 2.80 ms to
2.27 ms, against 2.29 ms for the same call given unsigned arrays; instruction
count, which weights a streaming compare far more heavily than wall clock does,
had risen about 29 percent and is recovered.

The last pass is deferred to the kernel rather than kept. Every entry point
that hands a *cell* array to native code already range-checks it there, and a
reinterpreted negative index arrives as a `u64` at or above `1 << 63`, which no
resolution can contain. `validate_cell_range()` therefore names that case
instead of reporting a generic out-of-range index, and the Python scan is
skipped wherever such a check is guaranteed to follow. The message a caller sees
is unchanged at every argument. Offset arrays keep the scan: they are
bounds-checked rather than range-checked, and they hold one value per segment
rather than one per hit, so the pass costs nothing worth removing. The
positional reduction query keeps a Python check for a different reason — the
kernel knows it as `requested_cells`, not the public `cells` — but classifies
both failures from a single maximum rather than two scans.

Re-importing the same coverage then took 1.56 ms, against 2.29 ms for the
unsigned arrays it is measured against, because a signed array views where an
unsigned one casts. The per-call overhead of a signed argument to
`cell_centers()` and `cell_corners()` fell from 3.5 and 3.9 microseconds to
about 0.8 and 0.7. Deferring validation is sound only argument by argument, so
the flag that enables it is explicit at each call site and defaults to
scanning.

These values are regression-design evidence from one machine, not public
cross-machine performance claims, and the cost constants above are calibration
rather than contract: a mis-estimate costs a constant factor, never a wrong
result. Repository benchmarks guard the dense and selected fused cap paths on
both sides of the work estimate, the generic reducers including a small-work
selected query, and both occupancy reducers. Tests assert that the two sides of
each estimate agree exactly.

## Trusting a validated Coverage

The reductions do not rescan the hits a `Coverage` already validated, which is
where two of their four passes went. That trust needs one qualification: a
result array owns its data, so Python can reset the read-only flag and write to
it afterwards, and rust-numpy documents the same limit on its own
`make_nonwriteable`. A mutated source must therefore still fail cleanly.

It costs nothing to make it. The dense accumulators index a grid-sized state
array, so the bound is already enforced on every hit; taking it as an `Option`
rather than a panic changes only where the failure goes. The map-keyed
accumulators have no such bound and compare against the grid explicitly, which
is negligible beside the hash probe that follows and replaces a silently wrong
result with an error. Measured across resolutions 6, 9, and 12 on both
reducers, the difference stays inside run-to-run noise. Revalidating the hits
would have cost the passes this decision removed; enforcing the bound that was
already there costs nothing.

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
- Unifying the dense and sparse occupancy paths behind one state-store
  abstraction, to remove the accumulation skeleton that is written out for each
  combination of reducer and memory profile. This was implemented and measured,
  and it is 2.8x to 4.2x slower: `occupancy()` on a sparse resolution-12
  workload went from 422 ms to 1.78 s, and the dense resolution-8 path from
  31 ms to 88 ms. Two independent costs explain it. Nesting the per-segment
  interval count inside a generic accumulator adds alignment padding, taking the
  dense run state from 16 to 24 bytes and the statistics state from 32 to 40.
  And a single shared driver forces one algorithm on both profiles: the dense
  path wants two passes over cheap array indexing so it can write boundaries
  into an exact allocation, while the sparse path wants one pass because every
  probe is expensive, and paying for two doubles its hashing. The duplication is
  therefore load-bearing rather than accidental, and it stays.

- A shared kernel sink abstraction, so every geometry could fuse every reducer.
  Admissible later and invisible to the API by construction, but unjustified
  now: only the cap kernel has a measured fusing advantage.
