# Coverage Reductions and Ordinal Occupancy Runs

Status: **accepted; pre-1.0 API consolidation**. Amended: the argument is
spelled `reduce=` rather than `into=`, cell selection moved from the reducer
tokens onto the operations, and lossless runs were withdrawn. See
[Amendments](#amendments).

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

  Attempted since, for dense polygon and sweep `Count`/`Sum`, and reverted. The
  scan seam itself is cheap and now exists - `cover_centers()` takes a visit
  closure rather than a `Vec`, at no measured cost - but the consumer side did
  not clear the bar this decision sets. Sequentially the fused path was 1.3x to
  1.4x faster and halved peak memory, from 930 MiB to 424 MiB, on 4000
  overlapping resolution-11 footprints; it was indistinguishable on ordinary
  sparse workloads, where allocating and zeroing the grid-sized result already
  dominates both paths and fusing cannot reduce it.

  The blocker is parallelism, not the seam. Chunking by footprint gives each
  worker a grid-sized accumulator to merge afterward, and that buffer does not
  shrink with more workers, so the merge alone made resolution-9 counts up to
  2x slower with threads than without, and a resolution-11 count 7x slower than
  materializing. Gating the fused path to `threads=1` recovered the win but put
  it behind a flag no caller would think to reach for, and left the default
  path - the one that would benefit from halved peak memory - unimproved.

  What would qualify: partition the *output* rather than the input. Give each
  worker a disjoint contiguous ring range and let it scan every footprint whose
  `z_bounds()` overlaps that range, emitting only into its own slice. No
  per-worker grid buffer, no merge pass, no memory budget, no work-ratio
  constant, and the memory win lands on the default path at every thread count.
  Revisit then, or once the scan emits ordered ranges rather than cells, at
  which point a dense count stops enumerating hits at all - range endpoints and
  one prefix sum - and becomes asymptotically better rather than a constant
  factor.


## Amendments

### `into=` is spelled `reduce=`

`into` reads as NumPy's `out=` - write the result into this preallocated buffer
- but nothing is ever written into a `Count()`. The argument selects a
reduction and changes the return type, which is what `Coverage.reduce()`
already called it, so the same operation had two names depending on where it
was written. Renamed; nothing else about the design above changes.

### Cell selection belongs to the operation, not the reducer

`Count(cells=...)` and `candidate_cells` reached the same native `selected`
argument by different routes, and the routing was backwards: supplying
`candidate_cells` *disabled* the fused cap counter while `Count(cells=...)`
enabled it, for the same set of cells. Supplying both was worse - the selection
governed the output while the candidates governed the scan, so a genuinely
covered site outside the candidate set reported zero, indistinguishable from
uncovered.

`candidate_cells` is now the only spelling, and what it means follows the
operation. Without a reducer the selection is the result, so it restricts the
scan unconditionally and keeps set semantics. With one it fixes the output's
index space - one value per requested cell, in query order, duplicates
preserved - and the restriction becomes the internal choice this decision
already justified, taken only below the size bound measured above. A stored
coverage spells the same selection as `Coverage.reduce(reducer, cells=...)`.

Two consequences are load-bearing rather than cosmetic. The cap fusion gate had
to move off `candidate_cells is None`, or every selected query would lose the
fused kernel. And a selection under a reducer had to switch the output to
positional, or `candidate_cells` with `reduce=Count()` at resolution 29 would
raise `MemoryError` allocating the dense grid, taking the selected
high-resolution query - answerable no other way - with it.

### Lossless runs withdrawn

`OccupancyRuns` and the `Stats` token are gone; `occupancy()` returns
`OccupancyStats` unconditionally.

The evidence is the history of the only caller. The Earth-observation example
went `summarize_occupancy()` -> `occupancy_runs()` -> `occupancy(...,
Stats())`: it started at a summary, was moved to lossless runs, and came back
to a summary. Nothing in the repository consumed the boundaries themselves. The
performance case above already favoured statistics on that shape - 208 ms and
132 MiB against 476 ms and 413 MiB - because the runs approach one boundary
pair per hit when cells are observed briefly and revisited later.

The uses that would justify runs are real and named above: percentiles,
minimum-duration filtering, short-gap merging, arbitrary per-run timestamps. If
one arrives, re-adding a return type behind a new argument breaks no caller,
where carrying an unused one to 1.0 fixes it permanently. Pre-1.0, removals are
breaking and additions are not, so the smaller surface is the reversible
choice.

The Rust unit tests that came with the runs accumulator covered the shared
source validation as well, so they were ported to `occupancy_stats()` rather
than deleted, and the statistics gained the independent reference oracle they
had lacked - previously they were only ever checked against runs.

### The statistics result carries only what the pass can produce

`OccupancyStats` went from eleven members to six, against one test: if a caller
can compute it in NumPy for free, or already holds it, Polypix should not
return it.

Removed as derivable: `internal_gap_counts`, which was a property returning
`run_counts - 1`. Removed as already the caller's: `resolution`,
`segment_count`, `minimum_sources`, and `source_count`, every one of them an
echo of an argument just passed in, and used nowhere outside the tests that
asserted them.

What remains is `cells`, `run_counts`, `internal_gap_steps_sum`,
`maximum_internal_gap_steps`, `first_start`, and `last_stop`. None of these is
recoverable from the others. `maximum_internal_gap_steps` is the sharpest case
and the one that justifies the whole fused pass: an individual gap exists only
in the instant one run closes and the next opens, so obtaining it downstream
would mean materializing every run - which is what this decision removed for
costing one boundary pair per hit. Cutting that field would put `OccupancyRuns`
straight back.

`first_start` and `last_stop` stay for the same structural reason rather than
because anything uses them heavily. This decision deliberately delegates
leading, trailing, and cyclic gap policy to the caller; those bounds are the
only mechanism that makes the delegation possible, and the trailing gap is
`segment_count - last_stop` against a segment count the caller already has.
Unused is a reason to cut. Unused but load-bearing for a delegation the library
makes on purpose is not.

### `occupancy()` is `revisit()`

`occupancy` was the only name on the surface that did not say what it produced.
Everything else is geometry to cells or a grid transform; this one operation
reads an *ordered* axis, and the name gave no hint of it. Worse, it named a
state - whether a cell is occupied - where the result is statistics about that
state over time: visit counts, gaps between visits, window bounds.

`revisit(timelines) -> RevisitStats` says all three parts, and the term travels
across the domains this library serves: EO revisit, communications outage,
survey cadence. It also finishes what renaming the argument to `timelines`
started; `occupancy(timelines) -> OccupancyStats` was committed to the temporal
reading in one place and neutral in two.

The cost is that the name now asserts a temporal reading the semantics still
refuse to fix - the result stays ordinal, and mapping bins to physical time
remains the caller's. That was judged the lesser evil: the operation is only
meaningful on an ordered axis, so a name that hides the ordering serves nobody.

### `Coverage` carries no derivable members

`segment_sizes`, `segment_indices()`, and `segment_count` are `np.diff`,
`np.repeat`, and `len(offsets) - 1`. None is cheaper inside Polypix than
outside it, and `segment_count` merely duplicated `len(coverage)`. Removed
under the same rule that trimmed the revisit result.

`__len__` and `__getitem__` stay. They are the container protocol rather than
computation, `coverage[i]` returns a zero-copy view rather than allocating, and
without them the offsets layout would have to be hand-indexed at every call
site. The guide now introduces `cells`, `offsets`, `len()`, and `np.diff` in
its first example, which teaches the layout earlier than the convenience
properties did.

### `vertex_offsets` was measured before it was judged

The packed form had no benchmark and no in-repo caller, which twice looked like
grounds to remove it. Measured against the only alternative a caller has -
splitting the buffer into one array per polygon and letting the ragged path
concatenate it back - it is 1.7x at 10,000 polygons and 2.0x at 200,000, and it
avoids 62 MiB of peak on the larger batch.

The gap does not come from validation. The ragged path was optimized first -
twice, once to stop converting entry by entry and once to stop reading every
shape twice - taking the sequence form from 432 to 271 milliseconds at 200,000
polygons and the ratio from 2.3x to 1.44x. What remains is reading one shape
per entry and the concatenate copy, neither removable while the input is a
sequence.

1.44x is a weaker case than `Sum`'s 1.9x, and it only applied to a caller whose
data *arrives* packed. For a caller holding a list of arrays, building the
offsets to pass them ended up within noise of just passing the list, which is
the honest test of whether the argument saved work or merely moved it: before
these optimizations, doing it by hand was 20% *faster*, and that 20% was our
inefficiency rather than a real benefit.

So the argument stood or fell on columnar interop - GeoArrow, Parquet, and
database geometry columns are flat coordinates plus offsets, and deconstructing
one into 200,000 Python objects to hand it over is backwards. Removed, because
no such caller exists here or is expected: the geometry this library is built
for comes out of orbital propagation, not out of Parquet. A caller who does
hold a packed buffer passes a sequence of slices into it and pays one
concatenate, the same one they would have written themselves.

Re-adding it would break nobody, and the benchmark that would justify it is now
written down above rather than left to be rediscovered.

Both sides are now benchmarked, so the next person to ask does not have to
rediscover this. Worth recording separately: the first measurement of this ran
under `tracemalloc` and reported 5.4x. It penalizes the allocation-heavy side,
which is exactly the side under test. Time without it; measure memory in a
separate run.

### The polygon scan cannot emit ranges by assuming contiguity

This decision records that a dense count would become asymptotically better
"once the scan emits ordered ranges rather than cells". The cheap way to get
there looks obvious: a cap meets every ring in one arc, a convex polygon is an
intersection of half-spaces, so surely its ring intersection is one arc too -
find the two ends with the containment predicate and emit everything between
without testing it.

It is not. Each half-space admits an arc, and an intersection of arcs on a
circle can be *two* arcs: the ring circle passes through the polygon on two
sides without the polygon containing the pole. Attempted, and the differential
suite rejected it in seconds - 36 failures, all of them extra cells and none
missing, which is the signature of merging two arcs into one.

The counterexample is now pinned as
`test_a_convex_polygon_can_meet_one_ring_in_two_arcs`: a quad that at
resolution 5 covers offsets 24-38 and 43-50 of one ring, with eleven uncovered
cells between them.

What a correct implementation needs is the exact arc arithmetic rather than the
shortcut. For a ring of radial `r` at height `z`, edge normal `n` admits
`n_x r cos(phi) + n_y r sin(phi) + n_z z >= 0`, which is `cos(phi - psi) >= -C/R`
for `R = hypot(n_x r, n_y r)` and `psi = atan2(n_y, n_x)` - one arc per edge in
closed form, with `R == 0` and `|C/R| > 1` as the degenerate cases. Intersecting
those arcs as a set on the circle yields one or more arcs, each of which becomes
a cell range. Endpoints should still be confirmed with the containment
predicate so the tie behaviour stays bit-identical to today's, which also means
the arc arithmetic only has to round outward rather than be exact.

That is a day of delicate work whose failure mode is silent over-coverage, so
it wants the oracle pointed at it from the first commit rather than the last.
The prize is real - it is the difference between O(cells) and O(edges) per ring
on the dominant geometry - but the shortcut is not available.
