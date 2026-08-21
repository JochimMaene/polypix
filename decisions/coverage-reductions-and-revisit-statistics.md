# Coverage Reductions and Revisit Statistics

Status: **accepted; pre-1.0 API consolidation**.

## Decision

Polypix names the *result* a caller wants with a reducer, and every producing
operation accepts one through `reduce=`:

```python
Count()                                Sum(values)

cover_convex_polygon(..., reduce=None)  cover_cap(..., reduce=None)
cover_sweep(..., reduce=None)           Coverage.reduce(reducer, cells=None)

revisit(timelines, *, minimum_sources=1) -> RevisitStats
```

`reduce=None` returns the materialized `Coverage`. A reducer returns its
accumulated array instead. `Coverage.reduce()` applies the same vocabulary to
already-materialized membership, so covering once and reducing several times
stays a single concept.

A reducer is a request, not a promise about the algorithm. Polypix fuses the
accumulation into the geometry kernel where measurement shows that wins and
materializes membership otherwise; the result is identical either way, so
fusing an additional pair later is an invisible optimization rather than an API
change.

Reductions remain native. Without `candidate_cells` they return a dense
fixed-grid array. With it they preserve query order and duplicates and return a
result sized by the request rather than by the grid; a small grid may still use
a bounded dense scratch array internally when the call touches enough of it.
`values` is a scalar or one finite `float64` value per coverage segment.
Floating addition is deterministic in segment and hit order; non-finite inputs
and overflow are rejected.

The fused cap kernel remains, but it is not a separate public verb.
`cover_cap(..., reduce=Count())` accumulates private RING spans without ever
materializing cap-cell membership, and measurements show that this is a
material end-to-end advantage. A `count_caps_per_cell()` spelling would make a
measured implementation detail look like an API asymmetry, and invite a
Cartesian product of geometry-specific reducer names that this decision
explicitly rejects. Convex polygons and sweeps accept the same reducers and
materialize first until a workload proves fusing them pays.

### `candidate_cells` belongs to the operation, not the reducer

`candidate_cells` is the only cell-selection spelling, and what it means
follows the operation. Without a reducer the selection *is* the result, so it
restricts the scan unconditionally and keeps set semantics. With one it fixes
the output's index space — one value per requested cell, in query order,
duplicates preserved — and restricting the scan becomes an internal choice,
taken only below the size bound measured below. A stored coverage spells the
same selection as `Coverage.reduce(reducer, cells=...)`.

Routing selection through the reducer instead was tried and was backwards:
supplying `candidate_cells` *disabled* the fused cap counter while
`Count(cells=...)` enabled it, for the same set of cells. Supplying both was
worse — the selection governed the output while the candidates governed the
scan, so a genuinely covered site outside the candidate set reported zero,
indistinguishable from uncovered.

Two consequences are load-bearing rather than cosmetic. The cap fusion gate had
to move off `candidate_cells is None`, or every selected query would lose the
fused kernel. And a selection under a reducer had to switch the output to
positional, or `candidate_cells` with `reduce=Count()` at resolution 29 would
raise `MemoryError` allocating the dense grid, taking the selected
high-resolution query — answerable no other way — with it.

### `revisit()` returns statistics, not runs

`revisit(timelines)` reads aligned ordered coverage and returns `RevisitStats`:
per observed cell, `cells`, `run_counts`, `internal_gap_steps_sum`,
`maximum_internal_gap_steps`, and the observed window bounds `first_start` and
`last_stop`. Working state is one accumulator per observed cell rather than one
entry per run, and the pass is single. Every field describes the same
thresholded, source-unioned axis, and none is recoverable from the others.

Each `Coverage` sequence entry counts as one source; a cell is covered in a bin
when at least `minimum_sources` entries contain it. Source uniqueness,
identical bin boundaries, and temporal adjacency are caller obligations; equal
resolution and segment count alone cannot prove them. A time discontinuity must
split the analysis or be represented by an empty separator bin so a run cannot
cross it.

`revisit()` does not own time. The result stays ordinal; callers map bins
through their own time-edge arrays and choose complete, leading, trailing,
finite-horizon, or cyclic gap policy. `first_start` and `last_stop` are the
only mechanism that makes that delegation possible — the trailing gap is
`segment_count - last_stop` against a segment count the caller already has —
which is why they stay despite little direct use. It also leaves mean, maximum,
median, frequency, minimum-duration, and short-gap-merging policies downstream.
Thresholding intentionally loses source attribution; a workflow needing
observer identity calls the operation per source or retains the original
`Coverage` values. For sweep-derived inputs the statistics describe sampled
occupied bins, not exact continuous access events.

`maximum_internal_gap_steps` justifies the whole fused pass: an individual gap
exists only in the instant one run closes and the next opens, so obtaining it
downstream would mean materializing every run.

The name asserts a temporal reading the semantics deliberately refuse to fix.
That is the lesser evil: the operation is only meaningful on an ordered axis,
so a name that hides the ordering serves nobody.

### Neither result carries derivable members

`Coverage` exposes `cells`, `offsets`, `resolution`, `__len__`, and
`__getitem__`. Sizes are `np.diff(offsets)` and the hit-aligned segment index is
`np.repeat(...)`; neither is cheaper inside Polypix than outside it, so neither
is returned. `__len__` and `__getitem__` stay because they are the container
protocol rather than computation, and `coverage[i]` returns a zero-copy view.

The same rule trimmed `RevisitStats`: no `internal_gap_counts` (it is
`run_counts - 1`), and no `resolution`, `segment_count`, `minimum_sources`, or
`source_count`, every one of them an echo of an argument just passed in.

### Naming

- `cover_convex_polygon()` names the accepted geometry precisely;
- `cell_centers()` and `cell_corners()` make the cell transform axis explicit;
- `cell_count(resolution)` exposes the fixed-grid size;
- public cells, offsets, and window bounds are non-negative signed `int64`,
  avoiding casts and mixed signed/unsigned promotion in NumPy.

`cover_sweep()` remains a first-class convenience. Paired left/right edges are
the natural output of scan geometry and the operation avoids Python-side
quadrilateral assembly. The name distinguishes a moving region from a HEALPix
latitude strip. Zero or one paired sample yields zero output segments, as other
empty batches do.

## Superseded: a conditional return type

`cap-and-occupancy-primitives.md` rejected `cover_cap(..., output="counts")`
because it "would make the return type unstable; a separate verb is clearer."
`reduce=` is that shape and supersedes that rejection.

Two things changed. The separate verbs were tried, and they produced exactly
the asymmetry this decision is fixing: one geometry had a reducer and the others
did not, on grounds no caller could see. And the return type is not in practice
unstable: it is a function of a literal reducer at the call site, statically
knowable through typed overloads, and the reducer names the result in the call
itself. A caller reading `cover_cap(..., reduce=Count())` is not in doubt about
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
maintenance measurements. They are regression-design evidence from one machine,
not public cross-machine performance claims, and the cost constants below are
calibration rather than contract: a mis-estimate costs a constant factor, never
a wrong result.

**Native reducers beat the NumPy expressions they replace.** For about 1.63
million cap-coverage hits, native generic counting took about 2.48 ms versus
3.36 ms for `numpy.bincount`; native weighted summation took about 2.98 ms
versus 10.08 ms for `repeat(values)` plus `bincount`. For a synthetic
6.4-million-hit coverage, native count and sum took about 9.66 ms and 14.78 ms,
versus 18.59 ms and 22.34 ms. They avoid per-hit segment-index and weight
temporaries, and none of these additions changes the cap traversal or explicit
coverage hot paths.

**The fused cap kernel earns its keep on dense grids.** For 10,771 caps,
`cover_cap(..., reduce=Count())` took about 9.7 ms and 1.8 MiB at resolution 6
and about 39 ms and 13 MiB at resolution 8, against about 12.5 ms and 14 MiB,
and about 127 ms and 212 MiB, for covering then reducing. The advantage
compounds with resolution, which is why the kernel survives even though its
public verb does not.

**It does not extend to a selected query, so the kernel estimates both costs.**
A fused selected count decodes each requested cell's centre and tests it
against every cap, so its cost is `cells * caps` rather than the hit count. For
10,771 caps at resolution 8 it took about 457 ms for 100,000 requested cells
against about 9.7 ms for covering once and reducing, and about 810 ms against
25 ms at 200,000 — 47x and 32x. The estimate uses measured native throughput:
about 10 ns to emit a coverage hit, about 10 ns to gather a requested cell, and
about 43 ns per requested cell plus 0.48 ns per cap test to fuse. Across 240
shapes spanning resolutions 5 to 10, one to 10,771 caps, three radii, and
requests from 10 to 100,000 cells, the worst remaining mis-pick costs about
1.3x.

The fused selected path is kept rather than deleted because it is the only one
that answers a query the coverage cannot express: a whole-sphere cap at
resolution 29 covers 1.4e19 cells, so covering first is impossible while four
requested cells resolve in about 150 microseconds. The estimate selects it there
automatically, since the covering cost it is compared against is astronomical.

**The dense scratch grid for selected reductions needs a work bound, not just a
size bound.** Choosing it on grid size alone meant a four-cell query at
resolution 8 zeroed 6 MiB for 79 probes, taking about 161 microseconds against
about 11 microseconds for the hash path one resolution higher — where the grid
is four times larger. Requiring the coverage and query together to touch at
least a thirty-second of the grid, as the dense revisit state already required
of itself, brought that case to about 9 microseconds while leaving large
queries on the dense path.

**Both dense fast paths were re-measured by disabling them, and both are
load-bearing.** Each looks like a removable optimization beside a hash path that
is already correct for every input, so the cost of deleting one was measured
directly rather than assumed. Median of seven, release build, selected
reductions over random coverage:

| selected reduction | dense scratch | hash only | ratio |
| --- | --- | --- | --- |
| resolution 5, 4 cells requested | 0.010 ms | 0.031 ms | 3.1x |
| resolution 6, 20,000 requested | 0.082 ms | 1.18 ms | 14x |
| resolution 7, 90,000 requested | 0.48 ms | 4.55 ms | 9.5x |
| resolution 8, 400,000 requested | 3.24 ms | 42.4 ms | 13x |
| resolution 9 (above the bound) | 33.8 ms | 31.4 ms | — |

The same for `revisit()`, four sources, disabling the dense state array:

| revisit workload | dense state | sparse only | ratio |
| --- | --- | --- | --- |
| resolution 4 | 0.14 ms | 1.48 ms | 10.6x |
| resolution 6 | 2.42 ms | 12.4 ms | 5.1x |
| resolution 8 | 20.9 ms | 95.2 ms | 4.6x |
| resolution 9 | 53.1 ms | 96.2 ms | 1.8x |
| resolution 10 (above the bound) | 86.2 ms | 108 ms | — |

Neither is a marginal win over a narrow window, which is what both looked like
from the code alone. The rows above the bound confirm the guards hand over to
the hash path without a penalty. Do not delete either without re-running this.

**Fused revisit statistics beat materializing runs on the workload that
motivated them.** On the Earth-observation example, resolution-6 dense-state
path:

| | reduce phase | example total | peak RSS |
| --- | --- | --- | --- |
| aggregated summary (removed) | 68 ms | 217 ms | 132 MiB |
| lossless runs, reduced in NumPy | 311 ms | 476 ms | 413 MiB |
| `revisit()` | 52 ms | 208 ms | 132 MiB |

The middle row is not an implementation defect. A cell here is observed briefly
and revisited hours later, so 9.28 million hits yield 9.20 million runs: the
representation compresses nothing on this shape of workload, and 147 MiB of run
boundaries exist only to be reduced away.

That table is the workload most favourable to `revisit()`. The dense state
array is sized by the grid rather than by the observed cells, so its zeroing
cost grows while the hit-bound work does not. At the largest grid the 24-byte
accumulator admits — resolution 9, where 75 MiB of state is zeroed for about
216,000 observed cells — it is *slower* than the aggregated summary it
replaced: 33 to 37 ns per hit against 26 to 28 ns, measured over four input
sizes at 1.4 million hits, or about 1.25x. It still holds peak RSS to 143 MiB
against 207 MiB. Resolutions 6 and 11 both favour it by roughly 1.8x, so the
regression is confined to the upper end of the dense band and is accepted for
the memory profile it buys. Narrowing every counter to 32 bits, which the
segment-count bound already permits, took the accumulator from 32 to 24 bytes
and recovered 9 to 16 percent across resolutions 9 to 12.

**Signed public indices cost one import path, since recovered.** A signed array
takes the non-negative branch of index validation, which an unsigned array
skipped, and that branch runs on every array Polypix itself returns. The int64
range check is dead for a signed input, since a value that is not negative is
already inside int64, so it runs only for unsigned and object inputs. The
non-negative check is a reduction rather than `any(array < 0)`, which read the
array once instead of also materializing a boolean temporary of equal length.
Re-importing the 921,600-hit Earth-observation coverage through
`Coverage.from_arrays()` went from 2.80 ms to 2.27 ms, against 2.29 ms for the
same call given unsigned arrays.

The last pass is deferred to the kernel rather than kept. Every entry point that
hands a *cell* array to native code already range-checks it there, and a
reinterpreted negative index arrives as a `u64` at or above `1 << 63`, which no
resolution can contain, so `validate_cell_range()` names that case instead of
reporting a generic out-of-range index. The message a caller sees is unchanged
at every argument. Offset arrays keep the scan: they are bounds-checked rather
than range-checked, and hold one value per segment rather than one per hit.
Re-importing then took 1.56 ms, and the per-call overhead of a signed argument
to `cell_centers()` and `cell_corners()` fell from 3.5 and 3.9 microseconds to
about 0.8 and 0.7. Deferring validation is sound only argument by argument, so
the flag that enables it is explicit at each call site and defaults to scanning.

Repository benchmarks guard the dense and selected fused cap paths on both sides
of the work estimate, the generic reducers including a small-work selected
query, and revisit. Tests assert that the two sides of each estimate agree
exactly.

### The two branches nobody had measured

Every tuned constant costs a constant, a comment, a benchmark, and two threshold
tests, so each one has to earn that. Two branches had never been measured.
Both were disabled behind a temporary environment switch, release build, best of
five.

`dense_accumulator_parallel_worthwhile`, forced to always parallelize:

| batch | shipped | always parallel | gate is worth |
| --- | --- | --- | --- |
| 2,000 caps x 0.5 deg, res 9 | 4.9 ms | 34.7 ms | **7.1x** |
| 5,000 caps x 1.0 deg, res 10 | 60.6 ms | 212.1 ms | **3.5x** |
| 500 caps x 3.0 deg, res 10 | 45.9 ms | 217.3 ms | **4.7x** |
| 2,000 caps x 2.0 deg, res 9 | 13.9 ms | 32.4 ms | 2.3x |
| 50,000 caps x 0.5 deg, res 9 | 51.7 ms | 52.9 ms | 1.02x (hands off) |
| 200,000 caps x 0.2 deg, res 8 | 39.8 ms | 41.1 ms | 1.03x (hands off) |

The candidate-centre cache, forced off. Worth re-measuring specifically because
`plan_from_ranges` used to take the centre span over empty ranges too, so any
batch with one out-of-band item silently ran with the cache disabled — its
original numbers were taken against a partly broken cache:

| batch | shipped | no cache | cache is worth |
| --- | --- | --- | --- |
| 400 x 1.5 deg, res 9, 50k candidates | 5.9 ms | 18.1 ms | **3.1x** |
| 200 x 3.0 deg, res 10, 50k candidates | 5.5 ms | 17.1 ms | **3.1x** |
| 5,000 x 0.2 deg, res 10, 100k candidates | 22.9 ms | 59.3 ms | **2.6x** |
| 400 x 1.5 deg, res 11, 200k candidates | 25.3 ms | 64.4 ms | **2.5x** |
| 2,000 x 0.5 deg, res 9, 50k candidates | 12.3 ms | 28.4 ms | 2.3x |

Both stay. The accumulator gate's last two rows are the cases it declines, and
declining costs nothing there; the cache never lost on any workload tried.

This is the third time on this branch that reading the code ranked a branch
differently from measuring it, and the third time measuring won. The rule worth
keeping is the one `plan.rs` claims but never enforced: a tuned constant must
show a win on a workload in `examples/`, and its full cost is counted.

### One cost model, in the layer that has the inputs

An earlier revision of this branch decided in two places whether a selection
should be tested directly or answered by scanning the rings.
`_restriction_is_cheaper` in `polypix/__init__.py` decided it for every geometry
from the selection size and the grid; `covering_beats_testing` in
`ring/plan.rs` decided the same question for caps from `CELL_DECODE_TESTS` and
`COVERAGE_HIT_TESTS`. Four machine-specific constants, two layers, one question.

The Python one was not merely redundant, it was wrong. Its bound could not see
the scan cost, so it was calibrated over small footprints and inverted for large
ones. Measured on 400 quads, `reduce=Count()` with a selection, release build:

| batch | resolution | selected | restrict | scan | old bound picked |
| --- | --- | --- | --- | --- | --- |
| 400 x 1.5 deg | 11 | 50,000 | 11.3 ms | 68.7 ms | scan (**6.1x worse**) |
| 400 x 1.5 deg | 9 | 4,096 | 1.2 ms | 6.4 ms | scan (**5.3x worse**) |
| 400 x 0.5 deg | 10 | 4,096 | 0.7 ms | 2.8 ms | restrict (correct) |
| 400 x 1.5 deg | 9 | 50,000 | 8.9 ms | 10.7 ms | scan (correct) |
| 400 x 0.05 deg | 9 | 50,000 | 6.0 ms | 4.8 ms | scan (correct) |

No value of the two Python constants fixes this, because the missing input is
not the selection size. So the decision moved to the only layer that can price
it. `scanning_beats_testing` now takes an estimated hit count rather than radii,
which is all that tied it to caps, and `should_test_candidates` wraps it for the
three covering entry points. Python passes the selection as a hint and one bit
saying whether it is binding: without a reducer the selection *is* the result,
with one it only fixes an index space the reduction applies anyway.

Estimating must not cost a pass over every item, so `sampled_total` extrapolates
`estimated_cap_cells` from the same 64-item sample `accumulated_scan_work`
already used — both now share it. Caps skip sampling because `expected_total_hits`
already prices the batch exactly from its radii.

The last two rows above are the cases where scanning genuinely wins, and the new
model still picks it; the worst residual error measured is 1.2x, against 6.1x
before. Two Python constants, a 28-line apology for them, and one of the two
cost models are gone.

`test_the_kernel_may_ignore_a_candidate_set_it_prices_as_more_expensive` drives
`restrict_output` directly and asserts the two dispatches agree bitwise,
including that they actually differed — so the model stays free to be retuned.

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
result with an error. Measured across resolutions 6, 9, and 12, the difference
stays inside run-to-run noise.

## Correctness evidence

Generic reducers are checked against independent NumPy oracles for dense and
queried outputs, duplicate queries, empty segments, scalar and per-segment
weights, malformed data, non-finite values, overflow, and sparse resolution-29
queries. Fused cap counts remain checked against materialized exact cap
coverage.

Revisit statistics are checked against a randomized boolean-state oracle and
fixed cases for source switches, simultaneous coverage, thresholds, gaps, empty
segments, impossible thresholds, malformed sources, and high-resolution sparse
state.

## Consequences

The typical analysis flow is now:

```text
resolved cap, polygon, or sweep geometry
    -> Coverage                      (reduce=None)
    -> count/sum maps                (reduce=Count()/Sum(), or Coverage.reduce)
    -> per-cell revisit statistics   (revisit)
    -> caller-owned time mapping, statistics, constraints, and visualization
```

Users doing only cap counts retain the faster fused path, spelled like every
other reduction. Users doing polygon or sweep analysis get the same reducers and
can reuse materialized coverage through `Coverage.reduce()`. Workflows needing
per-cell revisit numbers get them without paying for boundaries they discard.

The design still excludes arbitrary reducer callbacks, timestamps, cadence,
physical access rules, general map algebra, and a promise of conservative cell
intersection coverage.

## Rejected alternatives

- **Separate reduction verbs per geometry.** Tried, and `count_caps_per_cell()`
  was the result: a measured implementation detail promoted to an API asymmetry
  that no caller could predict, with `count_convex_polygons_per_cell()` and
  friends waiting behind it. Likewise `occupancy_runs()` and `occupancy_stats()`
  as separate names — two verbs for one analysis, differing only in whether an
  intermediate is materialized.

- **Lossless run boundaries as a public result.** Implemented, then withdrawn.
  The evidence is the history of the only caller: the Earth-observation example
  went summary -> lossless runs -> summary. Nothing in the repository consumed
  the boundaries themselves, and the table above shows why. The uses that would
  justify runs are real and named above — percentiles, minimum-duration
  filtering, short-gap merging, arbitrary per-run timestamps. If one arrives,
  re-adding a return type behind a new argument breaks no caller, where carrying
  an unused one to 1.0 fixes it permanently. Pre-1.0, removals are breaking and
  additions are not, so the smaller surface is the reversible choice. The Rust
  tests that came with the runs accumulator covered the shared source validation
  too, so they were ported rather than deleted, and the statistics gained the
  independent reference oracle they had lacked.

- **A lazy or deferred evaluation layer over `Coverage`.** It cannot fuse the
  downstream NumPy that consumes a result, so it would not have removed the
  Earth-observation cost, and it would cost `Coverage` its defining property as
  a concrete zero-copy interchange value for a pipeline only two nodes deep.

- **Unifying the dense and sparse revisit paths behind one state-store
  abstraction.** Implemented and measured, and it is 2.8x to 4.2x slower: a
  sparse resolution-12 workload went from 422 ms to 1.78 s, and the dense
  resolution-8 path from 31 ms to 88 ms. Two independent costs explain it.
  Nesting the per-segment interval count inside a generic accumulator adds
  alignment padding, taking the statistics state from 32 to 40 bytes. And a
  single shared driver forces one algorithm on both profiles: the dense path
  wants two passes over cheap array indexing so it can write into an exact
  allocation, while the sparse path wants one pass because every probe is
  expensive. The duplication is load-bearing rather than accidental.

- **A shared kernel sink abstraction, so every geometry could fuse every
  reducer.** Attempted for dense polygon and sweep `Count`/`Sum`, and reverted.
  The scan seam itself is cheap and now exists — `cover_centers()` takes a visit
  closure rather than a `Vec`, at no measured cost — but the consumer side did
  not clear the bar this decision sets. Sequentially the fused path was 1.3x to
  1.4x faster and halved peak memory, from 930 MiB to 424 MiB, on 4000
  overlapping resolution-11 footprints; it was indistinguishable on ordinary
  sparse workloads, where allocating and zeroing the grid-sized result already
  dominates both paths.

  The blocker is parallelism, not the seam. Chunking by footprint gives each
  worker a grid-sized accumulator to merge afterward, and that buffer does not
  shrink with more workers, so the merge alone made resolution-9 counts up to
  2x slower with threads than without, and a resolution-11 count 7x slower than
  materializing. Gating the fused path to `threads=1` recovered the win but put
  it behind a flag no caller would think to reach for.

  What would qualify: partition the *output* rather than the input. Give each
  worker a disjoint contiguous ring range and let it scan every footprint whose
  `z_bounds()` overlaps that range, emitting only into its own slice. No
  per-worker grid buffer, no merge pass, no memory budget, no work-ratio
  constant, and the memory win lands on the default path at every thread count.

- **`vertex_offsets`, a pre-packed ragged batch argument.** Measured against the
  only alternative a caller has — splitting the buffer into one array per
  polygon and letting the ragged path concatenate it back — it was 1.44x at
  200,000 polygons *after* the ragged path was optimized twice (once to stop
  converting entry by entry, once to stop reading every shape twice), which took
  the sequence form from 432 to 271 ms and the ratio from 2.3x. For a caller
  holding a list of arrays, building the offsets to pass them ended up within
  noise of just passing the list, which is the honest test of whether the
  argument saved work or merely moved it. So it stood or fell on columnar
  interop — GeoArrow, Parquet, and database geometry columns are flat
  coordinates plus offsets. Removed, because no such caller exists here or is
  expected: the geometry this library is built for comes out of orbital
  propagation, not out of Parquet. Re-adding it would break nobody.

  Worth recording separately: the first measurement of this ran under
  `tracemalloc` and reported 5.4x. It penalizes the allocation-heavy side, which
  is exactly the side under test. Time without it; measure memory in a separate
  run.

- **Emitting ordered ranges from the polygon scan by assuming ring
  contiguity.** A dense count would become asymptotically better "once the scan
  emits ordered ranges rather than cells", and the cheap way there looks
  obvious: a cap meets every ring in one arc, a convex polygon is an
  intersection of half-spaces, so surely its ring intersection is one arc too.

  It is not. An intersection of arcs on a circle can be *two* arcs: the ring
  circle passes through the polygon on two sides without the polygon containing
  the pole. Attempted, and the differential suite rejected it in seconds — 36
  failures, all extra cells and none missing, the signature of merging two arcs
  into one. The counterexample is pinned as
  `test_a_convex_polygon_can_meet_one_ring_in_two_arcs`: a quad that at
  resolution 5 covers offsets 24-38 and 43-50 of one ring, with eleven uncovered
  cells between them.

  The exact version was then written and measured, and it is **slower**. Do not
  write it again.

  | footprint | cells each | per-cell scan | arc scan |
  | --- | --- | --- | --- |
  | 0.7 x 0.5 degrees, resolution 9 | 23 | 0.5 ms | 28.7 ms |
  | 4 x 3 degrees, resolution 9 | 763 | 4.3 ms | 41.5 ms |
  | 16 x 12 degrees, resolution 9 | 12,272 | 61.1 ms | 100.7 ms |
  | 16 x 12 degrees, resolution 11 | 196,459 | 705 ms | 774 ms |

  The O(cells) versus O(edges) argument is right about the asymptotics and wrong
  about the constants. Testing a cell centre is not expensive here: the scan
  advances around a ring by incremental rotation, so a candidate costs one
  rotation step and a few dot products, with no transcendental call at all. The
  analytic span costs an `atan2`, an `acos` and a `hypot` per edge per ring,
  plus a `sin_cos` for each endpoint confirmation — several hundred flops
  against roughly ten.

  The correctness work is also worse than it looks. `acos` is ill-conditioned
  exactly where a ring is nearly tangent to an edge plane, which is exactly
  where a cell centre lands on a boundary. Getting the differential suite from
  36 failures down to 2 took a guard term, a clamp at tangency, a bounded
  outward search, and two widened fallbacks — all heuristic, none with a
  provable bound, guarding a kernel whose failure mode is silently wrong cells.

  What remains true is narrower: emitting ranges would let a dense count consume
  endpoints and one prefix sum instead of materializing hits. That is a memory
  argument, and it does not need the scan to be faster — only to coalesce the
  runs it already finds.

## History

This record was consolidated from an earlier version that was amended eight
times during the branch. The superseded spellings, in case they appear in older
commits or discussion: `into=` became `reduce=`; `Count(cells=)`/`Stats()`
became `candidate_cells`/no token; `occupancy()`, `OccupancyRuns`,
`OccupancyStats`, `occupancy_runs()`, `occupancy_stats()`, and
`summarize_occupancy()` became `revisit()` and `RevisitStats`;
`count_caps_per_cell()`, `cover_footprint()`, `centers()`, `corners()`,
`Coverage.segment_sizes`, `Coverage.segment_indices()`,
`Coverage.segment_count`, `Coverage.filter_hits()`, and `vertex_offsets` were
removed. The reasoning for each removal is folded into the sections above.
