# Exact Caps and Segmented Occupancy Reduction

Status: **accepted; measured primary-workload primitives**

## Decision

Polypix admits two focused operations beyond explicit polygon and sweep
materialization:

- exact spherical-cap coverage, with fused per-cell cap counts;
- ordinal source-run and merged-gap summaries over segmented
  `Coverage` sources.

The public API remains NumPy-first and type-stable. `cover_cap()` returns the
existing `Coverage`; `count_caps_per_cell()` always returns an integer array;
`summarize_occupancy()` always returns one sparse `OccupancySummary`.

This decision accepts the measured implementations; it does not freeze their
exact placement or naming for 1.0. The later cross-domain
[API Surface Beyond the Constellation Examples](api-surface-beyond-constellations.md)
review confirmed exact caps and region-to-cell accumulation as broad
primitives, while marking the exact `summarize_occupancy()` fields as provisional
until independent aligned-bin occupancy workloads validate them.

## Context

The executable constellation examples exposed two end-to-end bottlenecks that
the earlier geometry-only product boundary could not address.

The pinned Starlink workload constructed 657,031 inscribed 16-gons for regions
that were inherently circular, materialized 133,831,629 polygon-cell pairs, and
then discarded those IDs after `bincount()`. Its recorded analysis took 4.652 s:
3.061 s in polygon coverage, 0.834 s constructing vertices, and 0.629 s reducing
the materialized output. Even infinitely fast polygon coverage could not reach
an order-of-magnitude end-to-end gain.

The Earth-observation workload materialized 9,282,060 sweep interval-cell pairs
in 0.342 s, then spent 4.246 s crossing between Python and small NumPy operations
for 14,400 intervals and ten sources. Sorting the events in bulk was slower and
used substantially more memory. The useful operation was a deterministic
segmented state machine, not another geometry strategy.

Both operations pass the project's feature-admission test: satellite coverage
is the primary workload, users cannot reproduce the fused memory behavior with
a small downstream NumPy expression, and the concepts remain bounded.

## Exact cap semantics and implementation

A cap is a normalized center vector plus a finite angular radius in `[0, pi]`.
A HEALPix cell is covered when the angular center separation is no greater than
the radius plus the documented angular tolerance. A stable squared-chord
predicate avoids subtracting dot products and cosines close to one.

At each HEALPix latitude ring, the cap kernel solves the continuous longitude
arc analytically, converts it to one or two ascending RING spans, and corrects
numerically ambiguous endpoints with the definitive chord predicate. Explicit
coverage expands those spans. Dense per-cell counts apply signed updates at
span endpoints and take one prefix sum, so repeated cap-cell IDs never exist.
Requested-cell count mode evaluates only the positional query cells and retains
their order and duplicates.

This is deliberately exact cap geometry. It is not an optimization that
silently substitutes a cap for a caller's polygon. Replacing the Starlink
example's inscribed 16-gons therefore changes its cell counts slightly and is
documented as a model improvement.

## Occupancy-summary semantics and implementation

Each input `Coverage` is one independent source; matching segment indices
across sources represent the same aligned, ordered occupancy bin. An
observation is a maximal consecutive run for one source and cell. Revisit first
unions all sources, then measures the number of uncovered bins between
consecutive merged windows. Leading and trailing unobserved bins are excluded.

The reducer operates on ordinal bins and does not require a cadence or equal bin
durations. Equal duration is a caller assertion only when converting gap-step
counts to physical time.

The result contains ascending observed cells, source-run counts, and merged-gap
sums and counts. It carries no timestamps, cadence, or physical units. A
bounded dense native state machine serves moderate resolutions; a sparse map
strategy serves large grids without allocating by global cell count.

The reducer is intentionally separate from `cover_sweep()`. Coverage remains a
reusable, independently testable seam, and summaries can combine any segmented
Polypix result rather than one EO geometry path.

## Measured result

On the eight-logical-CPU development host, after warmup:

| Workload | Before | After | Observed gain |
| --- | ---: | ---: | ---: |
| Starlink complete analysis | 4.652 s | 0.388 s median | 12.0x |
| Starlink Polypix/count stage | 3.061 s | 0.301 s median | 10.2x |
| Earth-observation complete analysis | 4.676 s | 0.251 s median | 18.6x |
| Earth-observation occupancy reduction | 4.246 s | 0.066 s median | 64.3x |
| Earth-observation sweep coverage | 0.342 s | 0.151 s median | 2.3x |

These are repository-maintenance measurements, not cross-machine or
cross-library performance claims. The before values are the generated docs
measurements that triggered the work; the after values are medians of fourteen
analysis runs from the same host. Exact timings vary with load and toolchain.

## Correctness evidence

Cap coverage is checked against an independent brute-force HEALPix center
oracle across random centers, radii from zero through pi, poles, seams, exact
boundaries, a full sphere, candidates, and thread counts. Fused dense and query
counts are checked against materialized `cover_cap()` plus `bincount()`.

Occupancy summaries are checked against a Python state-machine oracle over
fixed-seed randomized sources and segments, along with explicit source-switch,
simultaneous-hit, gap, empty, malformed, and resolution-29 sparse cases. The EO
example retains its previous hand-calculated fixture and output checks.

## Rejected alternatives

- More polygon micro-optimization could not remove cap approximation,
  footprint construction, output materialization, or `bincount()`.
- A conditional `cover_cap(..., output="counts")` would make the return type
  unstable; a separate verb is clearer.
- An EO-specific helper would improve a benchmark without defining a reusable
  product concept. Ordinal segmented occupancy is the narrow general seam.
- Sorting nine million occupancy events used more time and memory than the
  original streaming reducer.
- Fusing occupancy reduction into sweep coverage would couple geometry to one
  downstream interpretation and make independent testing harder.

## Scope boundary

These admissions do not make Polypix an orbit, time, or raster-analysis
library. Weights, physical timestamps, variable step duration, window lists,
map algebra, arbitrary reducers, and geometry generation remain outside the
product. Additional aggregation must independently pass feature admission.
