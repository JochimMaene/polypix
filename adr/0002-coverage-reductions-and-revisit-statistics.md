# ADR-0002: Use reducers for coverage results and revisit statistics

- Status: Accepted
- Date: 2026-08-21

## Context

After covering a set of regions, callers usually want either the cells in each
segment or a value accumulated per cell. Separate functions for cap counts,
polygon counts, sweep counts, and weighted variants would make the API larger
without making it clearer.

At high resolutions, building a whole grid for a small query is wasteful.
Revisit analysis has the same issue when the caller wants a summary rather than
every interval for every cell.

## Decision

Coverage operations take `reduce=`. Without it they return `Coverage`.
`Count()` and `Sum(values)` return per-cell arrays, and `Coverage.reduce()` uses
the same operations on coverage that is already materialized.

`candidate_cells` stays on the coverage operation. For ordinary coverage it
restricts the result. With a reducer it describes the output positions, so the
caller’s order and duplicate queries are preserved.

`revisit()` returns `RevisitStats` for an ordered, thresholded coverage axis. It
reports run counts, internal gap totals and maxima, and the observed window
bounds. Its bins are ordinal. The caller supplies timestamps and decides how to
treat leading, trailing, or cyclic gaps.

The implementation may fuse a reduction or calculate it from materialized
coverage. That choice stays private; the call and its result do not change.

## Consequences

Caps, polygons, and sweeps use the same reduction vocabulary. A selected query
at a high resolution does not need a dense global array. Code that needs every
interval or owns physical time has to keep those things outside Polypix.

## Alternatives considered

- **One reduction function per geometry:** too many nearly identical functions
  for one underlying operation.
- **A flag that changes the return type:** less clear than a reducer that names
  the result the caller wants.
- **Public lossless intervals:** current callers need the statistics, not every
  interval. They can be added later without changing this interface.
