# Benchmarks

How quickly can each library count visible satellites or combine observation
footprints? These benchmarks run both workflows on shared synthetic geometry.
The charts show throughput, variation between runs, and the accuracy cost of
using different grids.

## Methodology

- **Machine:** Intel Core i7-1165G7, eight logical CPUs, Linux under WSL2.
  This is a development machine, not an isolated benchmark server.
- **Inputs:** deterministic geometry, seed 20261109. Circles have radii from
  3° to 12°; observation tracks have four overlapping convex footprints each.
  No orbit propagation, file reads, or network access occurs in the timed work.
- **Repetitions:** five fresh processes per case. Each records the first call
  and five subsequent calls. Warm throughput uses the median time within each
  process, then the median across processes; whiskers show the process range.
- **Timing:** shape conversion, coverage, accumulation or union, and lookup at
  20,000 shared evaluation points. Imports, input generation, point-to-cell
  indexing, and analytic validation are outside the timer.
- **Threads:** Polypix is shown serially and with automatic threading. Other
  packages retain native defaults; BLAS and OpenMP are limited to one thread.
  The automatic runs are not a comparison of equal CPU budgets.
- **Validation:** healpy and HPGeom must match Polypix's complete center-selected
  RING output. Independent analytic containment measures sampled error for all
  backends. Failures stop the matrix rather than silently disappearing.

The presentation follows the grouped charts and explicit methodology of
[Litestar's benchmarks](https://docs.litestar.dev/2/benchmarks.html).
Unlike that setup, these results were not collected on dedicated hardware.

## Results

Select a chart to open it at full size.

### Satellite visibility counts

Count how many of **10,000 coverage circles** contain each ground cell at one
instant. This models the coverage step in a large communications constellation;
it does not simulate satellites or service capacity.

```{figure} assets/benchmarks/visibility-throughput.svg
:target: _images/visibility-throughput.svg
:alt: Grouped throughput bars for Polypix, healpy and HPGeom at three resolutions, with first-call and warmed panels and min–max whiskers.

Visibility workloads completed per second. Higher is better. All four series
produce identical HEALPix count maps at each order.
```

### Accumulated Earth observation

Combine **4,000 footprints** from 1,000 short tracks into a covered-cell union,
then estimate the fraction of Earth observed using the shared sample points.
The target region is the whole sphere; all footprint edges are great-circle arcs.

```{figure} assets/benchmarks/observation-throughput.svg
:target: _images/observation-throughput.svg
:alt: Grouped throughput bars for the union of 4,000 footprints, comparing first-call and warmed results across three resolutions.

Observation workloads completed per second. Higher is better. One workload
includes the full footprint batch and union, not one polygon query.
```

### Speed at different accuracies

A grid with similar cell area is not necessarily equally accurate. These smaller
cases use **128 circles or 512 footprints**, so every backend can be compared
across three requested HEALPix orders. The labels identify those orders; H3 and
S2 use the nearest global cell count on their own resolution scales.

```{figure} assets/benchmarks/visibility-accuracy.svg
:target: _images/visibility-accuracy.svg
:alt: Runtime against mean absolute visibility-count error, showing three resolutions for each library.

Visibility: lower and farther left is better. Error is the mean absolute
count difference from analytic spherical containment at 20,000 points.
```

```{figure} assets/benchmarks/observation-accuracy.svg
:target: _images/observation-accuracy.svg
:alt: Runtime against observation classification error for all libraries, with requested orders labeled.

Observation: lower and farther left is better. Error is the fraction of sample
points whose covered/uncovered classification differs from analytic containment.
```

Polypix, healpy, and HPGeom select cell centers. MOCPy and S2 return outer
coverings, which deliberately include boundary cells and can overestimate area.
H3 selects polygon centers; circles are represented by 128-vertex polygons.
These are different outputs, so the accuracy charts are application comparisons,
not an equivalent-output speed ranking. Small footprints can disappear under
center selection; increasing resolution costs time and memory but improves detail.

MOCPy uses its native batch cone and polygon constructors and native union.
Its visibility counts are accumulated by querying each MOC at the sample points;
that cost is included. S2 uses its C++ binding and fixed-level region coverer.
healpy and HPGeom use their public per-region query APIs followed by NumPy
accumulation. Alternative representations or a different target workload may
change the relative performance substantially.

## Reproduce and inspect

The {download}`results table <assets/benchmarks/results.csv>` contains exact timing ranges,
process memory, error, and grid sizes. [Raw JSON and environment records](https://github.com/JochimMaene/polypix/tree/main/benchmarks/results)
include package versions, hardware, seed, source revision, and harness hash.
Peak RSS includes the interpreter, imports, inputs, and native allocations; it
is not incremental algorithm memory. It is captured before analytic validation.

Follow the [benchmark setup instructions](https://github.com/JochimMaene/polypix/blob/main/benchmarks/competitors.md), then run:

```sh
PYTHONPATH=. python benchmarks/run_comparisons.py
MPLBACKEND=Agg python benchmarks/plot_comparisons.py
```

The plot script uses Matplotlib from the documentation environment. Charts are
checked-in SVGs: building the docs never runs the competitor matrix.

The charts use seed 20261109. Additional runs with seeds 20261110 and 20261111,
three processes and three warmed calls per process, confirmed exact HEALPix
agreement at all three orders. Polypix remained faster than the tested healpy
and HPGeom paths for both small workflows; serial versus automatic rankings
varied. These checks are recorded in the raw-results `seeds/` directory and are
not pooled into the charts. The large workloads were only run with the chart seed.

These measurements cover three synthetic seeds, three resolutions, and one machine.
They do not establish performance for every polygon shape, constellation,
thread budget, or hardware platform. The large-workload charts cover only the
equivalent HEALPix outputs; cross-grid scaling at that size remains unmeasured.
