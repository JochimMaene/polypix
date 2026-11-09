# Benchmarks

## Methodology

- Run on an Intel Core i7-1165G7 (eight logical CPUs), under WSL2.
- Each library receives the same generated geometry. Timings include coverage,
  accumulation, and lookup at 20,000 ground points; input generation and imports
  are excluded.
- Five processes per case, each measuring its first call and five warmed calls.
  Bars show median throughput; whiskers show the range between processes.
- Polypix runs with one thread and with automatic threading. Other libraries use
  their native defaults; BLAS and OpenMP are limited to one thread.
- healpy and HPGeom results are checked against Polypix's complete cell maps.

[Benchmark code and setup](https://github.com/JochimMaene/polypix/blob/main/benchmarks/competitors.md)
· {download}`Results (CSV) <assets/benchmarks/results.csv>`
· [Raw measurements](https://github.com/JochimMaene/polypix/tree/main/benchmarks/results)

## Results

### Satellite visibility

Count overlapping coverage from 10,000 circles at one timestamp.

```{figure} assets/benchmarks/visibility-throughput.svg
:target: _images/visibility-throughput.svg
:alt: Throughput for 10,000 coverage circles at three HEALPix resolutions, comparing first-call and warmed performance.
```

### Earth observation

Combine 4,000 footprints into a covered-cell union and estimate the fraction
of Earth observed.

```{figure} assets/benchmarks/observation-throughput.svg
:target: _images/observation-throughput.svg
:alt: Throughput for the union of 4,000 footprints at three HEALPix resolutions, comparing first-call and warmed performance.
```

### Accuracy across grids

These cases use 128 circles or 512 footprints. H3 and S2 resolutions are chosen
by their nearest global cell count to HEALPix. Labels show the requested HEALPix
order; error is measured against spherical containment at 20,000 shared points.

```{note}
Polypix, healpy, and HPGeom select cell centers. MOCPy and S2 include boundary
cells, which can overestimate coverage. H3 approximates circles with 128-vertex
polygons. These results compare different coverage semantics.
```

```{figure} assets/benchmarks/visibility-accuracy.svg
:target: _images/visibility-accuracy.svg
:alt: Runtime versus mean absolute visibility-count error for each library.

Visibility: mean absolute error in the number of covering circles.
```

```{figure} assets/benchmarks/observation-accuracy.svg
:target: _images/observation-accuracy.svg
:alt: Runtime versus the fraction of incorrectly classified observation points.

Observation: fraction of points incorrectly classified as covered or uncovered.
```

## Interpreting the results

The throughput charts compare identical HEALPix outputs. The accuracy charts
compare each library's grid and coverage rules; MOCPy's visibility timing also
includes querying each coverage map to accumulate counts at the sample points.

These measurements come from one development machine. Performance depends on
region size, resolution, thread settings, and the result your application needs.
