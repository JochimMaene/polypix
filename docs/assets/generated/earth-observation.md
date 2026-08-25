```{figure} ../assets/generated/earth-observation-revisit.png
:alt: Global map of the mean time between Sentinel-2 overflights
:figclass: example-figure

Mean time between overflights over 14 days. Consecutive
one-minute intervals covered by any of the three spacecraft count as one
overflight.
```

```{figure} ../assets/generated/earth-observation-worst-gap.png
:alt: Global map of the longest wait between Sentinel-2 overflights
:figclass: example-figure

The longest single wait in the same 14 days. Waits running past
the start or the end of the window are excluded, and 1,520 cells
above 83° latitude are never overflown at all.
```

```{list-table} One measured run
:header-rows: 1
:class: example-timings
:widths: 70 30

* - Stage
  - Time
* - SGP4 propagation and swath edges
  - 16 ms
* - 3 `cover_sweep()` calls
  - **54 ms**
* - `revisit()` and gap conversion
  - 71 ms
* - Complete analysis
  - 141 ms
```
