```{figure} ../assets/generated/communications-availability.png
:alt: Global map of mean catalogued Starlink objects geometrically visible
:figclass: example-figure

Mean simultaneous catalogued Starlink objects geometrically visible above a 25°
elevation mask, sampled once per minute for one hour.
```

```{list-table} One measured run
:header-rows: 1
:class: example-timings
:widths: 70 30

* - Stage
  - Time
* - Parse pinned TLE snapshot
  - 53 ms
* - SGP4 propagation
  - 11 ms
* - 61 cap builds and `cover_cap(reduce=Count())` calls
  - **514 ms**
* - Complete analysis
  - 591 ms
```
