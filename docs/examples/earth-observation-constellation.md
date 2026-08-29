# What is Sentinel-2's revisit time?

Sentinel-2 images a 290 km swath from a sun-synchronous orbit, and the three
spacecraft between them pass over any given place every few days. This case
study measures how long that wait really is, everywhere, from a pinned
catalogue propagated for 14 days.

A pass counts here whenever the swath crosses a cell, by night as readily as by
day, while the Multi-Spectral Instrument images only in daylight. Illumination, cloud, and the acquisition plan are all outside the
scope of this example, so the maps show where a pass was possible and not where
an image was taken.

```{include} ../assets/generated/earth-observation.md
:parser: myst
```

Cells near the equator wait about 37 hours between overflights, mid latitudes about 27, and the 60° to 70° band about 15, since the orbit tracks crowd together towards the poles. The longest single wait is always worse than the average: 61 hours for a typical cell, and 109 hours at the worst spot on the map.

The maps look grainy because of the grid. At 50 km cells and a 290 km swath,
whether a pass catches a particular cell centre varies from one cell to the
next, and a near miss moves that cell's average by hours.

`revisit()` reports only complete waits between two overflights, so a cell needs
at least two before it reports anything. Fourteen days is enough here: the median
cell is overflown ten times, and every cell the swath reaches at all it reaches
at least twice. Above 83° latitude it reaches nothing, because the orbit is
inclined 98.6° and the swath is only 290 km wide.

## Performance

The [pinned CodSpeed run][codspeed-run] measures the EO-shaped `revisit()`
reference workload at about **75 ms** in simulation mode. That deterministic
[benchmark][reference-benchmark] processes ten sources and 9.28 million hits;
this case study uses three sources and 2.7 million hits. It is a regression
reference for the same operation, not a complete-analysis wall-clock estimate.

For context, one [GitHub-hosted documentation build][docs-build-run] of the
complete example at the same commit produced this pipeline profile:

| Stage | Time |
| --- | ---: |
| SGP4 propagation and swath edges | 10 ms |
| 3 `cover_sweep()` calls | **35 ms** |
| `revisit()` and gap conversion | 22 ms |
| Complete analysis | 67 ms |

Those wall-clock figures show the relative cost of the end-to-end stages; they
are pinned to that build and are not the regression benchmark.

## Method

| Parameter | Value |
| --- | ---: |
| Spacecraft | Sentinel-2A, 2B, 2C |
| Catalogue | CelesTrak, pinned 2026-08-24 |
| Propagation | Astroz 0.12.0, SGP4 |
| Analysis window | 14 days from 2026-08-24 00:00 UTC |
| Edge cadence | 60 seconds |
| Sweep intervals | 60,480 |
| Swath width | 290 km |
| Grid | HEALPix resolution 7, 196,608 cells |

Resolution 7 puts about 50 km between cell centres, so a 290 km swath is a few
cells wide. Each one-minute interval is a quadrilateral about 400 km along track
by 290 km across, and that cadence sets the precision of every wait reported
here.

First we propagate the snapshot and take each spacecraft's sub-satellite point,
from which the swath edges follow:

```{literalinclude} ../../examples/earth_observation_constellation.py
:language: python
:caption: examples/earth_observation_constellation.py
:start-after: "--8<-- [start:eo-swaths]"
:end-before: "--8<-- [end:eo-swaths]"
```

Each spacecraft's 20,160 intervals then go into one `cover_sweep()` call, which
joins consecutive samples by the shorter great-circle arc:

```{literalinclude} ../../examples/earth_observation_constellation.py
:language: python
:caption: examples/earth_observation_constellation.py
:start-after: "--8<-- [start:eo-cover]"
:end-before: "--8<-- [end:eo-cover]"
```

The three results share a resolution and a segment count, so `revisit()` reads
them as aligned timelines. Consecutive covered intervals from any spacecraft
count as one overflight:

```{literalinclude} ../../examples/earth_observation_constellation.py
:language: python
:caption: examples/earth_observation_constellation.py
:start-after: "--8<-- [start:eo-reduce]"
:end-before: "--8<-- [end:eo-reduce]"
```

That walks the 2.7 million interval-cell hits once, keeping one accumulator per
cell and never building the individual overflights. It counts in bins, not
seconds, so we convert to hours ourselves.


## Run the example

```bash
pixi run --environment docs docs-earth-observation
python examples/earth_observation_constellation.py \
    --mean-gap-output PATH --worst-gap-output PATH
```

[Full source](https://github.com/JochimMaene/polypix/blob/main/examples/earth_observation_constellation.py)

[reference-benchmark]: https://github.com/JochimMaene/polypix/blob/bf26c009e6529367e1165cecbe7dbda486b5479c/benchmarks/test_polypix_benchmarks.py#L956
[codspeed-run]: https://app.codspeed.io/JochimMaene/polypix/runs/compare/6a91b972915aa37294773c71..6a92bc48fd5591856db14fc8?q=revisit
[docs-build-run]: https://github.com/JochimMaene/polypix/actions/runs/33249059091
