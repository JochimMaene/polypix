# How often does a satellite fly over?

This case study follows ten idealized satellites for ten days. Each one-minute
interval becomes a quadrilateral between two sampled swath edges. Polypix first
rasterizes those intervals, then accumulates constellation-wide per-cell
revisit statistics and converts their internal uncovered gaps to time.

The orbit and sensor model is kept deliberately crude: circular orbit,
spherical rotating Earth, constant 7.5° ground half-width. Real mission analysis
would feed in propagated sensor edges from its own model.

## Results

```{raw} html
:file: ../assets/generated/earth-observation.html
```

The timing table is one wall-clock run on the documentation builder, not a
controlled benchmark.

These are sampled coverage bins, not exact continuous access events. A hit says
that a cell center lies in the region swept during a one-minute interval; event
boundaries are uncertain at that cadence. The gap map includes only complete
end-to-start gaps between two runs. It excludes the leading and trailing edges
of the ten-day horizon, and cells with fewer than two runs have no gap value.
Never-observed cells are reported separately rather than assigned a finite
revisit time.

## Method

| Parameter | Value |
| --- | ---: |
| Constellation | 10 satellites in 5 planes |
| Orbit | Circular, 550 km, 53° inclination |
| Analysis window | 10 days |
| Edge cadence | 60 seconds |
| Sweep intervals | 144,000 |
| Ground half-width | 7.5° |
| Grid | HEALPix resolution 6, 49,152 cells |

The vectorized orbit model produces a left and right edge for every sample:

```{literalinclude} ../../examples/earth_observation_constellation.py
:language: python
:caption: examples/earth_observation_constellation.py
:start-after: "--8<-- [start:eo-swaths]"
:end-before: "--8<-- [end:eo-swaths]"
```

One `cover_sweep()` call covers all 14,400 intervals for a satellite:

```{literalinclude} ../../examples/earth_observation_constellation.py
:language: python
:caption: examples/earth_observation_constellation.py
:start-after: "--8<-- [start:eo-cover]"
:end-before: "--8<-- [end:eo-cover]"
```

Consecutive covered intervals from any satellite form one merged visit. The
individual visit boundaries are never built, so the example reads the per-cell
statistics straight out of the single pass, then maps the ordinal gaps to
seconds to derive both mean and maximum complete end-to-start gaps:

```{literalinclude} ../../examples/earth_observation_constellation.py
:language: python
:caption: examples/earth_observation_constellation.py
:start-after: "--8<-- [start:eo-reduce]"
:end-before: "--8<-- [end:eo-reduce]"
```

[`revisit()`](../api.md#revisit) never expands and sorts the nine million
interval–cell hits as an event table, and it never builds the roughly nine
million visit boundaries either: working state is one accumulator per observed
cell. `first_start` and `last_stop` carry the observed window, so a mission
workflow can add horizon-edge gaps or periodic wraparound from this result.

Percentiles, minimum-duration filtering, and per-visit timestamps would need
the individual boundaries, which this result deliberately does not carry -
materializing them approaches one boundary pair per hit on exactly this shape
of workload. Reconstruct them downstream from the coverage if a study needs
them.

## Run the example

```bash
pixi run --environment docs docs-earth-observation
python examples/earth_observation_constellation.py \
    --observations-output PATH --revisit-output PATH
```

[Full source](https://github.com/JochimMaene/polypix/blob/main/examples/earth_observation_constellation.py)
