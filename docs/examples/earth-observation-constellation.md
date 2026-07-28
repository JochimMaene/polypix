# Earth-Observation Constellation Coverage

**How many distinct observations does each cell receive, and how long is the
revisit gap?** This example models ten satellites over ten days as swept
one-minute sensor intervals, then maps both answers.

Both maps and every number below are produced by executing the
[example source][source] during this documentation build.

```python exec="on" html="on" id="earth-observation-constellation-result"
from examples.earth_observation_constellation import render_documentation

print(render_documentation())
```

Timings are single wall-clock measurements from the documentation builder, not
controlled benchmarks. Plotting is listed separately because Matplotlib, not
Polypix, dominates it.

## What Is Being Counted

For every pair of adjacent one-minute samples, `cover_strip()` covers the swept
spherical quadrilateral:

```text
[left[i], right[i], right[i + 1], left[i + 1]]
```

This fills the motion between samples, rather than approximating a moving sensor
with disconnected circular footprints.

Consecutive covered intervals for the same satellite and cell form one
observation. A later pass starts another observation, and observations from
different satellites count separately.

Revisit merges all satellites: overlapping or consecutive hits form one
constellation access window, and revisit is the uncovered gap from the end of
one window to the start of the next. The map shows the mean gap measured during
the ten days and omits cells with fewer than two access windows. Gaps extending
past either end of the analysis window are excluded, and all boundaries are
quantized to one minute.

## Simulation Model

| Parameter | Value |
| --- | ---: |
| Satellites | 10 |
| Orbital planes | 5 |
| Orbit | Circular, 550 km |
| Inclination | 53° |
| Duration | 10 days |
| Edge-sampling cadence | 60 seconds |
| Edge samples per satellite | 14,401 |
| Swept intervals per satellite | 14,400 |
| Ground-swath half-width | 7.5° |
| HEALPix resolution | 6 |
| HEALPix cells | 49,152 |

The orbit model uses a spherical rotating Earth and a constant spherical ground
half-width. Operational analysis should supply propagated sensor edges instead.

## Build And Cover The Swaths

The complete ten-day tracks and their paired sensor edges are vectorized:

```python title="examples/earth_observation_constellation.py"
--8<-- "examples/earth_observation_constellation.py:eo-swaths"
```

One `cover_strip()` call then covers all 14,400 intervals of a satellite:

```python title="examples/earth_observation_constellation.py"
--8<-- "examples/earth_observation_constellation.py:eo-cover"
```

## Reduce Observations And Revisit

Every Polypix segment already contains unique cell IDs, so the reducer uses
integer last-seen timestamps to detect observation starts, merge simultaneous
satellite hits, and accumulate revisit gaps in one chronological pass:

```python title="examples/earth_observation_constellation.py"
--8<-- "examples/earth_observation_constellation.py:eo-reduce"
```

That avoids sorting nine million sparse observations, calling `unique()` per
interval, or allocating a dense `(satellites, intervals, cells)` visibility
cube.

## Run It

```bash
pixi run --environment docs docs-earth-observation
```

Or directly:

```bash
python examples/earth_observation_constellation.py
```

Pass `--observations-output` and `--revisit-output` to choose the map
destinations.

[source]: https://github.com/JochimMaene/polypix/blob/main/examples/earth_observation_constellation.py
