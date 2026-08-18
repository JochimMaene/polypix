# Earth-observation constellation coverage

How many distinct observations does each cell receive, and how long is the
revisit gap? This example models ten satellites over ten days as swept
one-minute sensor intervals and maps both answers.

Everything below is produced by running the [example source][source] during this
documentation build.

```python exec="on" html="on" id="earth-observation-constellation-result"
from examples.earth_observation_constellation import documentation_html

print(documentation_html())
```

Timings are single wall-clock measurements from the documentation builder, not
controlled benchmarks. Plotting is listed separately because Matplotlib, not
Polypix, dominates it.

## Model

| Parameter | Value |
| --- | ---: |
| Satellites | 10 |
| Orbital planes | 5 |
| Orbit | Circular, 550 km |
| Inclination | 53° |
| Duration | 10 days |
| Edge-sampling cadence | 60 s |
| Edge samples per satellite | 14,401 |
| Swept intervals per satellite | 14,400 |
| Ground-swath half-width | 7.5° |
| HEALPix resolution | 6 (49,152 cells) |

Spherical rotating Earth, constant spherical ground half-width. Operational
analysis should supply propagated sensor edges instead.

## What counts as an observation

For each pair of adjacent samples, `cover_sweep()` covers the swept quadrilateral
`[left[i], right[i], right[i+1], left[i+1]]`. This fills the motion between
samples rather than approximating a moving sensor with disconnected circles.

Consecutive covered intervals for the same satellite and cell are one
observation. A later pass is another observation, and satellites count
separately.

Revisit merges all satellites: overlapping or consecutive hits form one
constellation occupancy window, and revisit is the uncovered gap between
windows. The map shows mean gaps over the ten days, omits cells with fewer than
two windows, excludes gaps that extend past either end of the analysis window,
and samples time in one-minute bins.

## Swaths and coverage

The ten-day tracks and their paired sensor edges are vectorized:

```python title="examples/earth_observation_constellation.py"
--8<-- "examples/earth_observation_constellation.py:eo-swaths"
```

One `cover_sweep()` call then covers all 14,400 intervals of a satellite:

```python title="examples/earth_observation_constellation.py"
--8<-- "examples/earth_observation_constellation.py:eo-cover"
```

## Reducing to observations and revisit

`summarize_occupancy()` uses integer last-seen steps to detect source-specific
observation starts, merge simultaneous hits, and accumulate revisit gaps in a
native chronological pass:

```python title="examples/earth_observation_constellation.py"
--8<-- "examples/earth_observation_constellation.py:eo-reduce"
```

This avoids sorting nine million sparse observations, crossing the Python/NumPy
boundary once per interval and satellite, or allocating a dense
`(satellites, intervals, cells)` visibility cube. The returned summary is sparse;
the example scatters its 42,912 observed cells into dense plotting arrays.

## Running it

```bash
pixi run --environment docs docs-earth-observation
python examples/earth_observation_constellation.py \
    --observations-output PATH --revisit-output PATH
```

[source]: https://github.com/JochimMaene/polypix/blob/main/examples/earth_observation_constellation.py
