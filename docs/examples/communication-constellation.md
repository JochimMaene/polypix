# Communications constellation availability

How many satellites can each ground cell see at once? This example models a
Walker constellation of 500 satellites for one hour and maps the mean
simultaneous satellite count above a 25° elevation mask.

Everything below is produced by running the [example source][source] during this
documentation build.

```python exec="on" html="on" id="communication-constellation-result"
from examples.communication_constellation import documentation_html

print(documentation_html())
```

Timings are single wall-clock measurements from the documentation builder, not
controlled benchmarks. Plotting is listed separately because Matplotlib, not
Polypix, dominates it.

## Model

| Parameter | Value |
| --- | ---: |
| Satellites | 500 |
| Orbital planes | 20 |
| Orbit | Circular, 550 km |
| Inclination | 53° |
| Duration | 1 hour |
| Sampling cadence | 60 s |
| Time samples | 61 |
| Minimum elevation | 25° |
| Footprint vertices | 16 |
| HEALPix resolution | 6 (49,152 cells) |

The orbit is an idealized Walker distribution on a spherical rotating Earth. It
demonstrates coverage throughput; it is not an operational Starlink shell.

At each of the 61 timestamps a cell counts one satellite when its center falls
inside that satellite's service footprint. The plotted value is the mean of those
61 instantaneous counts — not the number of distinct satellites seen during the
hour.

A 25° minimum elevation defines a spherical cap around each sub-satellite point.
The cap radius follows from the 550 km altitude and a spherical Earth, and reaches
`cover_footprint()` as an inscribed 16-gon.

## Satellite positions

All 500 Earth-fixed sub-satellite vectors come from one vectorized operation:

```python title="examples/communication_constellation.py"
--8<-- "examples/communication_constellation.py:communications-orbits"
```

## Covering the footprints

All 500 footprints at one timestamp are covered as a single dense batch, reduced
with `bincount()`, then discarded before the next timestamp. Peak memory stays
flat across the full 30,500-footprint workload:

```python title="examples/communication_constellation.py"
--8<-- "examples/communication_constellation.py:communications-coverage"
```

## Running it

```bash
pixi run --environment docs docs-communications
python examples/communication_constellation.py --output PATH
```

[source]: https://github.com/JochimMaene/polypix/blob/main/examples/communication_constellation.py
