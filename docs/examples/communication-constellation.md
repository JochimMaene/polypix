# Communications Constellation Availability

**How many satellites can each ground cell see at the same time?** This example
models a Starlink-like Walker constellation of 500 satellites for one hour and
maps the mean simultaneous satellite count above a 25° minimum-elevation mask.

The map and every number below are produced by executing the
[example source][source] during this documentation build.

```python exec="on" html="on" id="communication-constellation-result"
from examples.communication_constellation import render_documentation

print(render_documentation())
```

Timings are single wall-clock measurements from the documentation builder, not
controlled benchmarks. Plotting is listed separately because Matplotlib, not
Polypix, dominates it.

## What Is Being Counted

Satellite positions are sampled once per minute, including both ends of the
one-hour window. At each of the 61 timestamps, a cell counts one satellite when
its center falls inside that satellite's service footprint. The plotted value is
the mean of those 61 instantaneous counts — not the number of distinct
satellites seen at some point during the hour.

A circular field of regard suits this communications question: a 25° minimum
elevation defines a spherical cap around each sub-satellite point. The cap
radius follows from the 550 km altitude and a spherical Earth, and is passed to
`cover_footprint()` as an inscribed 16-sided convex spherical polygon.

## Simulation Model

| Parameter | Value |
| --- | ---: |
| Satellites | 500 |
| Orbital planes | 20 |
| Orbit | Circular, 550 km |
| Inclination | 53° |
| Duration | 1 hour |
| Sampling cadence | 60 seconds |
| Time samples | 61 |
| Minimum elevation | 25° |
| Service-footprint vertices | 16 |
| HEALPix resolution | 6 |
| HEALPix cells | 49,152 |

The orbit is an idealized Walker-like distribution on a spherical rotating
Earth. It demonstrates coverage throughput; it does not reproduce an operational
Starlink shell.

## Generate Satellite Positions

All 500 Earth-fixed sub-satellite vectors come from one vectorized NumPy
operation:

```python title="examples/communication_constellation.py"
--8<-- "examples/communication_constellation.py:communications-orbits"
```

## Cover Service Footprints

All 500 footprints at one timestamp are covered as a single dense batch. The
result is reduced with `bincount()` and discarded before the next timestamp,
so peak memory stays flat across the full 30,500-footprint workload:

```python title="examples/communication_constellation.py"
--8<-- "examples/communication_constellation.py:communications-coverage"
```

## Run It

```bash
pixi run --environment docs docs-communications
```

Or directly:

```bash
python examples/communication_constellation.py
```

Pass `--output PATH` to choose the map destination.

[source]: https://github.com/JochimMaene/polypix/blob/main/examples/communication_constellation.py
