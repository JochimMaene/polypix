# Starlink snapshot visibility

How many catalogued Starlink objects can each ground cell geometrically see at
once? This example propagates a permanent [CelesTrak `STARLINK` snapshot][data]
for one hour and maps the mean simultaneous object count above a 25° elevation
mask.

This is a geometric visibility study, not a map of operational Starlink
service. It does not model operational status, beam assignment, capacity,
gateways, terrain, atmosphere, or user terminals.

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
| Catalogued objects | 10,771 |
| Catalog snapshot | CelesTrak `STARLINK`, 2026-07-29 |
| Propagator | Astroz 0.12.0, SGP4 |
| Analysis start | 2026-07-29 00:00 UTC |
| Duration | 1 hour |
| Sampling cadence | 60 s |
| Time samples | 61 |
| Minimum elevation | 25° |
| Visibility geometry | Exact spherical caps |
| HEALPix resolution | 6 (49,152 cells) |

The committed [TLE snapshot][snapshot] is the permanent input to this example.
The documentation build never contacts CelesTrak, and the snapshot is not
intended to be refreshed. Fixing both the catalog and analysis time makes the
map deterministic and keeps a third-party network service out of CI.

At each of the 61 timestamps a cell counts one satellite when its center falls
inside that object's visibility footprint. The plotted value is the mean of
those 61 instantaneous counts — not the number of distinct objects seen during
the hour.

A 25° minimum elevation defines a spherical cap around each sub-satellite
point. Each exact cap radius follows from that object's propagated
Earth-centered distance and a spherical Earth.

## Satellite positions

Astroz parses the pinned TLE catalog and propagates all 61 timestamps into one
small dense batch of Earth-fixed positions:

```python title="examples/communication_constellation.py"
--8<-- "examples/communication_constellation.py:communications-orbits"
```

## Counting visibility

`count_caps_per_cell()` processes all 10,771 caps at one timestamp as a batch.
It accumulates analytic RING spans directly into a 49,152-cell count array,
without materializing the roughly 2.25 million repeated cap-cell IDs at that
timestamp. Peak analysis memory therefore stays small across all 657,031 caps:

```python title="examples/communication_constellation.py"
--8<-- "examples/communication_constellation.py:communications-coverage"
```

## Running it

```bash
pixi run --environment docs docs-communications
python examples/communication_constellation.py --output PATH
```

[source]: https://github.com/JochimMaene/polypix/blob/main/examples/communication_constellation.py
[snapshot]: https://github.com/JochimMaene/polypix/blob/main/examples/data/starlink-2026-07-29.tle
[data]: https://celestrak.org/NORAD/elements/gp.php?GROUP=STARLINK&FORMAT=TLE
