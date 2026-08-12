# Snapshot visibility from 10,771 service caps

This case study maps how many catalogued Starlink objects are geometrically
visible above a 25° elevation mask. It uses a pinned TLE catalog, propagates one
hour at one-minute cadence, and accumulates exact spherical caps directly into
a HEALPix count map.

The result describes catalog geometry, not operational Starlink service. It
does not include satellite status, beams, capacity, gateways, terrain, or
atmospheric effects.

## Result

```{raw} html
:file: ../assets/generated/communications-availability.html
```

The timing table is one wall-clock run on the documentation builder. It is
useful for seeing where this example spends time; it is not a controlled
benchmark.

## Method

| Parameter | Value |
| --- | ---: |
| Catalog | 10,771 objects, pinned 2026-07-29 |
| Propagation | Astroz 0.12.0, SGP4 |
| Analysis window | 1 hour from 2026-07-29 00:00 UTC |
| Cadence | 60 seconds, 61 samples |
| Minimum elevation | 25° |
| Grid | HEALPix resolution 6, 49,152 cells |

The TLE snapshot is committed with the repository. Fixing both the catalog and
analysis time keeps the example deterministic and removes network access from
the documentation build.

Astroz produces Earth-fixed positions for every object and timestamp:

```{literalinclude} ../../examples/communication_constellation.py
:language: python
:caption: examples/communication_constellation.py
:start-after: "--8<-- [start:communications-orbits]"
:end-before: "--8<-- [end:communications-orbits]"
```

At each timestamp, the 10,771 altitude-dependent service caps are accumulated
with one `count_caps_per_cell()` call:

```{literalinclude} ../../examples/communication_constellation.py
:language: python
:caption: examples/communication_constellation.py
:start-after: "--8<-- [start:communications-coverage]"
:end-before: "--8<-- [end:communications-coverage]"
```

The count operation consumes analytic RING spans. It does not build the roughly
2.25 million cap–cell IDs that explicit coverage would return at each sample.

## Run the example

```bash
pixi run --environment docs docs-communications
python examples/communication_constellation.py --output PATH
```

[Full source](https://github.com/JochimMaene/polypix/blob/main/examples/communication_constellation.py)
· [Pinned TLE snapshot](https://github.com/JochimMaene/polypix/blob/main/examples/data/starlink-2026-07-29.tle)
· [CelesTrak](https://celestrak.org/NORAD/elements/gp.php?GROUP=STARLINK&FORMAT=TLE)
