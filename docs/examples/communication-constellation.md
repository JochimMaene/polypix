# How many Starlink satellites can you see?

Stand outside for an hour and count the Starlink satellites that climb above 25°
elevation. Now do it everywhere at once: all 10,771 catalogued objects,
propagated at one-minute cadence and counted cell by cell.

How many you see depends strongly on your latitude. Because the shells are
inclined, the count peaks in two bands between 40° and 50° north and south, where
70 to 90 objects are in view at once. It falls to around 37 over the equator, and
to about 20 near the poles.

One caveat before you read the map. An object counts when it is geometrically
above the elevation mask, whether or not it is operational or has any capacity to
spare. Satellite status, beams,
gateways, terrain, and the atmosphere are all outside the scope of the example.

## Result

```{include} ../assets/generated/communications-availability.md
:parser: myst
```

The timings come from a single wall-clock run on the machine that built this
page, so read them for the shape of the work and not as a controlled benchmark.
Nearly all of it is the covering calls; the propagation and the accumulation on
either side are small.

## Method

| Parameter | Value |
| --- | ---: |
| Catalog | 10,771 objects, pinned 2026-07-29 |
| Propagation | Astroz 0.12.0, SGP4 |
| Analysis window | 1 hour from 2026-07-29 00:00 UTC |
| Cadence | 60 seconds, 61 samples |
| Minimum elevation | 25° |
| Grid | HEALPix resolution 6, 49,152 cells |

Resolution 6 puts roughly 100 km between cell centres, which is fine enough
here. A satellite 550 km up with a 25° mask serves a circle about 1,900 km
across, so one cap lands on a couple of hundred cells; across this run the
average was 209. Remember that a cell is counted when its own centre falls
inside a cap, which makes the map a grid of sample points and not an area
intersection. [Center-sampled coverage](../concepts.md#center-sampled-coverage)
has the details, and [Picking one](../resolutions.md#picking-one) covers the
choice of resolution.

The TLE snapshot is committed to the repository. Pinning both the catalogue and
the analysis time keeps the example reproducible, and keeps the documentation
build off the network.

Astroz parses the snapshot and propagates every object to every timestamp,
giving Earth-fixed positions in kilometres. Each of those positions then has to
become a region. A satellite is visible from
everywhere inside a circle on the ground, and how wide that circle is depends on
how high the satellite is, so the radius comes out of the orbit radius and the
elevation mask:

```{literalinclude} ../../examples/constellation.py
:language: python
:caption: examples/constellation.py
:start-after: "--8<-- [start:service-caps]"
:end-before: "--8<-- [end:service-caps]"
```

That is the whole conversion from a propagated state to something Polypix
accepts: an array of centre directions and an array of radii in radians. All
10,771 of them at one timestamp then go into a single call:

```{literalinclude} ../../examples/communication_constellation.py
:language: python
:caption: examples/communication_constellation.py
:start-after: "--8<-- [start:communications-coverage]"
:end-before: "--8<-- [end:communications-coverage]"
```

`reduce=px.Count()` is what makes this cheap. Asking for counts instead of
membership lets the cap kernel consume analytic RING spans, so it never builds
the cap-cell pairs at all. Over the hour that is 657,031 caps and 137 million
cap-cell hits, none of which is ever stored: the running total is one array of
49,152 integers.

## Run the example

```bash
pixi run --environment docs docs-communications
python examples/communication_constellation.py --output PATH
```

[Full example source](https://github.com/JochimMaene/polypix/blob/main/examples/communication_constellation.py)

## Adapting it

The scenario is four constants and a catalogue, so the interesting variations
are cheap:

- Change `MINIMUM_ELEVATION_RAD` for a different mask. The cap radius follows
  from it, and no other part of the example needs to know.
- Swap the TLE file for your own catalogue, or propagate with whatever you
  already use. Polypix only ever sees centres and radii.
- Restrict the answer to a service area by passing `candidate_cells=` to
  `cover_cap()`, which is what you want when the grid is large and the region
  of interest is not.
- Raise `HEALPIX_RESOLUTION` for finer cells, remembering that each step up
  quadruples the count.
