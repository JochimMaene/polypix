---
html_theme.sidebar_secondary.remove: true
---

{.polypix-title}
# Polypix

:::{container} polypix-hero
<p class="tagline">Which grid cells does this region cover? Answered for a whole batch in one call.</p>

<p class="scope">Give it circles, polygons, or a swept sensor path. Get back the HEALPix cells they cover, as NumPy arrays.</p>

<p class="polypix-actions"><a href="guide.html">Get started</a><a href="api.html">API reference</a></p>

<div class="polypix-install"><span>pip install polypix</span></div>
:::

Say you have ten thousand satellite footprints, or a sky survey's worth of
telescope fields, and you need the grid cells each one lands on. Looping in
Python is slow, and the geometry gets awkward at the poles and the date line.

Two circles, one with a 5° radius and one with 8°:

```{literalinclude} ../examples/docs_diagrams.py
:language: python
:dedent: 4
:start-after: "--8<-- [start:quickstart]"
:end-before: "--8<-- [end:quickstart]"
```

That covers 1,502 and 3,824 cells. `coverage[0]` holds the IDs for the first
circle, `coverage[1]` the second.

New to HEALPix? This is the grid those IDs refer to. It starts as 12 equal-area
cells and splits each one into four at every step up in resolution:

```{figure} assets/generated/sphere-levels.png
:alt: The same sphere partitioned at HEALPix resolutions 0 to 3, cell count rising from 12 to 768.
:width: 100%
:align: center

Polypix goes to resolution 29, where a cell is about 12 mm across on the ground.
[Resolutions](resolutions.md) has the whole table.
```

Polypix stops there, on purpose. Orbits, pointing, ellipsoid intersection: that
stays in your code, or in the libraries you already use.

## Why Polypix?

- **Fast**: a native kernel that releases the GIL, so one call uses every core.
- **Batch-first**: 10,000 regions come back as two arrays, not 10,000 objects.
- **No special cases**: the poles and the date line are ordinary 3D vectors.
- **No wasted work**: if you only need counts per cell, it never builds the pairs.
- **Small**: NumPy is the only dependency.

## Examples

Both run from pinned inputs every time these docs are built, so the maps and
timings cannot drift from the code.

<div class="example-gallery">
  <a class="example-card" href="examples/communication-constellation.html">
    <img src="generated/communications-availability.png" alt="Global Starlink visibility map">
    <div>
      <h2>Snapshot visibility</h2>
      <p>657,031 exact caps reduced directly to per-cell counts.</p>
    </div>
  </a>
  <a class="example-card" href="examples/earth-observation-constellation.html">
    <img src="generated/earth-observation-count.png" alt="Global Earth-observation count map">
    <div>
      <h2>Earth-observation revisit</h2>
      <p>144,000 swept intervals reduced to observations and revisit gaps.</p>
    </div>
  </a>
</div>

```{toctree}
:hidden:
:maxdepth: 2

guide
concepts
examples/index
api
development
```
