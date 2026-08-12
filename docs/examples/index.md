# Examples

The documentation build runs both case studies from pinned inputs and records
their figures and one set of wall-clock timings. Start with [Getting
started](../guide.md) if you are learning the API.

<div class="example-gallery">
  <a class="example-card" href="communication-constellation.html">
    <img src="../generated/communications-availability.png" alt="Global Starlink visibility map">
    <div>
      <h2>Snapshot visibility</h2>
      <p>Count 10,771 simultaneous spherical service caps over one hour without materializing cap membership.</p>
    </div>
  </a>
  <a class="example-card" href="earth-observation-constellation.html">
    <img src="../generated/earth-observation-count.png" alt="Global Earth-observation count map">
    <div>
      <h2>Earth-observation revisit</h2>
      <p>Rasterize paired-edge sweeps, then reduce aligned occupancy into source runs and merged gaps.</p>
    </div>
  </a>
</div>

Timings on these pages describe the documentation builder that produced the
figures. For controlled performance measurements, see [Performance and
memory](../performance.md).

```{toctree}
:hidden:
:maxdepth: 1

communication-constellation
earth-observation-constellation
```
