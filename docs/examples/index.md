# Examples

Both case studies run from pinned inputs every time the docs are built, so the
figures and timings below always match the source. If you are still learning the
API, start with [Getting started](../guide.md).

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

The timings describe whatever machine built these pages, so treat them as a
sense of proportion rather than a benchmark. [Performance and
memory](../performance.md) is the place for sizing a real run.

```{toctree}
:hidden:
:maxdepth: 1

communication-constellation
earth-observation-constellation
```
