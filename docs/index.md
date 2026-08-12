# Polypix

Polypix maps directions to HEALPix RING cells and rasterizes exact spherical
caps, convex footprints, and paired-edge sweeps by center inclusion. Fused cap
counts and segmented occupancy summaries avoid enormous intermediate arrays in
measured workloads.

Use it when the spherical region geometry already exists and throughput matters.
Satellite and sensor coverage is the leading application, though the vectors
may belong to any spherical frame. Polypix does not model orbits, attitude,
sensors, clocks, ellipsoids, or coordinate reference systems.

## Install

```bash
python -m pip install polypix
```

Wheels cover CPython 3.12+ on Linux, macOS, and Windows. See
[Install](install.md).

## Quick start

```python
import numpy as np
import polypix as px

lon, lat = np.radians([[-5.0, 12.0, 10.0, -6.0], [-5.0, -4.0, 9.0, 7.0]])
footprint = np.stack(
    [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
    axis=-1,
)

coverage = px.cover_footprint(footprint, resolution=8)
```

```pycon
>>> coverage.cells
array([332313, 332314, 332315, ...], dtype=uint64)
>>> coverage.offsets
array([   0, 3992], dtype=uint64)
>>> px.centers(coverage.cells, coverage.resolution).shape
(3992, 3)
```

`cells` holds standard HEALPix RING indices at one resolution. `offsets` splits
it into one segment per input footprint, so a batch of 10,000 footprints returns
two arrays rather than 10,000 objects. See [Concepts](concepts.md) for batches,
sweeps, candidate cells, and threading.

## Choose an operation

| Input and desired result | Operation |
| --- | --- |
| Directions → RING cells | `cell_at()` |
| Circular regions → explicit membership | `cover_cap()` |
| Convex boundaries → explicit membership | `cover_footprint()` |
| Paired sampled edges → interval membership | `cover_sweep()` |
| Many caps → per-cell counts | `count_caps_per_cell()` |
| Imported segmented arrays → validated result | `Coverage.from_arrays()` |
| Aligned occupancy bins → run/gap aggregates | `summarize_occupancy()` (provisional in 0.x) |

See [Guide and recipes](guide.md) for small dependency-free examples. The two
constellation studies below are executable performance case studies, not the
starting point for learning the API.

## Examples

Two constellation studies execute during every documentation build, so their
maps and timings match the code that produced them.

- [Starlink snapshot visibility](examples/communication-constellation.md) —
  657,031 exact service caps from a pinned catalog of 10,771 objects.
- [Earth-observation revisit](examples/earth-observation-constellation.md) —
  observation counts and revisit gaps over ten days, 144,000 swept intervals.

## Documentation

| Page | Contents |
| --- | --- |
| [Install](install.md) | Wheels, source builds |
| [Guide and recipes](guide.md) | Operation chooser and small worked examples |
| [Concepts](concepts.md) | Cell IDs, geometry rules, batches, threading |
| [API](api.md) | Complete public interface |
| [Performance and memory](performance.md) | Resolution, output sizing, candidates, threading |
| [Interoperability](interoperability.md) | HEALPix conventions and NumPy ecosystem seams |
| [Development](development.md) | Contributing and releases |
| [Project goal](project-goal.md) | Scope and feature admission |
