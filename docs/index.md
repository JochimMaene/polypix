# Polypix

Polypix returns the HEALPix RING cells whose centers fall inside convex
spherical footprints. It is a small, NumPy-first kernel for coverage
simulations and spatial indexing.

Use it when the footprint geometry already exists and throughput matters.
Satellite and sensor coverage is the leading application, though the vectors may
belong to any spherical frame. Polypix does not model orbits, attitude, sensors,
time, ellipsoids, or coordinate reference systems.

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
strips, candidate cells, and threading.

## Examples

Two constellation studies execute during every documentation build, so their
maps and timings match the code that produced them.

- [Communications availability](examples/communication-constellation.md) —
  mean satellites in view for a 500-satellite shell, 30,500 footprints.
- [Earth-observation revisit](examples/earth-observation-constellation.md) —
  observation counts and revisit gaps over ten days, 144,000 swept intervals.

## Documentation

| Page | Contents |
| --- | --- |
| [Install](install.md) | Wheels, source builds |
| [Concepts](concepts.md) | Cell IDs, geometry rules, batches, threading |
| [API](api.md) | Complete public interface |
| [Development](development.md) | Contributing and releases |
| [Project goal](project-goal.md) | Scope and feature admission |
