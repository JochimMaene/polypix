"""Regenerate the HEALPix fixtures used by the ring-geometry tests.

Run this in an environment that provides ``healpy``; it is deliberately not a
Polypix development or runtime dependency, so no pixi environment installs it.

The geometry fixtures currently checked in were produced with astropy-healpix
2.0.1, which agrees with healpy on RING order. The neighbor signatures were
produced with healpy 1.19.0.

``FREQUENCIES`` below must stay equal to the tuple in ``_geometry_signature``
in ``tests/test_ring_geometry.py``; the signatures are compared against each
other, so the two drifting apart would make those assertions meaningless.
"""

from __future__ import annotations

import hashlib
import math

import healpy
import numpy as np

RESOLUTIONS = (0, 1, 3, 8, 16, 29)
FREQUENCIES = (0.6180339887498948, 1.4142135623730951, 2.718281828459045)


def signature(values: np.ndarray) -> tuple[float, float, float]:
    flattened = values.ravel()
    return tuple(
        math.fsum(
            float(value) * math.sin((index + 1) * frequency)
            for index, value in enumerate(flattened)
        )
        / flattened.size
        for frequency in FREQUENCIES
    )


for resolution in RESOLUTIONS:
    pixel_count = 12 * 4**resolution
    cells = np.asarray(
        [index * (pixel_count - 1) // 256 for index in range(257)],
        dtype=np.int64,
    )
    centers = np.stack(
        healpy.pix2vec(2**resolution, cells, nest=False),
        axis=-1,
    )
    corners = np.moveaxis(
        healpy.boundaries(2**resolution, cells, step=1, nest=False),
        1,
        2,
    )
    print(f"{resolution}: ({signature(centers)!r}, {signature(corners)!r}),")

for resolution in (8, 16):
    nside = 2**resolution
    pixel_count = 12 * nside * nside
    north_end = 2 * nside * (nside - 1) - 1
    equator_start = north_end + 1
    south_start = pixel_count - 2 * nside * (nside + 1)
    cells = np.asarray(
        [0, north_end, equator_start, pixel_count // 2, south_start, pixel_count - 1],
        dtype=np.int64,
    )
    centers = np.stack(healpy.pix2vec(nside, cells, nest=False), axis=-1)
    corners = np.moveaxis(
        healpy.boundaries(nside, cells, step=1, nest=False),
        1,
        2,
    )
    print(f"resolution {resolution} exact cells: {cells.tolist()!r}")
    print(f"resolution {resolution} exact centers: {centers.tolist()!r}")
    print(f"resolution {resolution} exact boundaries: {corners.tolist()!r}")


def neighbor_signature(resolution: int, cells: np.ndarray) -> str:
    neighbors = healpy.get_all_neighbours(2**resolution, cells, nest=False).T
    neighbors.sort(axis=1)
    return hashlib.sha256(neighbors.astype("<i8").tobytes()).hexdigest()


print("HEALPY_NEIGHBOR_SIGNATURES = {")
for resolution in range(7):
    cells = np.arange(12 * 4**resolution, dtype=np.int64)
    print(f"    {resolution}: {neighbor_signature(resolution, cells)!r},")
print("}")

resolution = 29
pixel_count = 12 * 4**resolution
cells = np.asarray(
    [index * (pixel_count - 1) // 256 for index in range(257)], dtype=np.int64
)
print(f"resolution 29 neighbors: {neighbor_signature(resolution, cells)!r}")
print(
    "resolution 1 exceptional cell 4:",
    sorted(
        int(cell) for cell in healpy.get_all_neighbours(2, 4, nest=False) if cell >= 0
    ),
)
