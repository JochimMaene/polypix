"""Regenerate the broad HEALPix C++ audit signatures used by the test suite.

Run this in an environment that provides ``healpy``; it is deliberately not a
Polypix development or runtime dependency.
"""

from __future__ import annotations

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
