"""Immediate HEALPix neighbor lookup."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import polypix as px

# SHA-256 of sorted, -1-padded neighbor rows generated independently with
# healpy 1.19.0 in RING order. These exhaust every face edge, polar transition,
# and exceptional vertex through resolution 6 without making healpy a runtime
# or test dependency.
HEALPY_NEIGHBOR_SIGNATURES = {
    0: "4709ada3a23953230ee9b58171d81c048e17484fceb6451d0d9486d69f0a7b9d",
    1: "cc3fa087b6c3c663d74ec35e29ad84dcc064846459d6a8518e93fd3244854ee3",
    2: "9aa2aca7a504f10f3aa433c6f4c36851f8f4da1157e98113bd9386533e5e6b3b",
    3: "f4ec38c3d6f0c665936b8cffc74d2020b434982ea7400253381fb415205a35ff",
    4: "263531391226103d9f7fed738c0c49bdebdd026224ee283bec44477508eccf5e",
    5: "696b3e3d73349ab1f59927d41cdcdf06fb09aa362c9442ea94003f38a3fe30b3",
    6: "c4e48373a2205a13fb8a0e5ea7396075636ded0a99049bc4d0b13079b0c310f0",
}


def _canonical_neighbors(result: px.Coverage) -> np.ndarray:
    rows = np.full((len(result), 8), -1, dtype="<i8")
    for index in range(len(result)):
        values = np.sort(result[index])
        rows[index, -len(values) :] = values
    return rows


def test_neighbors_match_exhaustive_independent_healpy_signatures() -> None:
    for resolution, expected in HEALPY_NEIGHBOR_SIGNATURES.items():
        cells = np.arange(px.cell_count(resolution), dtype=np.int64)
        result = px.cell_neighbors(cells, resolution)
        actual = hashlib.sha256(_canonical_neighbors(result).tobytes()).hexdigest()
        assert actual == expected


def test_neighbors_match_resolution_29_healpy_signature() -> None:
    resolution = 29
    total = px.cell_count(resolution)
    cells = np.asarray([index * (total - 1) // 256 for index in range(257)])
    result = px.cell_neighbors(cells, resolution)
    actual = hashlib.sha256(_canonical_neighbors(result).tobytes()).hexdigest()
    assert actual == "a42c08906013e3eac12db3b1a0811a8a57adc9559784425a3c713d245155014f"


def test_neighbors_preserve_batch_alignment_and_validate_cells() -> None:
    result = px.cell_neighbors([4, 4], resolution=0)
    assert result.resolution == 0
    assert len(result) == 2
    np.testing.assert_array_equal(result[0], result[1])
    np.testing.assert_array_equal(np.sort(result[0]), [0, 3, 5, 7, 8, 11])
    assert not result.cells.flags.writeable
    assert not result.offsets.flags.writeable

    exceptional = px.cell_neighbors(4, resolution=1)
    np.testing.assert_array_equal(np.sort(exceptional[0]), [0, 3, 5, 11, 12, 13, 20])

    empty = px.cell_neighbors([], resolution=3)
    np.testing.assert_array_equal(empty.cells, [])
    np.testing.assert_array_equal(empty.offsets, [0])

    with pytest.raises(TypeError, match="integers"):
        px.cell_neighbors([1.0], resolution=1)
    with pytest.raises(ValueError, match="non-negative"):
        px.cell_neighbors([-1], resolution=1)
    with pytest.raises(ValueError, match="valid RING indices"):
        px.cell_neighbors([48], resolution=1)
