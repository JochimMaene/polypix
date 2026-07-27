from __future__ import annotations

import math

import numpy as np

import polypix as px


# Independent fixtures generated with astropy-healpix 2.0.1 in RING order.
RESOLUTION_0_CENTERS = np.asarray(
    [
        [0.5270462766947299, 0.5270462766947298, 2.0 / 3.0],
        [-0.5270462766947298, 0.5270462766947299, 2.0 / 3.0],
        [-0.52704627669473, -0.5270462766947298, 2.0 / 3.0],
        [0.5270462766947298, -0.52704627669473, 2.0 / 3.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.5270462766947299, 0.5270462766947298, -2.0 / 3.0],
        [-0.5270462766947298, 0.5270462766947299, -2.0 / 3.0],
        [-0.52704627669473, -0.5270462766947298, -2.0 / 3.0],
        [0.5270462766947298, -0.52704627669473, -2.0 / 3.0],
    ]
)

RESOLUTION_1_BOUNDARY_CELLS = np.asarray(
    [0, 4, 12, 27, 44, 47],
    dtype=np.uint64,
)
RESOLUTION_1_BOUNDARIES = np.asarray(
    [
        [
            [0.0, 0.0, 1.0],
            [0.3996526269427268, 0.0, 0.9166666666666666],
            [0.5270462766947299, 0.5270462766947298, 2.0 / 3.0],
            [0.0, 0.3996526269427268, 0.9166666666666666],
        ],
        [
            [0.3996526269427268, 0.0, 0.9166666666666666],
            [0.7453559924999299, 0.0, 2.0 / 3.0],
            [0.871041976584251, 0.36079740009746464, 1.0 / 3.0],
            [0.5270462766947299, 0.5270462766947298, 2.0 / 3.0],
        ],
        [
            [0.7453559924999299, 0.0, 2.0 / 3.0],
            [0.8710419765842508, -0.36079740009746525, 1.0 / 3.0],
            [1.0, 0.0, 0.0],
            [0.871041976584251, 0.36079740009746464, 1.0 / 3.0],
        ],
        [
            [0.8710419765842508, -0.36079740009746525, 1.0 / 3.0],
            [0.7071067811865474, -0.7071067811865477, 0.0],
            [0.8710419765842508, -0.36079740009746525, -1.0 / 3.0],
            [1.0, 0.0, 0.0],
        ],
        [
            [0.5270462766947299, 0.5270462766947298, -2.0 / 3.0],
            [0.3996526269427264, 0.0, -0.9166666666666667],
            [0.0, 0.0, -1.0],
            [0.0, 0.3996526269427264, -0.9166666666666667],
        ],
        [
            [0.5270462766947298, -0.52704627669473, -2.0 / 3.0],
            [0.0, -0.3996526269427264, -0.9166666666666667],
            [0.0, 0.0, -1.0],
            [0.3996526269427264, 0.0, -0.9166666666666667],
        ],
    ]
)

RESOLUTION_29_CELLS = np.asarray(
    [
        0,
        3,
        576460751229681663,
        576460751229681664,
        576460752303423488,
        3458764513820540924,
        3458764513820540927,
    ],
    dtype=np.uint64,
)
RESOLUTION_29_CENTERS = np.asarray(
    [
        [1.0753986783132438e-9, 1.0753986783132438e-9, 1.0],
        [1.0753986783132436e-9, -1.075398678313244e-9, 1.0],
        [0.7453559913892629, -1.0903947199425992e-9, 0.6666666679084301],
        [0.7453559924999299, 1.090394756918265e-9, 2.0 / 3.0],
        [-0.7453559924999299, -1.0903947787326189e-9, 2.0 / 3.0],
        [1.0753986783132438e-9, 1.0753986783132438e-9, -1.0],
        [1.0753986783132436e-9, -1.075398678313244e-9, -1.0],
    ]
)

# Generated with healpy 1.19.0 (HEALPix C++) in RING order. For every
# resolution, this samples 257 cells evenly in integer index space. Three
# deterministic projections make the broad fixture compact and tolerant of
# harmless cross-platform libm differences. The targeted fixtures below retain
# near-machine precision.
HEALPY_AUDIT_SIGNATURES = {
    0: (
        (-0.0010407919060051458, 0.0016874391878383072, -0.0029535840950235107),
        (-0.0011849709393517932, -0.0007851345447184232, 0.0002748473435176043),
    ),
    1: (
        (0.0011224433212577063, 0.0007263991014049507, 0.000366822280390948),
        (0.0009520836026004392, 0.0008278925666396833, -0.0002057934514186156),
    ),
    3: (
        (0.0010397923456448707, -0.00024004507517473987, 0.0010485113698820174),
        (-0.0005165421348485419, -0.0004504773049630604, -0.0010571311986897235),
    ),
    8: (
        (-0.0015337419981973353, 0.003865961717953282, -0.0009529440256657414),
        (-0.00011643661736978893, -0.0013200233403872223, -0.00004058927572031214),
    ),
    16: (
        (-0.004673697477985021, -0.006910604004154949, -0.0038879081309960402),
        (-0.0000716377818796311, -0.0005889760656836777, 0.000427784661151509),
    ),
    29: (
        (0.0009966421714639718, -0.0016967514417623628, -0.00009565107089005326),
        (0.00010363500086787705, -0.00024355623103113557, 0.00006305953325996733),
    ),
}


def _geometry_signature(values: np.ndarray) -> np.ndarray:
    flattened = values.ravel()
    frequencies = (0.6180339887498948, 1.4142135623730951, 2.718281828459045)
    return np.asarray(
        [
            math.fsum(
                float(value) * math.sin((index + 1) * frequency)
                for index, value in enumerate(flattened)
            )
            / flattened.size
            for frequency in frequencies
        ]
    )


def test_ring_centers_match_independent_base_resolution_fixture() -> None:
    actual = px.centers(np.arange(12, dtype=np.uint64), resolution=0)

    np.testing.assert_allclose(actual, RESOLUTION_0_CENTERS, rtol=0.0, atol=2e-15)


def test_ring_boundaries_match_independent_polar_and_equatorial_fixtures() -> None:
    actual = px.boundaries(RESOLUTION_1_BOUNDARY_CELLS, resolution=1)

    np.testing.assert_allclose(actual, RESOLUTION_1_BOUNDARIES, rtol=0.0, atol=3e-15)


def test_ring_centers_keep_polar_precision_at_resolution_29() -> None:
    actual = px.centers(RESOLUTION_29_CELLS, resolution=29)

    np.testing.assert_allclose(actual, RESOLUTION_29_CENTERS, rtol=0.0, atol=4e-15)


def test_ring_geometry_matches_broad_healpix_cpp_audit() -> None:
    for resolution, (center_signature, boundary_signature) in (
        HEALPY_AUDIT_SIGNATURES.items()
    ):
        pixel_count = 12 * 4**resolution
        cells = np.asarray(
            [index * (pixel_count - 1) // 256 for index in range(257)],
            dtype=np.uint64,
        )

        np.testing.assert_allclose(
            _geometry_signature(px.centers(cells, resolution)),
            center_signature,
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            _geometry_signature(px.boundaries(cells, resolution)),
            boundary_signature,
            rtol=0.0,
            atol=1e-15,
        )
