from __future__ import annotations

import math
import unittest

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

RESOLUTION_1_CORNER_CELLS = np.asarray(
    [0, 4, 12, 27, 44, 47],
    dtype=np.uint64,
)
RESOLUTION_1_CORNERS = np.asarray(
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

# Exact HEALPix C++ fixtures at the polar/equatorial transition rings and poles.
HEALPY_EXACT_FIXTURES = {
    8: (
        np.asarray([0, 130559, 130560, 393216, 654848, 786431], dtype=np.uint64),
        np.asarray(
            [
                [0.002255271621290344, 0.0022552716212903435, 0.9999949137369791],
                [0.743019597693013, -0.002288502245918133, 0.6692657470703125],
                [0.7453524847126957, 0.0022867199580073774, 2.0 / 3.0],
                [-0.9999952938095762, -0.0030679567629652883, 0.0],
                [0.7453524847126957, 0.0022867199580073774, -2.0 / 3.0],
                [0.002255271621290343, -0.002255271621290344, -0.9999949137369791],
            ]
        ),
        np.asarray(
            [
                [
                    [0.0, 0.0, 1.0],
                    [0.0031894357136639626, 0.0, 0.9999949137369791],
                    [0.0045105260361471364, 0.004510526036147136, 0.9999796549479166],
                    [1.952966118912413e-19, 0.0031894357136639626, 0.9999949137369791],
                ],
                [
                    [0.7406830108925246, -1.814150156944709e-16, 0.6718546549479167],
                    [0.7430090248624353, -0.0045769827823064135, 0.6692657470703125],
                    [0.7453419613840095, -0.0045734183925356196, 2.0 / 3.0],
                    [0.7430231219810165, -1.8198817760530505e-16, 0.6692657470703125],
                ],
                [
                    [0.7430231219810165, 0.0, 0.6692657470703125],
                    [0.7453559924999299, 0.0, 2.0 / 3.0],
                    [0.747673548005822, 0.0022938409133473683, 0.6640625],
                    [0.7453419613840095, 0.004573418392535618, 2.0 / 3.0],
                ],
                [
                    [
                        -0.9999919029777714,
                        -0.0030679463599914063,
                        0.0026041666666666665,
                    ],
                    [-1.0, 1.2246467991473532e-16, 0.0],
                    [
                        -0.9999919029777714,
                        -0.0030679463599914063,
                        -0.0026041666666666665,
                    ],
                    [-0.9999811752826011, -0.006135884649154554, 0.0],
                ],
                [
                    [0.747673548005822, 0.0022938409133473683, -0.6640625],
                    [0.7453559924999299, 0.0, -2.0 / 3.0],
                    [0.7430231219810165, 0.0, -0.6692657470703125],
                    [0.7453419613840095, 0.004573418392535618, -2.0 / 3.0],
                ],
                [
                    [0.004510526036147135, -0.004510526036147137, -0.9999796549479166],
                    [
                        -5.858898356737238e-19,
                        -0.0031894357136639626,
                        -0.9999949137369791,
                    ],
                    [0.0, 0.0, -1.0],
                    [
                        0.0031894357136639626,
                        -7.811864475649652e-19,
                        -0.9999949137369791,
                    ],
                ],
            ]
        ),
    ),
    16: (
        np.asarray(
            [0, 8589803519, 8589803520, 25769803776, 42949541888, 51539607551],
            dtype=np.uint64,
        ),
        np.asarray(
            [
                [8.809665972571162e-06, 8.809665972571162e-06, 0.9999999999223897],
                [0.7453468938069785, -8.93254110820968e-06, 0.6666768391150981],
                [0.7453559924464053, 8.932513848460608e-06, 2.0 / 3.0],
                [-0.9999999999281892, -1.1984224904927685e-05, 0.0],
                [0.7453559924464053, 8.932513848460608e-06, -2.0 / 3.0],
                [8.80966597257116e-06, -8.809665972571164e-06, -0.9999999999223897],
            ]
        ),
        np.asarray(
            [
                [
                    [0.0, 0.0, 1.0],
                    [1.24587490983869e-05, 0.0, 0.9999999999223897],
                    [
                        1.7619331944116747e-05,
                        1.7619331944116744e-05,
                        0.9999999996895591,
                    ],
                    [7.628783602359745e-22, 1.24587490983869e-05, 0.9999999999223897],
                ],
                [
                    [0.7453377951100139, -1.8255510901300484e-16, 0.6666870114083092],
                    [0.7453468936464016, -1.786508221495386e-05, 0.6666768391150981],
                    [0.7453559922858314, -1.786502769579172e-05, 2.0 / 3.0],
                    [0.7453468938605041, -1.8255733756413768e-16, 0.6666768391150981],
                ],
                [
                    [0.7453468938605041, 0.0, 0.6666768391150981],
                    [0.7453559924999299, 0.0, 2.0 / 3.0],
                    [0.745365090905349, 8.932622886438887e-06, 0.666656494140625],
                    [0.7453559922858314, 1.7865027695638317e-05, 2.0 / 3.0],
                ],
                [
                    [
                        -0.999999999876449,
                        -1.198422490430762e-05,
                        1.0172526041666666e-05,
                    ],
                    [-1.0, 1.2246467991473532e-16, 0.0],
                    [
                        -0.999999999876449,
                        -1.198422490430762e-05,
                        -1.0172526041666666e-05,
                    ],
                    [-0.9999999997127567, -2.396844980825664e-05, 0.0],
                ],
                [
                    [0.745365090905349, 8.932622886438887e-06, -0.666656494140625],
                    [0.7453559924999299, 0.0, -2.0 / 3.0],
                    [0.7453468938605041, 0.0, -0.6666768391150981],
                    [0.7453559922858314, 1.7865027695638317e-05, -2.0 / 3.0],
                ],
                [
                    [
                        1.761933194411674e-05,
                        -1.761933194411675e-05,
                        -0.9999999996895591,
                    ],
                    [
                        -2.2886350807079236e-21,
                        -1.24587490983869e-05,
                        -0.9999999999223897,
                    ],
                    [0.0, 0.0, -1.0],
                    [1.24587490983869e-05, -3.051513440943898e-21, -0.9999999999223897],
                ],
            ]
        ),
    ),
}

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


class RingGeometryTests(unittest.TestCase):
    def test_direction_index_matches_independent_center_fixtures(self) -> None:
        fixtures = [(0, np.arange(12, dtype=np.uint64), RESOLUTION_0_CENTERS)]
        fixtures.extend(
            (resolution, cells, centers)
            for resolution, (cells, centers, _corners) in HEALPY_EXACT_FIXTURES.items()
        )
        fixtures.append((29, RESOLUTION_29_CELLS, RESOLUTION_29_CENTERS))

        for resolution, expected, vectors in fixtures:
            with self.subTest(resolution=resolution):
                np.testing.assert_array_equal(
                    px.cell_at(vectors * 7.5, resolution),
                    expected,
                )

    def test_ring_centers_match_independent_base_resolution_fixture(self) -> None:
        actual = px.centers(np.arange(12, dtype=np.uint64), resolution=0)

        np.testing.assert_allclose(
            actual,
            RESOLUTION_0_CENTERS,
            rtol=0.0,
            atol=2e-15,
        )

    def test_ring_corners_match_independent_fixtures(self) -> None:
        actual = px.corners(RESOLUTION_1_CORNER_CELLS, resolution=1)

        np.testing.assert_allclose(
            actual,
            RESOLUTION_1_CORNERS,
            rtol=0.0,
            atol=3e-15,
        )

    def test_ring_centers_keep_polar_precision_at_resolution_29(self) -> None:
        actual = px.centers(RESOLUTION_29_CELLS, resolution=29)

        np.testing.assert_allclose(
            actual,
            RESOLUTION_29_CENTERS,
            rtol=0.0,
            atol=4e-15,
        )

    def test_ring_geometry_matches_exact_transition_fixtures(self) -> None:
        for resolution, (cells, centers, corners) in HEALPY_EXACT_FIXTURES.items():
            with self.subTest(resolution=resolution):
                np.testing.assert_allclose(
                    px.centers(cells, resolution),
                    centers,
                    rtol=0.0,
                    atol=4e-15,
                )
                np.testing.assert_allclose(
                    px.corners(cells, resolution),
                    corners,
                    rtol=0.0,
                    atol=4e-15,
                )

    def test_ring_geometry_matches_broad_healpix_cpp_audit(self) -> None:
        for resolution, (
            center_signature,
            corner_signature,
        ) in HEALPY_AUDIT_SIGNATURES.items():
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
                _geometry_signature(px.corners(cells, resolution)),
                corner_signature,
                rtol=0.0,
                atol=1e-15,
            )

    def test_exhaustive_low_resolution_corner_topology(self) -> None:
        for resolution in range(7):
            cell_count = 12 * 4**resolution
            cells = np.arange(cell_count, dtype=np.uint64)
            centers = px.centers(cells, resolution)
            corners = px.corners(cells, resolution)

            np.testing.assert_allclose(
                np.linalg.norm(corners, axis=2),
                1.0,
                rtol=0.0,
                atol=3e-16,
            )
            normals = np.cross(corners, np.roll(corners, -1, axis=1))
            orientation = np.einsum("nvc,nc->n", normals, centers)
            sides = np.einsum("nvc,nc->nv", normals, centers)
            self.assertTrue(
                np.all(sides * np.sign(orientation)[:, np.newaxis] >= -1e-15)
            )

            _, sharing = np.unique(
                np.round(corners.reshape(-1, 3), decimals=13),
                axis=0,
                return_counts=True,
            )
            self.assertEqual(sharing.size, cell_count + 2)
            self.assertEqual(set(sharing.tolist()), {3, 4})
