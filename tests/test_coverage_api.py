"""Input shapes, grid helpers, and segment alignment of the covering calls."""

from __future__ import annotations

import math
import multiprocessing
import os
import platform
import subprocess
import sys

import numpy as np
import pytest

import polypix as px


def _xyz(longitude_deg: float, latitude_deg: float) -> tuple[float, float, float]:
    longitude = math.radians(longitude_deg)
    latitude = math.radians(latitude_deg)
    cos_latitude = math.cos(latitude)
    return (
        cos_latitude * math.cos(longitude),
        cos_latitude * math.sin(longitude),
        math.sin(latitude),
    )


def test_polygon_api_accepts_packed_and_ragged_vertices() -> None:
    first = np.asarray(
        [_xyz(-8, -5), _xyz(6, -5), _xyz(6, 7), _xyz(-8, 7)],
        dtype=np.float64,
    )
    second = np.asarray(
        [_xyz(32, 11), _xyz(43, 11), _xyz(43, 20), _xyz(32, 20)],
        dtype=np.float64,
    )

    # A uniform sequence, a dense array, and a ragged sequence containing the
    # same two polygons must all agree.
    sequence = px.cover_polygon([first, second], resolution=4)
    dense = px.cover_polygon(np.stack((first, second)), resolution=4)
    triangle = second[:3]
    ragged = px.cover_polygon([first, triangle], resolution=4)

    np.testing.assert_array_equal(dense.cells, sequence.cells)
    np.testing.assert_array_equal(dense.offsets, sequence.offsets)
    assert len(ragged) == 2
    np.testing.assert_array_equal(ragged[0], sequence[0])

    with pytest.raises(ValueError, match="geometry"):
        px.cover_polygon(np.ones((2, 2)), resolution=1)


def test_malformed_ragged_polygon_uses_the_public_validation_error() -> None:
    quad = np.asarray(
        [_xyz(-8, -5), _xyz(6, -5), _xyz(6, 7), _xyz(-8, 7)],
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match=r"geometry\[1\].*shape"):
        px.cover_polygon([quad, 1], resolution=4)


def test_grid_names_and_cell_count() -> None:
    assert px.cell_count(0) == 12
    assert px.cell_count(6) == 12 * 4**6
    with pytest.raises(ValueError):
        px.cell_count(30)

    cells = np.asarray([0, 7, 31], dtype=np.int64)
    assert px.cell_centers(cells, resolution=1).shape == (3, 3)
    assert px.cell_corners(cells, resolution=1).shape == (3, 4, 3)


@pytest.mark.parametrize("dtype", [np.float64, "<U3"])
def test_typed_empty_cell_arrays_require_an_integer_dtype(dtype: object) -> None:
    with pytest.raises(TypeError, match="integers"):
        px.cell_centers(np.empty(0, dtype=dtype), resolution=4)

    with pytest.raises(TypeError, match="integers"):
        px.Coverage.from_arrays(np.empty(0, dtype=dtype), [0], resolution=4)


def test_coverage_helpers_preserve_segment_alignment() -> None:
    coverage = px.Coverage.from_arrays(
        cells=[1, 4, 4, 2, 7],
        offsets=[0, 2, 2, 3, 5],
        resolution=1,
    )

    assert coverage.cells.dtype == np.int64
    assert coverage.offsets.dtype == np.int64
    np.testing.assert_array_equal(np.diff(coverage.offsets), [2, 0, 1, 2])
    np.testing.assert_array_equal(
        np.repeat(np.arange(len(coverage)), np.diff(coverage.offsets)), [0, 0, 2, 3, 3]
    )

    empty = px.Coverage.from_arrays([], [0], resolution=1)
    np.testing.assert_array_equal(np.diff(empty.offsets), [])
    np.testing.assert_array_equal(
        np.repeat(np.arange(len(empty)), np.diff(empty.offsets)), []
    )


def test_empty_cap_batches_validate_scalar_radii() -> None:
    empty = np.empty((0, 3), dtype=np.float64)
    for invalid in (-1.0, math.nextafter(math.pi, math.inf), np.nan, np.inf):
        with pytest.raises(ValueError, match="radii_rad"):
            px.cover_cap(empty, invalid, resolution=1)
        with pytest.raises(ValueError, match="radii_rad"):
            px.cover_cap(empty, invalid, resolution=1, reduce=px.Count())


def test_reducer_tokens_use_identity_equality() -> None:
    for first, second in (
        (px.Count(), px.Count()),
        (px.Sum(np.asarray([1.0, 2.0])), px.Sum(np.asarray([1.0, 2.0]))),
    ):
        assert first == first
        assert first != second


def test_sweep_accepts_an_empty_interval_axis() -> None:
    empty = np.empty((0, 3), dtype=np.float64)
    one = np.asarray([_xyz(0, 0)], dtype=np.float64)

    for edge in (empty, one):
        coverage = px.cover_sweep(edge, edge, resolution=2)
        assert len(coverage) == 0
        np.testing.assert_array_equal(coverage.cells, [])
        np.testing.assert_array_equal(coverage.offsets, [0])

    for invalid in ([0.0, 0.0, 0.0], [np.nan, 0.0, 1.0]):
        edge = np.asarray([invalid], dtype=np.float64)
        with pytest.raises(ValueError, match="left_edge_xyz"):
            px.cover_sweep(edge, edge, resolution=2)


@pytest.mark.parametrize(
    "call",
    [
        lambda: px.cover_cap([1.0, 0.0, 0.0], 0.1, 3),
        lambda: px.cover_polygon([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], 3),
        lambda: px.cover_sweep(
            [[1.0, 0, 0], [0, 1.0, 0]], [[1.0, 0, 0.1], [0, 1.0, 0.1]], 3
        ),
        lambda: px.cell_at([1.0, 0.0, 0.0], 3),
        lambda: px.cell_centers(5, 3),
        lambda: px.cell_corners(5, 3),
        lambda: px.cell_neighbors(5, 3),
    ],
)
def test_resolution_is_keyword_only(call) -> None:  # type: ignore[no-untyped-def]
    # Two adjacent numbers would otherwise swap silently.
    with pytest.raises(TypeError, match="positional"):
        call()


def _cover_in_forked_child(threads: int | None) -> int:
    vectors = np.random.default_rng(0).normal(size=(4000, 3))
    px.cell_centers(np.arange(1 << 17), resolution=9)
    return len(px.cover_cap(vectors, 0.05, resolution=9, threads=threads))


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork()")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_forked_child_does_not_deadlock_on_the_parent_pool() -> None:
    _cover_in_forked_child(None)
    _cover_in_forked_child(2)
    with multiprocessing.get_context("fork").Pool(1) as pool:
        assert pool.map_async(_cover_in_forked_child, [None, 2]).get(timeout=30)


_ALLOCATION_SWEEP = """
import resource, sys
import numpy as np
import polypix as px

def address_space():
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith("VmSize:"):
                return int(line.split()[1]) * 1024

revisit_input = px.Coverage.from_arrays(
    cells=np.arange(0, 400_000 * 7919, 7919), offsets=[0, 400_000], resolution=29
)
descending = np.arange(px.cell_count(10) - 1, px.cell_count(10) - 400_001, -1)
directions = np.random.default_rng(0).normal(size=(400_000, 3))
quads = directions[:, None, :] + np.array(
    [[0.0, 0.0, 0.0], [1e-3, 0.0, 0.0], [1e-3, 1e-3, 0.0], [0.0, 1e-3, 0.0]]
)
unsorted = np.arange(400_000)[::-1].copy()
calls = [
    lambda: px.cover_cap([0.0, 0.0, 1.0], 1e-3, resolution=12, threads=1),
    lambda: px.revisit(revisit_input),
    lambda: px.cover_cap(
        [1.0, 0.0, 0.0], 0.2, resolution=10, candidate_cells=descending, threads=1
    ),
    lambda: px.cover_cap(directions, 1e-3, resolution=0, threads=1),
    lambda: px.cover_cap(
        directions, 1e-3, resolution=4, mode="overlap", candidate_cells=[0], threads=1
    ),
    lambda: px.cover_polygon(quads, resolution=4, candidate_cells=[0], threads=1),
    lambda: px.cover_sweep(directions, directions + 1e-3, resolution=0, threads=1),
    lambda: px.Coverage.from_arrays(unsorted, [0, unsorted.size], resolution=10),
    lambda: px.cover_polygon(
        [px.Polygon(quads[0])] * 100_000, resolution=4, candidate_cells=[], threads=1
    ),
]
_, hard = resource.getrlimit(resource.RLIMIT_AS)
for megabytes in (*range(0, 41), 48, 64, 96, 128):
    for call in calls:
        resource.setrlimit(resource.RLIMIT_AS, (address_space() + (megabytes << 20), hard))
        try:
            call()
        except MemoryError:
            pass
        finally:
            resource.setrlimit(resource.RLIMIT_AS, (hard, hard))
print("survived")
"""


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or platform.libc_ver()[0] != "glibc",
    reason="needs RLIMIT_AS, /proc, and glibc's mmap threshold",
)
def test_allocation_failures_raise_instead_of_aborting() -> None:
    # Under an address-space limit an infallible allocation aborts the whole
    # process; every input-sized buffer must raise MemoryError instead. A fixed
    # mmap threshold keeps glibc from serving large buffers out of memory it
    # freed earlier, so each headroom step lands on the allocation it targets.
    result = subprocess.run(
        [sys.executable, "-c", _ALLOCATION_SWEEP],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "MALLOC_MMAP_THRESHOLD_": "131072"},
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == "survived"


def test_cap_ending_exactly_at_longitude_zero_includes_that_cell() -> None:
    # The cap's longitude span ends exactly at 2 pi, the center of cell 4.
    axis = [math.cos(0.1), -math.sin(0.1), 0.0]
    radius = 0.1 - 1e-14
    assert px.cover_cap(axis, radius, resolution=0).cells.tolist() == [4]
    assert px.cover_cap(
        axis, radius, resolution=0, candidate_cells=[4]
    ).cells.tolist() == [4]
    assert px.cover_cap(axis, radius, resolution=0, reduce=px.Count())[4] == 1
    assert 4 in px.cover_cap(axis, radius, resolution=0, mode="overlap").cells


def test_native_calls_reject_fortran_order() -> None:
    # The public functions convert; the native layer must not misread rows.
    vectors = np.asfortranarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="C-contiguous"):
        px._core._cell_at(vectors, 1)
    assert px.cell_at(vectors, resolution=1).tolist() == [20, 22]


_THREAD_SHORTAGE = """
import resource
import numpy as np
import polypix as px

def native_threads():
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith("Threads:"):
                return int(line.split()[1])

cells = np.arange(1 << 18)
# Other libraries, such as NumPy's BLAS, may already run threads of their own.
baseline = native_threads()
soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
resource.setrlimit(resource.RLIMIT_NPROC, (1, hard))
during = px.cell_centers(cells, resolution=10)
explicit = px.cell_neighbors(cells[:1000], resolution=10)
caps = px.cover_cap(
    np.random.default_rng(0).normal(size=(4000, 3)), 0.05, resolution=9, threads=4
)
assert native_threads() == baseline, "the shortage should force a serial call"
resource.setrlimit(resource.RLIMIT_NPROC, (soft, hard))
after = px.cell_centers(cells, resolution=10)
assert native_threads() > baseline, "parallel calls should recover after the shortage"
np.testing.assert_array_equal(during, after)
"""


@pytest.mark.skipif(
    not sys.platform.startswith("linux")
    or (os.cpu_count() or 1) < 2
    or os.geteuid() == 0,
    reason="needs RLIMIT_NPROC, which root ignores, /proc, and several CPUs",
)
def test_thread_shortage_falls_back_to_serial_and_recovers() -> None:
    # Rayon's global pool is built once; if that fails, every later parallel
    # call panics. Polypix's own pool must fall back and retry instead.
    result = subprocess.run(
        [sys.executable, "-c", _THREAD_SHORTAGE],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-1000:]


_FRESH_FORK = """
import os, sys, warnings
import numpy as np
import polypix as px

warnings.simplefilter("ignore", DeprecationWarning)
pid = os.fork()
if pid == 0:
    before = len(os.listdir("/proc/self/task"))
    px.cell_centers(np.arange(1 << 20), resolution=10)
    os._exit(0 if len(os.listdir("/proc/self/task")) > before else 1)
sys.exit(os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]))
"""


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or (os.cpu_count() or 1) < 2,
    reason="needs fork, /proc, and several CPUs",
)
def test_child_forked_before_any_pool_still_runs_in_parallel() -> None:
    # Only pools that existed at fork() are unusable in the child.
    result = subprocess.run(
        [sys.executable, "-c", _FRESH_FORK], capture_output=True, timeout=120
    )
    assert result.returncode == 0, result.stderr[-500:]
