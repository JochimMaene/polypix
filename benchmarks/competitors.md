# Comparing Polypix with other libraries

`competitors.py` runs Polypix next to healpy, HPGeom, MOCPy, H3, and S2 on the
same shapes, and `plot_comparisons.py` turns the results into the charts on the
[benchmarks page](../docs/benchmarks.md).

## Running it

Use a separate environment, so the other libraries never end up next to your
normal Polypix install. Installing `.` builds Polypix from this checkout, which
needs a Rust toolchain:

```sh
python -m venv .venv-bench
.venv-bench/bin/pip install -r benchmarks/competitors-requirements.txt .
.venv-bench/bin/python benchmarks/competitors.py --quick   # about ten seconds
.venv-bench/bin/python benchmarks/competitors.py           # about twenty minutes
pixi run -e docs python benchmarks/plot_comparisons.py
```

The full run writes `results/cross-grid.json` and `results/healpix-large.json`;
the quick one writes to `results/quick/` so it never replaces them.
Each file records the commit, whether the checkout had uncommitted changes, the
CPU, and every library version. Run it from a clean commit if you want to
publish the numbers.

## The workloads

We draw 128 random tracks between 63° south and 63° north. Each track gets:

- a circle with a radius between 3° and 12°, like a satellite's coverage area;
- four overlapping 1.4° × 0.9° footprints stepping east, like consecutive images
  along a ground track.

**Visibility** counts, for every cell, how many circles cover it.
**Observation** finds the cells covered by at least one footprint.

## What we time

Only the coverage itself: the library's coverage call, plus whatever it takes to
end up with per-cell counts, such as adding up the cells of each shape. Each
library gets 3D unit vectors, so converting to latitude and longitude counts
towards the library that needs them. H3 has no circles, so we hand it a
128-point outline of each one and include building it.

Each case runs in five fresh processes. A process makes one call to warm up,
then five timed calls, and keeps the median. The charts show the median over
processes, with whiskers for the fastest and slowest; the CSV also has the
first call. Polypix runs on one thread unless the label says "threaded", and
MOCPy is held to one thread too. healpy, HPGeom, H3, and S2 are single-threaded
anyway.

## Comparing fairly

Libraries differ in when a cell counts as covered, so we only compare libraries
that use the same rule:

| Rule | Libraries |
|---|---|
| Cell's center is inside the shape | Polypix, healpy, HPGeom, H3 |
| Any part of the cell touches the shape | Polypix (`mode="overlap"`), MOCPy, S2, H3 (`contain="overlap"`) |

The grids don't line up either. HEALPix has 12 × 4ⁿ cells, S2 6 × 4ⁿ, and H3
grows sevenfold per level. So we run each library at several of its own levels
and plot time against the number of cells on the globe.

healpy and HPGeom must select exactly the cells Polypix selects; the run fails
otherwise. The second suite uses that to compare speed on much bigger inputs:
10,000 circles and 4,000 footprints, HEALPix libraries only.

## Checking the answers

After timing, we look each result up at 100,000 random points and compare it
with the exact answer from spherical geometry. This is a sanity check, not a
ranking: the error mostly comes from using cells at all, and it is about the
same on every grid with the same number of cells. It does catch a broken
adapter straight away, and it is how we noticed that MOCPy approximates circles.

The JSON keeps two numbers per case. `mean_abs_error` is the average number of
circles miscounted per point for visibility, or the fraction of points wrongly
marked as seen or unseen for observation. `mean_error` keeps the sign, so you
can see that the touch rule over-counts.

## Things to keep in mind

- These are numbers from one laptop, an Intel i7-1165G7 under WSL2. Expect
  different ratios on other hardware.
- H3 draws polygon edges as straight lines in latitude and longitude, not great
  circles. The edges here are short enough that the difference is far below
  one cell.
- H3's overlap mode is marked experimental in h3-py.
- The answer check uses random points, so its error figures are estimates. At
  the finest levels they rest on a couple of hundred points out of 100,000.
