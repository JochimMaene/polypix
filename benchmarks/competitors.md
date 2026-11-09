# Reproducible coverage comparisons

Run from the repository root on Linux or macOS, in a separate environment:

```sh
python -m venv .venv-competitors
.venv-competitors/bin/python -m pip install -r benchmarks/competitors-requirements.txt
.venv-competitors/bin/python -m pip install --no-deps .
PYTHONPATH=. .venv-competitors/bin/python benchmarks/competitors.py --output competitor-results.json
.venv-competitors/bin/python -m pip freeze > competitor-environment.txt
```

Keep the JSON, environment freeze, and source commit together. Python package
availability varies by platform; the recorded versions were validated on Linux
with Python 3.14. No competitor is a Polypix runtime dependency.

Two deterministic synthetic workloads:

* **Visibility:** 128 cap centers spanning longitudes and latitudes up to 63
  degrees, with radii from 3 to 12 degrees. Produce per-cell overlapping service
  counts for one timestamp. Repetitions measure the same timestamp independently;
  they do not simulate an orbit or measure time-series throughput.
* **Observation:** four consecutive overlapping convex footprints per track,
  across 128 tracks. Produce their union and estimate the observed fraction of
  the whole Earth (the target region) using equal-area random samples.

Use `--regions 8 --order 4 --samples 2000 --repeats 1` for a smoke run. Use
`--backends polypix-serial healpy hpgeom` for direct HEALPix comparisons.
Order means HEALPix order, not nside. H3 and S2 levels minimize the logarithmic
ratio of global cell counts to the requested HEALPix count. JSON records actual
levels, counts, and mean cell areas; equal mean area does not mean equal accuracy.

## What is measured

Each repetition runs in a fresh process. Timing includes shape construction,
covering, count accumulation or union, and lookup of the common evaluation points. Imports, deterministic input creation,
evaluation-point indexing, and analytic correctness checks are excluded. There
are separate first-call and warmed measurements. Each process records its first
call followed by `--iterations` warmed calls (default five). `seconds` is the
within-process warm median; `warm_seconds` retains every sample. Report the
median and range across processes (`--repeats`, default five).

Peak RSS is the entire worker process high-water mark, including imports and
input preparation, captured before validation. It includes native allocations;
it is not incremental algorithm memory. Threads in BLAS/OpenMP are limited to
one. Polypix runs both explicitly serial and automatic; other libraries retain
their own native threading policies. These are not identical CPU budgets.

## Comparison semantics

* **Polypix / healpy / HPGeom:** center-selected RING cells at identical order.
  Full count maps or unions must equal Polypix exactly; a mismatch fails the run.
  Independent analytic containment at sampled points measures rasterization error.
* **H3:** center-selected polygons on its own grid. Caps are approximated by
  128-vertex inscribed spherical polygons, including conversion in the timing.
  This approximation contributes to the reported accuracy error.
* **S2:** native C++ `s2geometry` binding, fixed-level outer coverings. Caps are
  native caps; polygons have great-circle edges. Boundary cells can over-cover.
* **MOCPy:** native hierarchical outer coverage. Observation uses the native
  batch polygon constructor and a native union.
  Visibility retains one MOC per cap and accumulates counts at evaluation points.
  This evaluation is included in visibility timing; no dense map expansion is
  imposed on MOCPy.

The last three are application comparisons, not equivalent-output races. JSON
reports mean absolute count error, sample disagreement, and estimated/reference
covered fractions on the same independent points. Sampling is an estimate, not
an exact area computation or maximum boundary-error bound. Increase `--samples`
for small footprints and report sample count with results. Repeat at several
orders and seeds before drawing conclusions.

A missing package or adapter failure is recorded with its traceback and makes
the command exit nonzero. Successful records remain available. There are no
performance thresholds or claims that one backend universally wins.

API references: [healpy](https://healpy.readthedocs.io/en/stable/generated/healpy.query_polygon.html),
[HPGeom](https://hpgeom.readthedocs.io/), [MOCPy](https://cds-astro.github.io/mocpy/),
[H3](https://h3geo.org/docs/api/regions/), [S2](https://s2geometry.io/devguide/examples/coverings.html).

## Published matrix and charts

```sh
PYTHONPATH=. python benchmarks/run_comparisons.py
# Use the documentation environment for Matplotlib:
pixi run -e docs python benchmarks/plot_comparisons.py
```

The matrix runs all backends at orders 5, 6, 7 with 128 circles / 512 footprints.
It also runs equivalent HEALPix workloads with 10,000 circles or 4,000 footprints
at all three orders. Outputs go to `benchmarks/results`; SVG charts and a CSV
summary go to `docs/assets/benchmarks`. No timings run during a docs build.
See the [documentation page](../docs/benchmarks.md) for interpretation and limits.

The additional small-workload seed checks can be reproduced with:

```sh
mkdir -p benchmarks/results/seeds
for seed in 20261110 20261111; do
  for order in 5 6 7; do
    PYTHONPATH=. python benchmarks/competitors.py --seed "$seed" --order "$order" \
      --repeats 3 --iterations 3 --output "benchmarks/results/seeds/$seed-$order.json"
  done
done
```

These are separate validation runs, not pooled into the headline charts.
