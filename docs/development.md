# Development

## Repository layout

```text
polypix/
  __init__.py        Python API and input validation
  _core.pyi          private compiled-extension typing stub
  py.typed            PEP 561 marker
rust/
  access.rs           segmented occupancy run and gap reduction
  geometry.rs         convex spherical-polygon validation
  lib.rs              PyO3 bindings and native-buffer safety
  ring.rs             HEALPix RING coverage and threading
tests/
  test_polypix.py     behavior tests
  test_ring_geometry.py  independent HEALPix geometry fixtures
benchmarks/
  test_polypix_benchmarks.py  CodSpeed public-call regression benchmarks
examples/
  constellation.py                    shared orbit and plotting helpers
  communication_constellation.py      one-hour Starlink snapshot visibility
  data/                                permanent example input snapshots
  earth_observation_constellation.py  ten-day observation and revisit analysis
  documentation_assets.py             pre-build figure and measurement step
tools/
  generate_ring_geometry_fixtures.py  external-oracle fixture generator
decisions/
  owned-healpix-kernel.md          architecture and licensing evidence
  cap-and-occupancy-primitives.md  measured cap/occupancy scope evidence
  api-surface-beyond-constellations.md  pre-1.0 API discovery gate
docs/
  *.md                Zensical documentation pages
```

## Build model

Polypix is a mixed Python/Rust project built by Maturin. PyO3 exposes the Rust
kernel as `polypix._core`; the Python package provides the public NumPy-first
API. The focused HEALPix RING kernel is owned by Polypix and has no HEALPix
runtime dependency.

PyPI wheels contain the compiled kernel. NumPy is the only runtime dependency:
HEALPix C++, CFITSIO, CMake, and a C++ compiler are not part of the build or
runtime dependency chain.

## Common commands

Run tests:

```bash
pixi run test
```

Run Rust/Python linting, formatting, typing, and stub checks:

```bash
pixi run --environment test lint
```

Build the extension in the development environment:

```bash
pixi run maturin develop
```

Build documentation:

```bash
pixi run --environment docs docs-build
```

The communications example uses the docs-only Astroz binary dependency. The
published documentation build runs on supported Linux x86-64 with glibc 2.36
or newer; this does not change Polypix's runtime platform support.

Run either standalone constellation example, writing its maps to the working
directory:

```bash
pixi run --environment docs docs-communications
pixi run --environment docs docs-earth-observation
```

`docs-build` and `docs-serve` first run `docs-figures`, which executes both
examples and writes their maps and measurements to `docs/assets/generated/`.
The example pages then embed that output. Zensical collects the files under
`docs/` before rendering pages, so a figure written while a page renders is
never copied into the built site; the examples have to run first. Regenerate the
figures with `pixi run --environment docs docs-figures` after changing an
example, because `docs-serve` does not re-run them on reload.

Preview documentation:

```bash
pixi run --environment docs docs-serve
```

Build a release-mode wheel:

```bash
pixi run wheel
```

Run the CodSpeed benchmark suite locally:

```bash
pixi run --environment bench bench
```

Cross-library benchmarks and their optional dependencies live in a separate
comparison repository. That repository is not public yet. Until it
is linked here, this repository makes no public cross-library performance
claim; it keeps only focused CodSpeed regression benchmarks and product
correctness tests.

The broad independent center-and-corner fixtures can be regenerated in a
temporary environment containing `healpy`:

```bash
python tools/generate_ring_geometry_fixtures.py
```

`healpy` remains an external oracle, not a development or runtime dependency.

CodSpeed simulation reports deterministic instruction regressions for
single-threaded benchmarks through `.github/workflows/codspeed.yml` on pull
requests and pushes to `main`. Benchmarks marked `parallel` are excluded
because Valgrind simulation serializes worker threads. The same workflow runs
`tools/check_parallel_speedup.py` natively as a coarse wall-time guard for
actual multicore scaling.

Architecture and licensing rationale is retained in the repository's
`decisions/` directory and linked from the documentation. Decision records are
maintainer evidence, not user-facing performance instructions.

## Release builds

`.github/workflows/release.yml` uses Maturin to build a source distribution and
CPython 3.12 stable-ABI wheels for:

- Linux x86-64 and ARM64;
- macOS 11 or newer on x86-64 and ARM64;
- Windows x86-64.

Each release wheel contains the Rust kernel. Pull requests build and test a
smaller native-platform smoke matrix without importing from the source checkout;
releases and manual workflow runs build the complete platform matrix.

Publishing is release-driven. Publishing a GitHub release builds the artifacts
and uploads them to PyPI through trusted publishing.

## Release procedure

The package version comes from `Cargo.toml`; keep its Polypix entry in
`Cargo.lock` synchronized.

1. Prepare a release pull request that sets the version, dates and completes the
   matching `CHANGELOG.md` section, and updates license notices when needed. Run
   the test, lint, and documentation commands above, then merge only after all
   required workflows pass.
2. On `main`, manually run **Build and publish** as a dry run. Manual runs build
   and test the complete wheel matrix but do not publish to PyPI.
3. Create a draft GitHub release tagged `v<version>` at the exact release
   commit, using the changelog section as its notes. Publishing the release
   rebuilds the artifacts and uploads them to PyPI through trusted publishing.
4. When the workflow succeeds, verify the release from a clean environment:

   ```bash
   python -m pip install --no-cache-dir polypix==<version>
   python -c "import polypix as px; print(px.__version__)"
   ```

Check the PyPI metadata and wheel set. Published files are immutable; corrections
require a new patch release.

## Documentation publishing

The documentation source lives in `docs/` and is configured by
`docs/conf.py`. Sphinx renders the Markdown through MyST using the PyData
Sphinx Theme. Build it locally with:

```bash
pixi run --environment docs docs-build
```

The `.github/workflows/docs.yml` workflow builds the same site on pull requests
and publishes `site/` to GitHub Pages on pushes to `main`.

## License and notices

Polypix is distributed under Apache-2.0. The cell-corner transform includes a
small BSD-3-Clause adaptation from Astrometry.net. Keep these files current
whenever native dependencies or adapted code change:

- `LICENSE`;
- `THIRD_PARTY_NOTICES.md`;
- `Cargo.lock`.

Release maintainers should check the locked Rust dependency graph before
publishing and preserve every required third-party attribution.

## Design constraints

Keep the Python layer thin:

- normalize ergonomic array inputs and construct results in Python;
- perform expensive geometry, HEALPix, and parallel work in Rust;
- return NumPy arrays rather than Python lists for large results.

When adding public functions, update:

- `polypix/__init__.py`;
- `docs/api.md`;
- tests and benchmarks where applicable.

```{toctree}
:hidden:
:maxdepth: 1

project-goal
decisions
```
