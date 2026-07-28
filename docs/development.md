# Development

## Repository Layout

```text
polypix/
  __init__.py        Python API and input validation
  _core.pyi          private compiled-extension typing stub
  py.typed            PEP 561 marker
rust/
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
  communication_constellation.py      one-hour communications availability
  earth_observation_constellation.py  ten-day observation and revisit analysis
tools/
  generate_ring_geometry_fixtures.py  external-oracle fixture generator
decisions/
  owned-healpix-kernel.md  internal architecture and licensing evidence
docs/
  *.md                Zensical documentation pages
```

## Build Model

Polypix is a mixed Python/Rust project built by Maturin. PyO3 exposes the Rust
kernel as `polypix._core`; the Python package provides the public NumPy-first
API. The focused HEALPix RING kernel is owned by Polypix and has no HEALPix
runtime dependency.

PyPI wheels contain the compiled kernel. NumPy is the only runtime dependency:
HEALPix C++, CFITSIO, CMake, and a C++ compiler are not part of the build or
runtime dependency chain.

## Common Commands

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

Run either standalone constellation example:

```bash
pixi run --environment docs docs-communications
pixi run --environment docs docs-earth-observation
```

The documentation build executes both examples through Markdown Exec and embeds
their current results and performance measurements in the corresponding pages.

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

Cross-library benchmarks and their optional dependencies intentionally live in
a separate comparison repository. That repository is not public yet. Until it
is linked here, this repository makes no public cross-library performance
claim; it keeps only focused CodSpeed regression benchmarks and product
correctness tests.

The broad independent boundary fixtures can be regenerated in a temporary
environment containing `healpy`:

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

Internal architecture and licensing rationale is retained in the repository's
`decisions/` directory rather than published as user-facing performance
documentation.

## Release Builds

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

## Release Procedure

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

## Documentation Publishing

The documentation source lives in `docs/` and is configured by
`zensical.toml`. Build it locally with:

```bash
pixi run --environment docs docs-build
```

The `.github/workflows/docs.yml` workflow builds the same site on pull requests
and publishes `site/` to GitHub Pages on pushes to `main`.

## License And Notices

Polypix is distributed under Apache-2.0. The boundary transform includes a
small BSD-3-Clause adaptation from Astrometry.net. Keep these files current
whenever native dependencies or adapted code change:

- `LICENSE`;
- `THIRD_PARTY_NOTICES.md`;
- `Cargo.lock`.

Release maintainers should check the locked Rust dependency graph before
publishing and preserve every required third-party attribution.

## Design Constraints

Keep the Python layer thin:

- normalize ergonomic array inputs and construct results in Python;
- perform expensive geometry, HEALPix, and parallel work in Rust;
- return NumPy arrays rather than Python lists for large results.

When adding public functions, update:

- `polypix/__init__.py`;
- `docs/api.md`;
- tests and benchmarks where applicable.
