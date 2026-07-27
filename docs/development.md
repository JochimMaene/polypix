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
tools/
  generate_ring_geometry_fixtures.py  external-oracle fixture generator
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

CodSpeed reports performance regressions through
`.github/workflows/codspeed.yml` on pull requests and pushes to `main`.

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
