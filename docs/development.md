# Development

## Repository Layout

```text
polypix/
  __init__.py        Python API and input validation
  __init__.pyi       public typing stub
  py.typed            PEP 561 marker
rust/
  lib.rs              PyO3 module and native coverage kernel
tests/
  test_polypix.py     behavior tests
benchmarks/
  scorecard.py        correctness and performance scorecard
docs/
  *.md                Zensical documentation pages
```

## Build Model

Polypix is a mixed Python/Rust project built by Maturin. PyO3 exposes the Rust
kernel as `polypix._core`; the Python package provides the public NumPy-first
API. The kernel uses `cdshealpix` under its Apache-2.0 licensing option.

PyPI wheels contain the compiled kernel. NumPy is the only runtime dependency:
HEALPix C++, CFITSIO, CMake, and a C++ compiler are not part of the build or
runtime dependency chain.

## Common Commands

Run tests:

```bash
pixi run test
```

Build the extension in the development environment:

```bash
pixi run maturin develop
```

Build documentation:

```bash
pixi run docs-build
```

Preview documentation:

```bash
pixi run docs-serve
```

Build a release-mode wheel:

```bash
pixi run wheel
```

Run the CodSpeed benchmark suite locally:

```bash
pixi run --environment bench bench
```

Run the end-to-end scorecard separately when comparing correctness or
performance with optional competitor installations:

```bash
python -m benchmarks.scorecard --output scorecard.json
```

### Released C++ baseline

The v0.2.1 `healpix_cxx` implementation remains reproducible at commit
`20d2df6`. The compatibility driver below runs the same fixed fixtures and
normalizes v0.2.1's packed tokens to standard fixed-resolution NESTED indices:

```bash
git worktree add --detach /tmp/polypix-v021 20d2df6
cd /tmp/polypix-v021
pixi run -e test python /path/to/current/polypix/benchmarks/legacy_cpp_baseline.py \
  --output /tmp/polypix-v021.json
cd /path/to/current/polypix
pixi run -e test python -m benchmarks.legacy_cpp_baseline \
  --output /tmp/polypix-current.json
git worktree remove /tmp/polypix-v021
```

Compare only records with identical membership digests. The driver measures
complete public calls, but the implementations still differ in validation and
result contracts; report the exact commit, machine, thread mode, and workload
instead of presenting the result as a general Rust-versus-C++ claim.

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
pixi run docs-build
```

The `.github/workflows/docs.yml` workflow builds the same site on pull requests
and publishes `site/` to GitHub Pages on pushes to `main`.

## License And Notices

Polypix is distributed under Apache-2.0. `cdshealpix` is used under its
Apache-2.0 option. Keep these files current whenever native dependencies
change:

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
- `polypix/__init__.pyi`;
- `docs/api.md`;
- tests and the scorecard where applicable.
