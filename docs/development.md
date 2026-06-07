# Development

## Repository Layout

```text
polypix/
  __init__.py        Python API and input validation
  __init__.pyi       public typing stub
  py.typed           PEP 561 marker
src/polypix/
  core.cpp           spherical geometry and HEALPix implementation
  core.h             native API declarations
  module.cpp         nanobind bindings
tests/
  test_polypix.py    behavior tests
docs/
  *.md               Zensical documentation pages
```

## Build Model

Polypix uses scikit-build-core and CMake for Python extension builds. The normal
dependency path is:

- nanobind from the build environment,
- `healpix_cxx` from the build environment,
- C++17 compiler from the build environment.

Native dependencies are expected to come from the active build environment. In
development, Pixi provides them from conda-forge.

## Common Commands

Run tests:

```bash
pixi run test
```

Configure a manual CMake build:

```bash
pixi run configure
```

Compile the extension in that manual build tree:

```bash
pixi run compile
```

Build documentation:

```bash
pixi run docs-build
```

Preview documentation:

```bash
pixi run docs-serve
```

Build a wheel in the active Pixi environment:

```bash
pixi run wheel
```

This is a local development wheel for smoke testing. It bundles runtime
libraries from the active Pixi environment and is not the artifact to upload to
PyPI.

Run the CodSpeed benchmark suite locally:

```bash
pixi run --environment bench bench
```

Local benchmark runs validate that the benchmark cases execute. Performance
regression reports are produced by `.github/workflows/codspeed.yml` on pull
requests and pushes to `main`.

## Release Builds

Publishable wheels are built by `.github/workflows/release.yml` with
`cibuildwheel`. The workflow prepares a pinned native dependency prefix, builds
wheels, repairs them with `auditwheel` on Linux or `delocate` on macOS, and
runs the test suite against the repaired wheels.

Publishing is release-driven: creating or publishing a GitHub release builds the
source distribution and release wheels, then uploads them to PyPI through trusted
publishing. Pull requests run a smaller wheel smoke matrix so packaging changes
are checked before a release.

The published wheel targets are Linux x86_64 and macOS 11 or newer on Intel and
Apple Silicon. The Linux wheels use a `manylinux_2_34` image because the native
runtime dependencies come from conda-forge. The macOS deployment target is 11.0
because the bundled conda-forge runtime libraries require macOS 11 or newer.

## Stable ABI Build

The CMake build has an optional stable-ABI mode:

```bash
python -m build --wheel --no-isolation \
  --config-setting=cmake.define.POLYPIX_STABLE_ABI=ON
```

This mode requires Python 3.12 or newer. It is not part of the current release
matrix, which builds version-specific CPython wheels.

## Documentation Publishing

The documentation source lives in `docs/` and is configured by
`zensical.toml`. Build it locally with:

```bash
pixi run docs-build
```

The `.github/workflows/docs.yml` workflow builds the same site on pull requests
and publishes `site/` to GitHub Pages on pushes to `main`.

## License And Notices

Polypix is distributed under GPL-3.0-or-later because binary wheels link
against HEALPix C++, which is GPL-2.0-or-later. Keep these files current when
native dependencies change:

- `LICENSE`
- `THIRD_PARTY_NOTICES.md`
- `ci/licenses/*`

Release maintainers must ensure that source availability for GPL-covered
bundled components matches the published binary wheel set.

## Design Constraints

Polypix should keep the Python layer thin:

- validate and normalize array inputs in Python,
- do expensive geometry and HEALPix work in C++,
- return NumPy arrays rather than Python lists for large results.

When adding public functions, update:

- `polypix/__init__.py`,
- `polypix/__init__.pyi`,
- `docs/api.md`,
- tests.
