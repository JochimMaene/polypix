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

Publishable wheels are built by `.github/workflows/release.yml` with
`cibuildwheel`. That workflow prepares a pinned native dependency prefix,
builds wheels, repairs them with `auditwheel` on Linux or `delocate` on macOS,
and runs the test suite against the repaired wheels.

Build with Python's stable ABI:

```bash
python -m build --wheel --no-isolation \
  --config-setting=cmake.define.POLYPIX_STABLE_ABI=ON
```

Polypix supports Python 3.12 and newer, so release builds can use one `abi3`
wheel per platform when `POLYPIX_STABLE_ABI=ON`.

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
