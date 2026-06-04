# Install

Install Polypix from PyPI:

```bash
python -m pip install polypix
```

Verify the installation:

```bash
python -c "import polypix as px; print(px.__version__)"
```

## Supported Wheels

Published wheels support:

| Platform | Architecture | Python |
| --- | --- | --- |
| Linux | x86_64 | 3.12 and newer |
| macOS 11+ | x86_64 | 3.12 and newer |
| macOS 11+ | arm64 | 3.12 and newer |

Windows wheels are not published because `healpix_cxx` is not currently
available as a conda-forge `win-64` package.

## Source Builds

Most users should install a wheel from PyPI. A source build requires a C++17
toolchain and the native HEALPix C++ dependency to be available in the build
environment.

The repository's supported source-build environment is Pixi:

```bash
pixi run test
```

This creates a conda-forge environment with Python, NumPy, CMake, Ninja,
nanobind, `healpix_cxx`, and pytest, then installs Polypix in editable mode.

## Local Wheels

To build a local wheel from the active Pixi environment:

```bash
pixi run wheel
```

That wheel is intended for local smoke testing. Release wheels are built by the
GitHub Actions release workflow with `cibuildwheel` and repaired with
`auditwheel` on Linux or `delocate` on macOS.

For contributor workflows such as documentation authoring, packaging, and
release steps, see [Development](development.md).
