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
| Linux | x86-64 | CPython 3.12 and newer |
| Linux | ARM64 | CPython 3.12 and newer |
| macOS 11+ | x86-64 | CPython 3.12 and newer |
| macOS 11+ | ARM64 | CPython 3.12 and newer |
| Windows | x86-64 | CPython 3.12 and newer |

The wheel contains the native coverage kernel. NumPy is the only runtime
dependency: installing a wheel does not require Rust, a C++ compiler, a system
HEALPix library, or CFITSIO.

## Source Builds

Most users should install a wheel from PyPI. A source build requires a stable
Rust toolchain and a supported CPython installation. The Maturin build backend
fetches and compiles the Rust crate dependencies; no system HEALPix library is
required.

The repository's supported source-build environment is Pixi:

```bash
pixi run test
```

This creates an environment with Python, NumPy, Rust, Maturin, and pytest, then
installs Polypix in editable mode.

## Local Wheels

To build a local wheel from the active Pixi environment:

```bash
pixi run wheel
```

That wheel is intended for local smoke testing. Release wheels are built by the
GitHub Actions release workflow with Maturin for every supported platform.

For contributor workflows such as documentation authoring, packaging, and
release steps, see [Development](development.md).
