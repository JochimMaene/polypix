# Install

```bash
python -m pip install polypix
```

NumPy is the only runtime dependency.

## Wheels

| Platform | Architectures | Python |
| --- | --- | --- |
| Linux | x86-64, ARM64 | CPython 3.12+ |
| macOS 11+ | x86-64, ARM64 | CPython 3.12+ |
| Windows | x86-64 | CPython 3.12+ |

Wheels bundle the native coverage kernel. Installing one requires no Rust
toolchain, C++ compiler, system HEALPix library, or CFITSIO.

## Building from source

A source build requires a stable Rust toolchain and CPython 3.12 or newer. The
Maturin backend fetches and compiles the Rust dependencies:

```bash
python -m pip install -e .
```

The supported development environment is Pixi, which provisions Python, NumPy,
Rust, Maturin, and pytest, then installs Polypix in editable mode:

```bash
pixi run test
```

Checked-in Pixi platforms are Linux x86-64 and macOS. Windows and Linux ARM64
contributors use the `pip install -e .` path, which CI also exercises.

For a local wheel:

```bash
pixi run wheel
```

Release wheels for every supported platform come from the GitHub Actions release
workflow. See [Development](development.md) for contributor workflows.
