# Install

Polypix supports Python 3.12 and newer on Linux x86_64, macOS Intel, and macOS
Apple Silicon:

```bash
python -m pip install polypix
```

Windows wheels are not enabled yet because `healpix_cxx` is not currently
available as a conda-forge `win-64` package.

## Development Environment

Install [Pixi](https://pixi.sh), then run:

```bash
pixi run test
```

This creates a conda-forge environment with:

- Python 3.12 or newer and NumPy,
- CMake, Ninja, and a C++17 compiler,
- nanobind,
- `healpix_cxx`,
- pytest.

The Pixi configuration lives in `pyproject.toml`.

## Binary Wheels

Release wheels are built in CI with `cibuildwheel`, not from the local Pixi
development environment. Linux wheels are repaired with `auditwheel`; macOS
wheels are repaired with `delocate`.

## Documentation

The documentation site is built with Zensical:

```bash
pixi run docs-build
```

To preview the site while editing:

```bash
pixi run docs-serve
```
