# Install

Polypix has not published release wheels yet. The supported path today is a
Linux or macOS source checkout built through Pixi and conda-forge dependencies.

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

Release wheels will be built in CI with `cibuildwheel`, not from the local Pixi
development environment. Linux wheels are repaired with `auditwheel`; macOS
wheels are repaired with `delocate`.

Windows wheels are not enabled yet because `healpix_cxx` is not currently
available as a conda-forge `win-64` package.

## Documentation

The documentation site is built with Zensical:

```bash
pixi run docs-build
```

To preview the site while editing:

```bash
pixi run docs-serve
```
