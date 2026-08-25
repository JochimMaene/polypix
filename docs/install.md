# Install

```bash
pip install polypix
```

NumPy is the only runtime dependency. There are wheels for CPython 3.12 and
newer on Linux (x86-64 and ARM64), macOS 11+ (Intel and Apple Silicon), and
Windows x86-64, and each one carries the compiled coverage kernel.

## Building from source

A source build needs CPython 3.12 or newer and a stable Rust toolchain, which
[rustup](https://rustup.rs) installs. Nothing else: pip takes the source
distribution from PyPI, and the Maturin backend fetches and compiles the Rust
dependencies for you.

```bash
pip install polypix --no-binary polypix
```

[Development](development.md) covers building from a clone in the Pixi
environment we use for contributions, together with the test, lint, benchmark,
and documentation commands.
