#!/usr/bin/env bash

set -euo pipefail

: "${POLYPIX_DEPS_PREFIX:?POLYPIX_DEPS_PREFIX must point to the dependency install prefix}"

HEALPIX_CXX_VERSION="${HEALPIX_CXX_VERSION:-3.83}"
CFITSIO_VERSION="${CFITSIO_VERSION:-4.6.3}"
MICROMAMBA_VERSION="${MICROMAMBA_VERSION:-latest}"

if [ -f "$POLYPIX_DEPS_PREFIX/lib/pkgconfig/healpix_cxx.pc" ]; then
    echo "Using cached dependency prefix: $POLYPIX_DEPS_PREFIX"
    exit 0
fi

case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)
        micromamba_platform="linux-64"
        install_compiler=false
        ;;
    Darwin-x86_64)
        micromamba_platform="osx-64"
        install_compiler=true
        ;;
    Darwin-arm64)
        micromamba_platform="osx-arm64"
        install_compiler=true
        ;;
    *)
        echo "Unsupported platform for Polypix wheel dependencies: $(uname -s)-$(uname -m)" >&2
        exit 1
        ;;
esac

deps_root="$(dirname "$POLYPIX_DEPS_PREFIX")"
micromamba_root="$deps_root/micromamba"
micromamba_bin="$micromamba_root/bin/micromamba"

mkdir -p "$micromamba_root" "$POLYPIX_DEPS_PREFIX"

if [ ! -x "$micromamba_bin" ]; then
    echo "Installing micromamba for $micromamba_platform"
    curl -Ls "https://micro.mamba.pm/api/micromamba/$micromamba_platform/$MICROMAMBA_VERSION" |
        tar -xj -C "$micromamba_root" bin/micromamba
fi

packages=(
    "healpix_cxx=$HEALPIX_CXX_VERSION" \
    "cfitsio=$CFITSIO_VERSION" \
    "pkg-config"
)

if [ "$install_compiler" = true ]; then
    packages+=("cxx-compiler")
fi

"$micromamba_bin" create -y -p "$POLYPIX_DEPS_PREFIX" -c conda-forge \
    "${packages[@]}"

if [ "$install_compiler" = true ]; then
    cc_path="$(find "$POLYPIX_DEPS_PREFIX/bin" -maxdepth 1 -name '*-apple-darwin*-clang' \( -type f -o -type l \) | head -n 1)"
    cxx_path="$(find "$POLYPIX_DEPS_PREFIX/bin" -maxdepth 1 -name '*-apple-darwin*-clang++' \( -type f -o -type l \) | head -n 1)"
    if [ -z "$cc_path" ] || [ -z "$cxx_path" ]; then
        echo "Could not find conda-forge macOS clang compiler wrappers in $POLYPIX_DEPS_PREFIX/bin" >&2
        exit 1
    fi
    ln -sf "$cc_path" "$POLYPIX_DEPS_PREFIX/bin/polypix-cc"
    ln -sf "$cxx_path" "$POLYPIX_DEPS_PREFIX/bin/polypix-cxx"
fi
