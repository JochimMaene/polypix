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
        ;;
    Darwin-x86_64)
        micromamba_platform="osx-64"
        ;;
    Darwin-arm64)
        micromamba_platform="osx-arm64"
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

"$micromamba_bin" create -y -p "$POLYPIX_DEPS_PREFIX" -c conda-forge \
    "healpix_cxx=$HEALPIX_CXX_VERSION" \
    "cfitsio=$CFITSIO_VERSION" \
    "pkg-config"
