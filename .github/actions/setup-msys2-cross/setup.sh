#!/bin/bash

# Set up a cross build environment for Windows on Debian or Ubuntu.
#
# - llvm-mingw (https://github.com/mstorsjo/llvm-mingw) provides the cross
#   compiler, the linker and the binutils replacements.
# - The pacman repositories of MSYS2 (https://www.msys2.org/) provide the
#   C runtime, the C++ standard library and any other library for Windows.
#
# The MSYS2 packages are installed in the same directory as on Windows
# (for example /clang64). That directory is the sysroot of the compiler,
# so the cross build uses the same headers and libraries as a native build
# in that MSYS2 environment.
#
# Usage: setup.sh [MSYSTEM [PACKAGE...]]
#
# MSYSTEM is CLANG64 (default), CLANGARM64 or UCRT64. Packages are given
# without the prefix of the environment, for example "leptonica" instead of
# "mingw-w64-clang-x86_64-leptonica".
#
# Environment variables:
#
# LLVM_MINGW_VERSION  release of llvm-mingw
# LLVM_MINGW_SHA256   checksum of the release archive (optional)
# LLVM_MINGW_URL      URL of the release archive (optional)
# LLVM_MINGW_DIR      installation directory, default /opt/llvm-mingw
# MSYS2_CROSS_DIR     directory for the wrapper scripts, default /opt/msys2-cross

# Author: Stefan Weil (2026)
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

MSYSTEM=${1:-CLANG64}
shift || true

# Flags which select the C++ standard library and the compiler runtime
# of the MSYS2 environment.
LIBCXX_FLAGS="-stdlib=libc++ -rtlib=compiler-rt -unwindlib=libunwind"
LIBSTDCXX_FLAGS="-stdlib=libstdc++ -rtlib=libgcc -unwindlib=libgcc -pthread"

case "${MSYSTEM^^}" in
CLANG64)
  HOST=x86_64-w64-mingw32
  PACKAGE_PREFIX=mingw-w64-clang-x86_64
  BASE_PACKAGES="crt headers libc++ libunwind libwinpthread winpthreads"
  FLAGS=$LIBCXX_FLAGS
  ;;
CLANGARM64)
  HOST=aarch64-w64-mingw32
  PACKAGE_PREFIX=mingw-w64-clang-aarch64
  BASE_PACKAGES="crt headers libc++ libunwind libwinpthread winpthreads"
  FLAGS=$LIBCXX_FLAGS
  ;;
UCRT64)
  HOST=x86_64-w64-mingw32
  PACKAGE_PREFIX=mingw-w64-ucrt-x86_64
  # The GCC package provides libstdc++ and libgcc. Its compiler is not used.
  BASE_PACKAGES="gcc"
  FLAGS=$LIBSTDCXX_FLAGS
  ;;
*)
  echo "$0: unsupported MSYS2 environment '$MSYSTEM'" \
    "(supported: CLANG64, CLANGARM64, UCRT64)" >&2
  exit 1
  ;;
esac

REPO=${MSYSTEM,,}
PREFIX=/$REPO

# The default release of llvm-mingw with the checksums of its archives.
LLVM_MINGW_DEFAULT_VERSION=20261117
declare -A LLVM_MINGW_DEFAULT_SHA256=(
  [aarch64]=6d9442ead9ace3b3e4c3389bfef69f93f873305e0705e6cc60978ac895320872
  [x86_64]=e7a15a49a2ea95e9f3d8a9d0cfabe5052a4b10fb4777cc54de5a8c27b64bb701
)

ARCH=$(uname -m)
LLVM_MINGW_VERSION=${LLVM_MINGW_VERSION:-$LLVM_MINGW_DEFAULT_VERSION}
LLVM_MINGW_DIR=${LLVM_MINGW_DIR:-/opt/llvm-mingw}
LLVM_MINGW_SHA256=${LLVM_MINGW_SHA256:-}
LLVM_MINGW_URL=${LLVM_MINGW_URL:-}
if [ -z "$LLVM_MINGW_URL" ]; then
  LLVM_MINGW_URL=https://github.com/mstorsjo/llvm-mingw/releases/download/$LLVM_MINGW_VERSION/llvm-mingw-$LLVM_MINGW_VERSION-ucrt-ubuntu-22.04-$ARCH.tar.xz
  if [ -z "$LLVM_MINGW_SHA256" ] &&
    [ "$LLVM_MINGW_VERSION" = "$LLVM_MINGW_DEFAULT_VERSION" ]; then
    LLVM_MINGW_SHA256=${LLVM_MINGW_DEFAULT_SHA256[$ARCH]:-}
  fi
fi
MSYS2_CROSS_DIR=${MSYS2_CROSS_DIR:-/opt/msys2-cross}
BINDIR=$MSYS2_CROSS_DIR/$REPO/bin

if [ "$(id -u)" -eq 0 ]; then
  SUDO=
else
  SUDO=sudo
fi

set -x

# Install the required packages of the Linux distribution.
$SUDO apt-get update --quiet
$SUDO apt-get install --assume-yes --no-install-recommends --quiet \
  ca-certificates curl gnupg makepkg pacman-package-manager pkg-config \
  xz-utils zstd

# Install llvm-mingw.
if [ ! -x "$LLVM_MINGW_DIR/bin/$HOST-clang" ]; then
  archive=$(mktemp)
  curl --fail --location --retry 3 --silent --show-error \
    --output "$archive" "$LLVM_MINGW_URL"
  if [ -n "$LLVM_MINGW_SHA256" ]; then
    echo "$LLVM_MINGW_SHA256  $archive" | sha256sum --check --quiet
  fi
  $SUDO mkdir -p "$LLVM_MINGW_DIR"
  $SUDO tar --extract --xz --file "$archive" --directory "$LLVM_MINGW_DIR" \
    --strip-components=1 --no-same-owner
  rm -f -- "$archive"
fi

# Add the MSYS2 environment to the pacman configuration.
# See https://www.msys2.org/docs/repos-mirrors/.
MSYS2_RAW=https://raw.githubusercontent.com/msys2
$SUDO mkdir -p /etc/pacman.d /usr/share/keyrings
$SUDO curl --fail --location --retry 3 --silent --show-error \
  --output /etc/pacman.d/mirrorlist.mingw \
  $MSYS2_RAW/MSYS2-packages/master/pacman-mirrors/mirrorlist.mingw
if ! grep -q "^\[$REPO\]" /etc/pacman.conf; then
  printf '\n[%s]\nInclude = /etc/pacman.d/mirrorlist.mingw\n' "$REPO" |
    $SUDO tee -a /etc/pacman.conf >/dev/null
fi

# Add the MSYS2 keys for pacman.
for file in msys2.gpg msys2-revoked msys2-trusted; do
  $SUDO curl --fail --location --retry 3 --silent --show-error \
    --output /usr/share/keyrings/$file \
    $MSYS2_RAW/MSYS2-keyring/master/$file
done
$SUDO pacman-key --init
$SUDO pacman-key --populate msys2

# Install the MSYS2 packages.
packages=()
for package in $BASE_PACKAGES "$@"; do
  case "$package" in
  mingw-w64-*) packages+=("$package") ;;
  *) packages+=("$PACKAGE_PREFIX-$package") ;;
  esac
done
$SUDO pacman --sync --refresh --sysupgrade --needed --noconfirm \
  "${packages[@]}"

# libtool before version 2.5 does not know the builtins library of
# compiler-rt and omits it when it links a shared C++ library, which then
# fails with undefined symbols like ___chkstk_ms. Therefore that library
# is always passed to the linker.
case "$FLAGS" in
*-rtlib=compiler-rt*)
  FLAGS+=" -Wl,$("$LLVM_MINGW_DIR/bin/$HOST-clang" -print-libgcc-file-name)"
  ;;
esac

# Add wrapper scripts for the compiler which use the MSYS2 environment
# as sysroot, and for pkg-config which then only finds MSYS2 packages.
$SUDO mkdir -p "$BINDIR"
for tool in c++ cc clang clang++ g++ gcc; do
  $SUDO tee "$BINDIR/$HOST-$tool" >/dev/null <<EOF
#!/bin/sh
exec "$LLVM_MINGW_DIR/bin/$HOST-$tool" --start-no-unused-arguments \\
  --sysroot="$PREFIX" $FLAGS --end-no-unused-arguments "\$@"
EOF
done
$SUDO tee "$BINDIR/$HOST-pkg-config" >/dev/null <<EOF
#!/bin/sh
PKG_CONFIG_LIBDIR="$PREFIX/lib/pkgconfig:$PREFIX/share/pkgconfig"
export PKG_CONFIG_LIBDIR
exec pkg-config "\$@"
EOF
$SUDO chmod +x "$BINDIR"/*

set +x

if [ -n "${GITHUB_PATH:-}" ]; then
  # The last entry comes first in PATH.
  echo "$LLVM_MINGW_DIR/bin" >>"$GITHUB_PATH"
  echo "$BINDIR" >>"$GITHUB_PATH"
fi
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  {
    echo "host=$HOST"
    echo "prefix=$PREFIX"
    echo "package-prefix=$PACKAGE_PREFIX"
    echo "path=$BINDIR:$LLVM_MINGW_DIR/bin"
  } >>"$GITHUB_OUTPUT"
fi

cat <<EOF

Cross build environment for MSYS2 $MSYSTEM:

  host triplet    $HOST
  prefix          $PREFIX
  package prefix  $PACKAGE_PREFIX
  PATH            $BINDIR:$LLVM_MINGW_DIR/bin

EOF
"$BINDIR/$HOST-clang" --version
