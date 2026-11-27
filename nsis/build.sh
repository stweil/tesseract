#!/bin/bash

# GitHub actions - Create Tesseract installer for Windows

# Author: Stefan Weil (2010-2026)

# Usage: nsis/build.sh [HOST [PREFIX]]
#
# The cross compiler for HOST and the libraries for Windows in PREFIX
# must be installed before this script is run. The GitHub workflow uses
# the action https://github.com/stweil/setup-msys2-cross for that. Without
# GitHub Actions, the script setup.sh from that repository does the same.
# For the MSYS2 environment CLANG64 it is run like this:
#
#     setup.sh CLANG64 \
#       curl-winssl giflib icu leptonica libarchive libidn2 openjpeg2 \
#       openssl pango libpng libtiff libwebp
#
# It also shows the directories which must be added to PATH.
#
# Supported combinations of HOST and PREFIX:
#
#     x86_64-w64-mingw32   /clang64      (default)
#     x86_64-w64-mingw32   /ucrt64
#     aarch64-w64-mingw32  /clangarm64   (Windows on ARM)

set -e
set -x

LANG=C.UTF-8

# Host triplet of the cross compiler.
HOST=${1:-x86_64-w64-mingw32}
# Installation directory of the MSYS2 packages.
MINGW=${2:-/clang64}

ROOTDIR=$PWD
TAG=$(cat VERSION).$(date +%Y%m%d)
BUILDDIR=bin/ndebug/$HOST

if [ "$(id -u)" -eq 0 ]; then
  SUDO=
else
  SUDO=sudo
fi

# Install packages.
$SUDO apt-get update --quiet
$SUDO apt-get install --assume-yes --no-install-recommends --quiet \
  asciidoctor ruby-asciidoctor-pdf curl \
  automake libtool make pkg-config default-jdk-headless \
  nsis python3-venv unzip

git config --global user.email "sw@weilnetz.de"
git config --global user.name "Stefan Weil"
git tag -a "v$TAG" -m "Tesseract $TAG"

# Run autogen.
./autogen.sh

# Build Tesseract installer.
mkdir -p "$BUILDDIR" && cd "$BUILDDIR"

# Run configure.
# The compiler uses $MINGW as sysroot, and $HOST-pkg-config only finds
# the packages which are installed there, so no search paths are needed.
# Disable OpenMP (see https://github.com/tesseract-ocr/tesseract/issues/1662).
../../../configure --disable-openmp --host="$HOST" --prefix="/usr/$HOST" \
  CXX="$HOST-clang++" \
  CXXFLAGS="-fno-math-errno -Wall -Wextra -Wpedantic -g -O2"

make all -j$(nproc)
make training -j$(nproc)

MINGW_INSTALL=${PWD}${MINGW}
# Strip the installed files with the cross tool. The strip program of the
# build machine does not always support Windows executables.
make install-jars install training-install html prefix="$MINGW_INSTALL" \
  INSTALL_STRIP_FLAG="-s --strip-program=$HOST-strip"
test -d venv || python3 -m venv venv
source venv/bin/activate
pip install pefile
# Get the DLL files which are needed for the installer. This includes the
# DLL files of the compiler (C++ standard library, unwinder, threads).
# They are part of the MSYS2 environment which was used as sysroot.
mkdir -p dll
ln -svf $("$ROOTDIR/nsis/find_deps.py" --dlldir "$MINGW/bin" "$MINGW_INSTALL"/bin/*.exe "$MINGW_INSTALL"/bin/*.dll) dll/
make winsetup prefix="$MINGW_INSTALL"
