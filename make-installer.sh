#!/bin/sh

TAG=4.1.0.$(date +%Y%m%d)

git tag -a v$TAG -m "Tesseract $TAG"

ARCHS="i686 x86_64 x86_64"

./autogen.sh

for ARCH in $ARCHS; do
  HOST=$ARCH-w64-mingw32

  rm -rf bin/ndebug/$HOST
  mkdir -p bin/ndebug/$HOST
  (
  cd bin/ndebug/$HOST
  ../../../configure --host=$HOST --prefix=/usr/$HOST CXXFLAGS="-fno-math-errno -Wall -Wextra -Wpedantic -g -O2"
  make install-jars install training-install winsetup prefix=$PWD/usr/$HOST
  )
done
