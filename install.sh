#!/usr/bin/env bash
set -e

ROOT="$(pwd)"
DEPS_SRC="$ROOT/deps/src"
DEPS_INSTALL="$ROOT/deps/install"

# Ensure deps dirs exist
mkdir -p "$DEPS_SRC"
mkdir -p "$DEPS_INSTALL"

###############################
# Install BLAS++ and LAPACK++
###############################
git clone https://github.com/icl-utk-edu/blaspp.git "$DEPS_SRC/blaspp" || true
mkdir -p "$DEPS_SRC/blaspp/build"
cd "$DEPS_SRC/blaspp/build"

cmake .. \
  -DCMAKE_INSTALL_PREFIX="$DEPS_INSTALL" \
  -DBLASPP_BUILD_TESTS=OFF
make -j$(nproc)
make install

git clone https://github.com/icl-utk-edu/lapackpp.git "$DEPS_SRC/lapackpp" || true
mkdir -p "$DEPS_SRC/lapackpp/build"
cd "$DEPS_SRC/lapackpp/build"

cmake .. \
  -DCMAKE_INSTALL_PREFIX="$DEPS_INSTALL" \
  -DLAPACKPP_BUILD_TESTS=OFF
make -j$(nproc)
make install

###################################
# Install RandBLAS & RandLAPACK
###################################
cd "$DEPS_SRC"
git clone https://github.com/BallisticLA/RandLAPACK.git || true

mkdir -p RandLAPACK/build
cd RandLAPACK/build

cmake .. \
  -DCMAKE_INSTALL_PREFIX="$DEPS_INSTALL" \
  -DCMAKE_CXX_STANDARD=20 \
  -DUSE_CUDA=ON \
  -DCUDAHOSTCXX=g++-12

make -j$(nproc)
make install

echo "Dependencies installed successfully."
