#!/usr/bin/env bash
set -e

ROOT="$(pwd)"
DEPS_SRC="$ROOT/deps/src"
DEPS_INSTALL="$ROOT/deps/install"

# Ensure deps dirs exist
mkdir -p "$DEPS_SRC"
mkdir -p "$DEPS_INSTALL"

###############################
# Install Random123 (header-only)
###############################
cd "$DEPS_SRC"
if [ ! -d "$DEPS_SRC/Random123" ]; then
    git clone https://github.com/DEShawResearch/random123.git Random123
fi

mkdir -p "$DEPS_INSTALL/include"
# Copy only the headers that RandBLAS / RandLAPACK expect: Random123/*.h
cp -R "$DEPS_SRC/Random123/include/Random123" "$DEPS_INSTALL/include/"

###############################
# Install BLAS++ and LAPACK++
###############################
git clone https://github.com/icl-utk-edu/blaspp.git "$DEPS_SRC/blaspp" || true
mkdir -p "$DEPS_SRC/blaspp/build"
cd "$DEPS_SRC/blaspp/build"

cmake .. \
  -DCMAKE_INSTALL_PREFIX="$DEPS_INSTALL" \
  -DCMAKE_CXX_STANDARD=20 \
  -DBLASPP_BUILD_TESTS=OFF

make -j$(nproc)
make install

git clone https://github.com/icl-utk-edu/lapackpp.git "$DEPS_SRC/lapackpp" || true
mkdir -p "$DEPS_SRC/lapackpp/build"
cd "$DEPS_SRC/lapackpp/build"

cmake .. \
  -DCMAKE_INSTALL_PREFIX="$DEPS_INSTALL" \
  -DLAPACKPP_BUILD_TESTS=OFF \
  -DCMAKE_CXX_STANDARD=20 \
  -DBLASPP_DIR="$DEPS_INSTALL/lib/cmake/blaspp"

make -j$(nproc)
make install

###############################
# Install RandLAPACK
###############################
cd "$DEPS_SRC"
git clone https://github.com/BallisticLA/RandLAPACK.git || true
(cd RandLAPACK && git submodule update --init --recursive)

mkdir -p RandLAPACK/build
cd RandLAPACK/build

cmake .. \
  -DCMAKE_INSTALL_PREFIX="$DEPS_INSTALL" \
  -DCMAKE_CXX_STANDARD=20 \
  -DRandom123_DIR="$DEPS_INSTALL/include" \
  -DUSE_CUDA=ON \
  -DCUDAHOSTCXX=g++-12

make -j$(nproc)
make install

echo "Dependencies installed successfully."
