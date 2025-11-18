#!/usr/bin/env bash
set -e

ROOT="$(pwd)"
DEPS_SRC="$ROOT/deps/src"
DEPS_INSTALL="$ROOT/deps/install"

mkdir -p "$DEPS_SRC"
mkdir -p "$DEPS_INSTALL"

###############################
# System dependencies (apt-based)
###############################
if command -v apt-get >/dev/null 2>&1; then
    echo "[install.sh] Installing/ensuring system dependencies via apt-get..."
    sudo apt-get update
    sudo apt-get install -y \
      build-essential \
      cmake \
      git \
      ninja-build \
      wget \
      curl \
      python3-dev \
      python3-pip \
      python3-setuptools \
      python3-wheel \
      libopenblas-dev \
      liblapack-dev \
      liblapacke-dev
else
    echo "[install.sh] WARNING: apt-get not found."
    echo "  Please ensure the following are installed manually:"
    echo "    - build-essential (gcc, g++)"
    echo "    - cmake, git, ninja-build, wget, curl"
    echo "    - python3-dev, python3-pip, setuptools, wheel"
    echo "    - libopenblas-dev, liblapack-dev, liblapacke-dev"
fi

###############################
# Parallel jobs
###############################
if command -v nproc >/dev/null 2>&1; then
    NPROC="$(nproc)"
else
    NPROC=4
fi

###############################
# Install Random123 (header-only)
###############################
cd "$DEPS_SRC"
if [ ! -d "$DEPS_SRC/Random123" ]; then
    echo "[install.sh] Cloning Random123..."
    git clone https://github.com/DEShawResearch/random123.git Random123
else
    echo "[install.sh] Random123 already present."
fi

mkdir -p "$DEPS_INSTALL/include"
# Copy only the headers that RandBLAS / RandLAPACK expect: Random123/*.h
cp -R "$DEPS_SRC/Random123/include/Random123" "$DEPS_INSTALL/include/"

###############################
# Install BLAS++
###############################
if [ ! -d "$DEPS_SRC/blaspp/.git" ]; then
    echo "[install.sh] Cloning BLAS++..."
    git clone https://github.com/icl-utk-edu/blaspp.git "$DEPS_SRC/blaspp"
else
    echo "[install.sh] BLAS++ already present."
fi

mkdir -p "$DEPS_SRC/blaspp/build"
cd "$DEPS_SRC/blaspp/build"

cmake .. \
  -DCMAKE_INSTALL_PREFIX="$DEPS_INSTALL" \
  -DCMAKE_CXX_STANDARD=20 \
  -DBLASPP_BUILD_TESTS=OFF

make -j"$NPROC" || make
make install

###############################
# Install LAPACK++
###############################
if [ ! -d "$DEPS_SRC/lapackpp/.git" ]; then
    echo "[install.sh] Cloning LAPACK++..."
    git clone https://github.com/icl-utk-edu/lapackpp.git "$DEPS_SRC/lapackpp"
else
    echo "[install.sh] LAPACK++ already present."
fi

mkdir -p "$DEPS_SRC/lapackpp/build"
cd "$DEPS_SRC/lapackpp/build"

cmake .. \
  -DCMAKE_INSTALL_PREFIX="$DEPS_INSTALL" \
  -DLAPACKPP_BUILD_TESTS=OFF \
  -DCMAKE_CXX_STANDARD=20 \
  -DBLASPP_DIR="$DEPS_INSTALL/lib/cmake/blaspp"

make -j"$NPROC" || make
make install

###############################
# Install RandLAPACK
###############################
cd "$DEPS_SRC"

if [ ! -d "$DEPS_SRC/RandLAPACK/.git" ]; then
    echo "[install.sh] Cloning RandLAPACK..."
    git clone https://github.com/BallisticLA/RandLAPACK.git
else
    echo "[install.sh] RandLAPACK already present."
fi

# IMPORTANT: keep the recursive submodule update
(
  cd RandLAPACK
  git submodule update --init --recursive
)

mkdir -p RandLAPACK/build
cd RandLAPACK/build

cmake .. \
  -DCMAKE_INSTALL_PREFIX="$DEPS_INSTALL" \
  -DCMAKE_CXX_STANDARD=20 \
  -DRandom123_DIR="$DEPS_INSTALL/include" \
  -DUSE_CUDA=ON \
  -DCUDAHOSTCXX=g++-12

make -j"$NPROC" || make
make install

echo "Dependencies installed successfully."
