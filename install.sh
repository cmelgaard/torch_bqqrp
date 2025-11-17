#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Config
# ============================================================

# Default: REQUIRE CUDA unless explicitly disabled.
#   - USE_CUDA unset  → treated as 1 (CUDA required)
#   - USE_CUDA=1      → CUDA required
#   - USE_CUDA=0      → CPU-only build
USE_CUDA=${USE_CUDA:-1}

# Repo root = folder where this script lives
REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DEPS_SRC="$REPO_ROOT/deps/src"
DEPS_INSTALL="$REPO_ROOT/deps/install"
NPROC=$(nproc)

mkdir -p "$DEPS_SRC" "$DEPS_INSTALL"

echo "============================================================"
echo " torch_bqrrp install"
echo " REPO_ROOT    = $REPO_ROOT"
echo " DEPS_SRC     = $DEPS_SRC"
echo " DEPS_INSTALL = $DEPS_INSTALL"
echo " USE_CUDA     = $USE_CUDA"
echo "============================================================"

# ============================================================
# 1. System dependencies (Ubuntu)
# ============================================================

echo "[1/7] Installing system packages (requires sudo)..."
sudo apt update -y
sudo apt install -y \
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
  liblapack-dev

# NOTE:
# We DO NOT upgrade pip here to avoid PEP 668 "externally-managed"
# errors on newer Ubuntu. You should manage your Python environment
# (and torch install) yourself, e.g. via venv/conda.

# ============================================================
# 2. CUDA requirement check
# ============================================================

CUDA_CMAKE_FLAGS_BLASPP=""
CUDA_CMAKE_FLAGS_RANDLAPACK=""

if [[ "$USE_CUDA" -eq 0 ]]; then
  echo "[CUDA] CPU-only mode selected (USE_CUDA=0)."
  CUDA_CMAKE_FLAGS_BLASPP="-DBLASPP_ENABLE_CUBLAS=OFF"
  CUDA_CMAKE_FLAGS_RANDLAPACK="-DRequireCUDA=OFF"
else
  echo "[CUDA] CUDA mode REQUIRED (USE_CUDA != 0)."
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: CUDA was requested (USE_CUDA != 0) but 'nvcc' is not in PATH."
    echo "       Either install CUDA, or rerun in CPU-only mode:"
    echo "           USE_CUDA=0 ./install.sh"
    exit 1
  fi
  echo "[CUDA] nvcc detected:"
  nvcc --version || true
  CUDA_CMAKE_FLAGS_BLASPP="-DBLASPP_ENABLE_CUBLAS=ON"
  CUDA_CMAKE_FLAGS_RANDLAPACK="-DRequireCUDA=ON"
fi

# ============================================================
# Helper: clone or reuse repo
# ============================================================

clone_if_missing () {
  local url="$1"
  local dir="$2"
  if [[ -d "$dir/.git" ]]; then
    echo "[git] Using existing repo at $dir"
  else
    echo "[git] Cloning $url into $dir"
    git clone "$url" "$dir"
  fi
}

# ============================================================
# 3. Random123
# ============================================================

echo "[2/7] Installing Random123..."
clone_if_missing "https://github.com/DEShawResearch/random123.git" \
                 "$DEPS_SRC/random123"
cd "$DEPS_SRC/random123"
make install-include PREFIX="$DEPS_INSTALL"

# ============================================================
# 4. BLAS++
# ============================================================

echo "[3/7] Installing BLAS++..."
clone_if_missing "https://github.com/icl-utk-edu/blaspp.git" \
                 "$DEPS_SRC/blaspp"
cd "$DEPS_SRC/blaspp"
mkdir -p build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$DEPS_INSTALL" \
  $CUDA_CMAKE_FLAGS_BLASPP

make -j"$NPROC"
make install

# ============================================================
# 5. LAPACK++
# ============================================================

echo "[4/7] Installing LAPACK++..."
clone_if_missing "https://github.com/icl-utk-edu/lapackpp.git" \
                 "$DEPS_SRC/lapackpp"
cd "$DEPS_SRC/lapackpp"
mkdir -p build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$DEPS_INSTALL" \
  -DBLASPP_DIR="$DEPS_INSTALL/lib/cmake/blaspp"

make -j"$NPROC"
make install

# ============================================================
# 6. RandBLAS
# ============================================================

echo "[5/7] Installing RandBLAS..."
clone_if_missing "https://github.com/BallisticLA/RandBLAS.git" \
                 "$DEPS_SRC/RandBLAS"
cd "$DEPS_SRC/RandBLAS"
mkdir -p build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$DEPS_INSTALL"

make -j"$NPROC"
make install

# ============================================================
# 7. RandLAPACK
# ============================================================

echo "[6/7] Installing RandLAPACK..."
if [[ -d "$DEPS_SRC/RandLAPACK/.git" ]]; then
  echo "[git] Using existing RandLAPACK at $DEPS_SRC/RandLAPACK"
else
  git clone --recursive https://github.com/BallisticLA/RandLAPACK.git \
            "$DEPS_SRC/RandLAPACK"
fi

cd "$DEPS_SRC/RandLAPACK"
mkdir -p build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$DEPS_INSTALL" \
  -DBLASPP_DIR="$DEPS_INSTALL/lib/cmake/blaspp" \
  -DLAPACKPP_DIR="$DEPS_INSTALL/lib/cmake/lapackpp" \
  -DRandom123_DIR="$DEPS_INSTALL/include" \
  $CUDA_CMAKE_FLAGS_RANDLAPACK

make -j"$NPROC"
make install

# ============================================================
# 8. Environment for this build
# ============================================================

export CMAKE_PREFIX_PATH="$DEPS_INSTALL:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="$DEPS_INSTALL/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$DEPS_INSTALL/lib:${LIBRARY_PATH:-}"
export CPATH="$DEPS_INSTALL/include:${CPATH:-}"

# ============================================================
# 9. Build torch_bqrrp extension
# ============================================================

echo "[7/7] Building torch_bqrrp extension..."
cd "$REPO_ROOT"
python3 setup.py develop

echo "Running examples/bqrrp_demo.py to verify..."
python3 examples/bqrrp_demo.py

echo "============================================================"
echo " SUCCESS: torch_bqrrp installed."
echo "   USE_CUDA     = $USE_CUDA"
echo "   DEPS_INSTALL = $DEPS_INSTALL"
echo "============================================================"
echo "You can now run:"
echo "  python3 examples/airbench94_lisao.py"
echo "============================================================"
