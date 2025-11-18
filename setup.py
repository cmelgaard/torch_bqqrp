from pathlib import Path
import os

from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

# ---------------------------------------------------------------------
# Paths: repo root and deps/ tree
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

DEPS_INSTALL = ROOT / "deps" / "install"
INCLUDE_ROOT = DEPS_INSTALL / "include"

# RandLAPACK source tree (for rl_cuda_kernels.cuh)
RANDLAPACK_SRC_ROOT = ROOT / "deps" / "src" / "RandLAPACK"

# Installed headers + RandLAPACK source tree
INCLUDE_DIRS = [
    # Installed headers
    str(INCLUDE_ROOT),                             # RandBLAS.hh, RandLAPACK.hh, etc.
    str(INCLUDE_ROOT / "RandLAPACK"),
    str(INCLUDE_ROOT / "RandLAPACK" / "drivers"),
    str(INCLUDE_ROOT / "RandLAPACK" / "misc"),
    str(INCLUDE_ROOT / "RandLAPACK" / "gpu_functions"),

    # Source tree (where rl_cuda_kernels.cuh actually lives):
    #   deps/src/RandLAPACK/RandLAPACK/gpu_functions/rl_cuda_kernels.cuh
    # With this in the include path, the header
    #   #include "RandLAPACK/gpu_functions/rl_cuda_kernels.cuh"
    # resolves correctly.
    str(RANDLAPACK_SRC_ROOT),
]

LIBRARY_DIRS = [
    str(DEPS_INSTALL / "lib"),
]

LIBRARIES = [
    "RandLAPACK",
    "RandBLAS",
    "blaspp",
    "lapackpp",
]


# ---------------------------------------------------------------------
# CUDA toggle: USE_CUDA
#   USE_CUDA unset or "1" -> try to build CUDA extension
#   USE_CUDA = "0"        -> force CPU-only
# ---------------------------------------------------------------------
def want_cuda() -> bool:
    env = os.environ.get("USE_CUDA", "1")
    if env == "0":
        return False

    try:
        if not torch.cuda.is_available():
            print("torch.cuda.is_available() is False -> building CPU-only extension")
            return False
    except Exception:
        return False

    return True


# RandBLAS/RandLAPACK want C++20 for concepts; otherwise they fall back to macros
# like SignedInteger/SparseMatrix that clash with `using` declarations.
CXX_FLAGS = [
    "-O3",
    "-std=c++20",
    "-fopenmp",
    "-DTORCH_API_INCLUDE_EXTENSION_H",
    "-D_GLIBCXX_USE_CXX11_ABI=0",
]

NVCC_FLAGS = [
    "-O3",
    "-std=c++20",
    "--expt-relaxed-constexpr",
]


def make_extension():
    use_cuda = want_cuda()
    cpu_sources = [
        "csrc/bqrrp_cpu.cpp",
        "csrc/bqrrp_binding.cpp",
    ]

    if use_cuda:
        print("** Building torch_bqrrp with CUDA support (csrc/bqrrp_gpu.cu) **")
        sources = cpu_sources + ["csrc/bqrrp_gpu.cu"]
        Extension = CUDAExtension
        extra_compile_args = {
            "cxx": CXX_FLAGS,
            "nvcc": NVCC_FLAGS,
        }
    else:
        print("** Building torch_bqrrp in CPU-only mode **")
        sources = cpu_sources
        Extension = CppExtension
        extra_compile_args = CXX_FLAGS

    return Extension(
        name="torch_bqrrp._bqrrp",
        sources=[str(ROOT / s) for s in sources],
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=LIBRARIES,
        extra_compile_args=extra_compile_args,
    )


setup(
    name="torch_bqrrp",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[make_extension()],
    cmdclass={"build_ext": BuildExtension},
)
