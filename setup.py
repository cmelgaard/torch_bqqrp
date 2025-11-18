#!/usr/bin/env python3
import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# -------------------------------------------------------------------
# Environment: force CUDA + safe host compiler
# -------------------------------------------------------------------

# GPU is required
os.environ.setdefault("USE_CUDA", "1")

# Use GCC/G++ 12 as host for nvcc (CUDA 12.x is happiest with <= 12)
os.environ.setdefault("CUDAHOSTCXX", "g++-12")
os.environ.setdefault("CC", "gcc-12")
os.environ.setdefault("CXX", "g++-12")

# -------------------------------------------------------------------
# Paths for dependencies
# -------------------------------------------------------------------

ROOT = Path(__file__).parent.resolve()
DEPS_INSTALL = ROOT / "deps" / "install"

INCLUDE_BASE   = str(DEPS_INSTALL / "include")
RANDLAPACK_DIR = str(DEPS_INSTALL / "include" / "RandLAPACK")
RANDBLAS_DIR   = str(DEPS_INSTALL / "include" / "RandBLAS")
BLAS_DIR       = str(DEPS_INSTALL / "include" / "blas")
LAPACK_DIR     = str(DEPS_INSTALL / "include" / "lapack")
RAND123_DIR    = str(DEPS_INSTALL / "include" / "Random123")

LIB_DIR = str(DEPS_INSTALL / "lib")

# -------------------------------------------------------------------
# Custom BuildExtension: import torch *only* inside build_extensions
# -------------------------------------------------------------------

class TorchCUDAExtensionBuilder(BuildExtension):
    def build_extensions(self):
        # IMPORTANT: torch is only imported here, *not* at module import time.
        import torch

        # Torch include paths
        torch_includes = torch.utils.cpp_extension.include_paths()

        for ext in self.extensions:
            # Core deps includes
            ext.include_dirs.extend([
                INCLUDE_BASE,
                RANDLAPACK_DIR,
                RANDBLAS_DIR,
                BLAS_DIR,
                LAPACK_DIR,
                RAND123_DIR,
                *torch_includes,
            ])

            # Library dirs: our deps + torch libs
            ext.library_dirs.extend([
                LIB_DIR,
                str(Path(torch.__file__).parent / "lib"),
            ])

            # Runtime search path so the linker can find libs at import time
            rpaths = list(getattr(ext, "runtime_library_dirs", []) or [])
            if LIB_DIR not in rpaths:
                rpaths.append(LIB_DIR)
            ext.runtime_library_dirs = rpaths

            # Set CUDA arch list based on your actual GPU
            if torch.cuda.is_available():
                cc = torch.cuda.get_device_capability()
                os.environ["TORCH_CUDA_ARCH_LIST"] = f"{cc[0]}{cc[1]}"

        super().build_extensions()

# -------------------------------------------------------------------
# Sources
# -------------------------------------------------------------------

sources = [
    "csrc/bqrrp_binding.cpp",
    "csrc/bqrrp_cpu.cpp",
    "csrc/bqrrp_gpu.cu",  # GPU is required
]

# -------------------------------------------------------------------
# Compiler flags
# -------------------------------------------------------------------

extra_compile_args = {
    "cxx": [
        "-O3",
        "-std=c++20",
        "-fopenmp",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
    ],
    "nvcc": [
        "-O3",
        "-std=c++20",
        "--expt-relaxed-constexpr",
        "-Xcompiler=-fPIC",
        # Avoid the AVX512 FP16 header mess with GCC 13 headers
        "-Xcompiler=-mno-avx512fp16",
        # Your TITAN X is compute capability 5.2
        "-gencode=arch=compute_52,code=sm_52",
    ],
}

# -------------------------------------------------------------------
# Libraries to link against
#   NOTE: we KEEP RandLAPACK / RandBLAS to avoid regressing.
# -------------------------------------------------------------------

libraries = [
    "RandLAPACK",
    "RandBLAS",
    "blaspp",
    "lapackpp",
    "c10",
    "torch",
    "torch_cpu",
    "torch_python",
]

# -------------------------------------------------------------------
# CUDA extension definition
# -------------------------------------------------------------------

ext = CUDAExtension(
    name="torch_bqrrp._bqrrp",
    sources=sources,
    include_dirs=[INCLUDE_BASE],  # extended in build_extensions
    library_dirs=[LIB_DIR],       # extended in build_extensions
    libraries=libraries,
    extra_compile_args=extra_compile_args,
)

# -------------------------------------------------------------------
# setup()
# -------------------------------------------------------------------

setup(
    name="torch_bqrrp",
    version="0.1.0",
    packages=["torch_bqrrp"],
    ext_modules=[ext],
    cmdclass={"build_ext": TorchCUDAExtensionBuilder},
    zip_safe=False,
)
