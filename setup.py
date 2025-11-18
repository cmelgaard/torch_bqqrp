#!/usr/bin/env python

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# -------------------------------------------------------------------
# Hard-set compilers: FORCE gcc/g++-11 everywhere
# -------------------------------------------------------------------

os.environ["USE_CUDA"] = "1"
os.environ["CC"] = "gcc-11"
os.environ["CXX"] = "g++-11"
os.environ["CUDAHOSTCXX"] = "g++-11"

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

ROOT = Path(__file__).parent.resolve()
DEPS_INSTALL = ROOT / "deps" / "install"
DEPS_SRC = ROOT / "deps" / "src"
DEPS_SRC_RL = DEPS_SRC / "RandLAPACK"  # for rl_cuda_kernels.cuh, etc.


def ensure_path(p: Path, what: str) -> str:
    if not p.exists():
        raise RuntimeError(f"{what} not found: {p}")
    return str(p)


# Core include dirs – this is the important part
include_dirs = [
    ensure_path(DEPS_INSTALL / "include", "deps/install/include"),
    # RandBLAS / RandLAPACK hierarchy
    ensure_path(DEPS_INSTALL / "include" / "RandBLAS", "RandBLAS headers"),
    ensure_path(DEPS_INSTALL / "include" / "RandLAPACK", "RandLAPACK headers"),
    ensure_path(DEPS_INSTALL / "include" / "RandLAPACK" / "drivers", "RandLAPACK drivers"),
    ensure_path(DEPS_INSTALL / "include" / "RandLAPACK" / "misc", "RandLAPACK misc"),
    ensure_path(DEPS_INSTALL / "include" / "RandLAPACK" / "gpu_functions", "RandLAPACK gpu_functions"),
    # Random123 + BLAS++ / LAPACK++
    ensure_path(DEPS_INSTALL / "include" / "Random123", "Random123 headers"),
    ensure_path(DEPS_INSTALL / "include" / "blas", "BLAS++ headers"),
    ensure_path(DEPS_INSTALL / "include" / "lapack", "LAPACK++ headers"),
    # Needed so that RandLAPACK/gpu_functions/rl_cuda_kernels.cuh resolves
    ensure_path(DEPS_SRC_RL, "RandLAPACK source tree (for rl_cuda_kernels.cuh)"),
]

library_dirs = [
    ensure_path(DEPS_INSTALL / "lib", "deps/install/lib"),
]

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
# Compiler flags (with AVX512 FP16 + libstdc++ SIMD workarounds)
# -------------------------------------------------------------------

extra_compile_args = {
    "cxx": [
        "-O3",
        "-fopenmp",
        "-std=c++20",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        "-mno-avx512fp16",          # avoid AVX512 FP16 intrinsics
        "-D_GLIBCXX_SIMD_ENABLE=0", # disable libstdc++ experimental SIMD
    ],
    "nvcc": [
        "-O3",
        "-std=c++20",
        "--expt-relaxed-constexpr",
        "-Xcompiler=-fPIC",
        "-Xcompiler=-mno-avx512fp16",
        "-Xcompiler=-D_GLIBCXX_SIMD_ENABLE=0",
        "-gencode=arch=compute_52,code=sm_52",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        # *** KEY: force nvcc to use g++-11 as host, not gcc/g++-12 ***
        "-ccbin",
        "g++-11",
    ],
}

# -------------------------------------------------------------------
# Custom BuildExtension: add torch paths & rpaths, set arch list
# -------------------------------------------------------------------


class TorchCUDAExtensionBuilder(BuildExtension):
    """
    Custom BuildExtension that:
      * imports torch lazily
      * adds torch's include + lib dirs
      * sets runtime_library_dirs so deps/install/lib and torch's lib are found
      * sets TORCH_CUDA_ARCH_LIST based on the actual GPU (if available)
    """

    def build_extensions(self):
        import torch

        torch_includes = torch.utils.cpp_extension.include_paths()
        torch_lib_dir = Path(torch.__file__).parent / "lib"

        for ext in self.extensions:
            # include dirs: our deps + torch includes
            ext.include_dirs.extend(torch_includes)

            # library dirs: deps + torch libs
            ext.library_dirs.extend([
                str(torch_lib_dir),
            ])

            # runtime search path so the extension can find shared libs at import
            rpaths = list(getattr(ext, "runtime_library_dirs", []) or [])
            deps_lib_dir = str(DEPS_INSTALL / "lib")
            if deps_lib_dir not in rpaths:
                rpaths.append(deps_lib_dir)
            if str(torch_lib_dir) not in rpaths:
                rpaths.append(str(torch_lib_dir))
            ext.runtime_library_dirs = rpaths

            # Set CUDA arch list based on current GPU (if CUDA is available)
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
    "csrc/bqrrp_gpu.cu",
]

ext = CUDAExtension(
    name="torch_bqrrp._bqrrp",
    sources=sources,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_compile_args=extra_compile_args,
)

# -------------------------------------------------------------------
# setup()
# -------------------------------------------------------------------

setup(
    name="torch_bqrrp",
    version="0.1.0",
    description="Torch bindings for RandLAPACK BQRRP",
    packages=["torch_bqrrp"],
    ext_modules=[ext],
    cmdclass={"build_ext": TorchCUDAExtensionBuilder},
    zip_safe=False,
)
