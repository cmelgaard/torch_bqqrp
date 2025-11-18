import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

# Repository root and deps/install paths
_THIS_DIR = Path(__file__).parent.resolve()
_DEPS_INSTALL = _THIS_DIR / "deps" / "install"

include_dirs = [
    str(_DEPS_INSTALL / "include"),
    str(_DEPS_INSTALL / "include" / "RandLAPACK"),
    str(_DEPS_INSTALL / "include" / "RandLAPACK" / "drivers"),
    str(_DEPS_INSTALL / "include" / "RandLAPACK" / "misc"),
    str(_DEPS_INSTALL / "include" / "RandLAPACK" / "gpu_functions"),
]

library_dirs = [
    str(_DEPS_INSTALL / "lib"),
]

# BLAS++ / LAPACK++ / OpenBLAS
libraries = [
    "blaspp",
    "lapackpp",
    "lapacke",
    "openblas",
]

extra_compile_args = {
    "cxx": [
        "-O3",
        "-fopenmp",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
    ],
}

ext_modules = []

# Simple CUDA toggle:
#   USE_CUDA=0  -> CPU-only build
#   anything else (default) -> build with CUDA
use_cuda = os.environ.get("USE_CUDA", "1") != "0"

if use_cuda:
    extra_compile_args["nvcc"] = [
        "-O3",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        "-gencode=arch=compute_52,code=sm_52",
        "-gencode=arch=compute_52,code=compute_52",
    ]

    ext_modules.append(
        CUDAExtension(
            name="torch_bqrrp._bqrrp",
            sources=[
                "csrc/bqrrp_binding.cpp",
                "csrc/bqrrp_cpu.cpp",
                "csrc/bqrrp_gpu.cu",
            ],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=extra_compile_args,
        )
    )
else:
    ext_modules.append(
        CppExtension(
            name="torch_bqrrp._bqrrp",
            sources=[
                "csrc/bqrrp_binding.cpp",
                "csrc/bqrrp_cpu.cpp",
            ],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=extra_compile_args,
        )
    )

setup(
    name="torch_bqrrp",
    version="0.1.0",
    packages=["torch_bqrrp"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
