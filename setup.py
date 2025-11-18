#!/usr/bin/env python3
import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# Force CUDA
os.environ.setdefault("USE_CUDA", "1")

# Force safe host compiler (CUDA 12 requires <= GCC 12)
os.environ["CUDAHOSTCXX"] = "g++-12"
os.environ["CC"] = "gcc-12"
os.environ["CXX"] = "g++-12"

# Extension sources
sources = [
    "csrc/bqrrp_binding.cpp",
    "csrc/bqrrp_cpu.cpp",
    "csrc/bqrrp_gpu.cu",
]

include_dirs = [
    "deps/install/include",
    "deps/install/include/RandLAPACK",
    "deps/install/include/RandLAPACK/drivers",
    "deps/install/include/RandLAPACK/misc",
    "deps/install/include/RandLAPACK/gpu_functions",
    "deps/install/include/RandBLAS",
    "deps/src/RandLAPACK",
]

library_dirs = [
    "deps/install/lib",
]

libraries = [
    "RandLAPACK",
    "RandBLAS",
    "blaspp",
    "lapackpp",
]

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
        "-Xcompiler=-mno-avx512fp16",   # necessary for GCC 13 systems
        "-gencode=arch=compute_52,code=sm_52",
    ],
}

setup(
    name="torch_bqrrp",
    version="0.1.0",
    packages=["torch_bqrrp"],
    ext_modules=[
        CUDAExtension(
            name="torch_bqrrp._bqrrp",
            sources=sources,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
