from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="torch_bqrrp",
    version="0.1.0",
    packages=["torch_bqrrp"],
    ext_modules=[
        CUDAExtension(
            name="torch_bqrrp._bqrrp",
            sources=[
                "csrc/bqrrp_binding.cpp",
                "csrc/bqrrp_cpu.cpp",
                "csrc/bqrrp_cuda.cu",
            ],
            include_dirs=[
                # TODO: point these at your actual RandLAPACK / RandBLAS install:
                "/path/to/RandLAPACK",
                "/path/to/RandBLAS",
            ],
            libraries=[
                "RandLAPACK",
                "RandBLAS",
                # and whatever BLAS/LAPACK backend RandLAPACK was linked against:
                # e.g. "blaspp", "lapackpp", "blas", "lapack", "mkl_rt"
            ],
            library_dirs=[
                "/path/to/RandLAPACK/lib",
                "/path/to/RandBLAS/lib",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
