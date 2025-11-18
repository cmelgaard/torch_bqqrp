from pathlib import Path
import os

from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


# -----------------------------------------------------------------------------
# Paths: repo root + deps/install
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DEPS_INSTALL = ROOT / "deps" / "install"

INCLUDE_DIRS = [str(DEPS_INSTALL / "include")]
LIBRARY_DIRS = [str(DEPS_INSTALL / "lib")]

LIBRARIES = [
    "RandLAPACK",
    "RandBLAS",
    # RandLAPACK should already be linked against lapackpp/blaspp/BLAS/LAPACKE
]


# -----------------------------------------------------------------------------
# Decide whether to build with CUDA
#   - USE_CUDA=0   -> force CPU-only
#   - USE_CUDA!=0  -> try CUDA, fall back to CPU-only if not available
# -----------------------------------------------------------------------------
def want_cuda() -> bool:
    env = os.environ.get("USE_CUDA", "1")  # default: CUDA on
    if env == "0":
        return False

    try:
        return torch.cuda.is_available()
    except Exception:
        return False


use_cuda = want_cuda()
ExtensionClass = CUDAExtension if use_cuda else CppExtension

# -----------------------------------------------------------------------------
# Sources and compile flags
# -----------------------------------------------------------------------------
cpu_sources = [
    "csrc/bqrrp_binding.cpp",
    "csrc/bqrrp_cpu.cpp",
]

cuda_sources = [
    "csrc/bqrrp_cuda.cu",
]

sources = list(cpu_sources)
extra_compile_args = {"cxx": ["-O3"]}

if use_cuda:
    sources += cuda_sources
    extra_compile_args["nvcc"] = ["-O3"]


ext_modules = [
    ExtensionClass(
        name="torch_bqrrp._bqrrp",
        sources=sources,
        include_dirs=INCLUDE_DIRS,
        libraries=LIBRARIES,
        library_dirs=LIBRARY_DIRS,
        extra_compile_args=extra_compile_args,
    )
]

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
setup(
    name="torch_bqrrp",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
