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

INCLUDE_DIRS = [
    str(DEPS_INSTALL / "include"),
    str(DEPS_INSTALL / "include" / "RandLAPACK"),
    str(DEPS_INSTALL / "include" / "RandLAPACK" / "drivers"),
]

LIBRARY_DIRS = [str(DEPS_INSTALL / "lib")]

LIBRARIES = [
    "RandLAPACK",
    "RandBLAS",
    # RandLAPACK should already be linked against lapackpp/blaspp/BLAS/LAPACKE
]


# -----------------------------------------------------------------------------
# Decide whether to build with CUDA
#   - USE_CUDA=0   -> force CPU-only
#   - USE_CUDA=1   -> try CUDA if available AND bqrrp_gpu.cu exists
#   - USE_CUDA unset -> default to 1 (try CUDA)
# -----------------------------------------------------------------------------
def want_cuda() -> bool:
    env = os.environ.get("USE_CUDA", "1")  # default: CUDA on
    if env == "0":
        # user explicitly requested CPU-only
        return False

    # If torch doesn't have CUDA, no point trying
    try:
        if not torch.cuda.is_available():
            return False
    except Exception:
        return False

    return True


cuda_source = ROOT / "csrc" / "bqrrp_gpu.cu"
has_cuda_source = cuda_source.is_file()

use_cuda = want_cuda() and has_cuda_source

if use_cuda:
    print("** Building torch_bqrrp with CUDA support (bqrrp_gpu.cu) **")
else:
    if not has_cuda_source:
        print("** Building torch_bqrrp in CPU-only mode (no csrc/bqrrp_gpu.cu found) **")
    else:
        print("** Building torch_bqrrp in CPU-only mode (CUDA disabled or unavailable) **")

ExtensionClass = CUDAExtension if use_cuda else CppExtension

# -----------------------------------------------------------------------------
# Sources and compile flags
# -----------------------------------------------------------------------------
cpu_sources = [
    "csrc/bqrrp_binding.cpp",
    "csrc/bqrrp_cpu.cpp",
]

sources = list(cpu_sources)
extra_compile_args = {"cxx": ["-O3"]}

if use_cuda:
    sources.append("csrc/bqrrp_gpu.cu")
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
