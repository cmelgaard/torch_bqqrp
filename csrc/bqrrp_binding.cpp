#include <torch/extension.h>
#include <tuple>

using torch::Tensor;

// Implemented in bqrrp_cpu.cpp / bqrrp_cuda.cu
std::tuple<Tensor, Tensor, Tensor> bqrrp_cpu(
    const Tensor& A,
    int64_t block_size,
    int64_t d);

#ifdef TORCH_BQRRP_WITH_CUDA
std::tuple<Tensor, Tensor, Tensor> bqrrp_cuda(
    const Tensor& A,
    int64_t block_size,
    int64_t d);
#endif

std::tuple<Tensor, Tensor, Tensor> bqrrp(
    const Tensor& A,
    int64_t block_size,
    int64_t d)
{
    if (A.is_cuda()) {
#ifdef TORCH_BQRRP_WITH_CUDA
        return bqrrp_cuda(A, block_size, d);
#else
        TORCH_CHECK(
            false,
            "torch_bqrrp was built without CUDA (USE_CUDA=0) but got a CUDA tensor. "
            "Rebuild with USE_CUDA=1 to enable GPU."
        );
#endif
    }
    return bqrrp_cpu(A, block_size, d);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "bqrrp",
        &bqrrp,
        "RandLAPACK BQRRP (CPU/GPU)",
        pybind11::arg("A"),
        pybind11::arg("block_size"),
        pybind11::arg("d")
    );
}
