#include <torch/extension.h>
#include <ATen/ATen.h>

#include <tuple>

#include "rl_bqrrp_gpu.hh"  // RandLAPACK::BQRRP_GPU<T,RNG>
#include "rl_blaspp.hh"
#include <RandBLAS.hh>
#include <Random123/philox.h>

using torch::Tensor;

std::tuple<Tensor, Tensor, Tensor> bqrrp_cuda(
    const Tensor& A_in,
    int64_t block_size,
    int64_t d)
{
    TORCH_CHECK(A_in.is_cuda(), "bqrrp_cuda: A must be a CUDA tensor");
    TORCH_CHECK(A_in.dim() == 2, "bqrrp_cuda: A must be 2D");

    const int64_t m = A_in.size(0);
    const int64_t n = A_in.size(1);

    TORCH_CHECK(m >= n, "bqrrp_cuda: currently expects m >= n");
    TORCH_CHECK(d > 0, "bqrrp_cuda: d must be positive");

    // Work copy of A on device
    Tensor A = A_in.clone().contiguous();

    auto opts = A.options();
    Tensor tau      = torch::empty({n}, opts);
    Tensor J        = torch::empty({n}, opts.dtype(torch::kInt64));
    Tensor A_sk_dev = torch::empty({d, n}, opts);  // sketch on device

    using scalar_t = float;
    using RNG      = r123::Philox4x32;

    // 1. Build sketch on CPU using RandBLAS (S * A_cpu)
    Tensor A_cpu    = A_in.to(torch::kCPU).contiguous();
    Tensor A_sk_cpu = torch::empty({d, n}, A_cpu.options());

    scalar_t* A_cpu_ptr    = A_cpu.data_ptr<scalar_t>();
    scalar_t* A_sk_cpu_ptr = A_sk_cpu.data_ptr<scalar_t>();

    int64_t m_cpu = A_cpu.size(0);
    int64_t n_cpu = A_cpu.size(1);
    int64_t lda_cpu = m_cpu;
    int64_t d_ll = d;

    // S ~ Gaussian(d x m), A_sk = S * A
    scalar_t* S = new scalar_t[d_ll * m_cpu]();
    RandBLAS::DenseDist Ddist(d_ll, m_cpu);
    RandBLAS::RNGState<RNG> state;
    state = RandBLAS::fill_dense(Ddist, S, state);

    blas::gemm(
	blas::Layout::ColMajor,
	blas::Op::NoTrans,
	blas::Op::NoTrans,
	d_ll, n_cpu, m_cpu,
	(scalar_t)1.0,
	S, d_ll,
	A_cpu_ptr, lda_cpu,
	(scalar_t)0.0,
	A_sk_cpu_ptr, d_ll
    );
    delete[] S;

    // Copy sketch to device
    A_sk_dev.copy_(A_sk_cpu.to(opts.device()));

    // 2. Call GPU BQRRP with sketch on device
    Tensor A_cm = A.to(torch::kCUDA).transpose(0, 1).contiguous();  // (n, m), CUDA tensor
    scalar_t* A_dev_ptr    = A_cm.data_ptr<scalar_t>();
    scalar_t* A_sk_dev_ptr = A_sk_dev.data_ptr<scalar_t>();
    scalar_t* tau_ptr      = tau.data_ptr<scalar_t>();
    int64_t*  J_ptr        = J.data_ptr<int64_t>();

    int64_t lda_dev = m;

    RandLAPACK::BQRRP_GPU<scalar_t, RNG> alg(
	/*time_subroutines=*/false,
	block_size
    );

    int info = alg.call(
	m,             // m
	n,             // n
	A_dev_ptr,     // A
	lda_dev,       // lda
	A_sk_dev_ptr,  // A_sk
	d_ll,          // d
	tau_ptr,       // tau
	J_ptr          // J
    );

    TORCH_CHECK(info == 0,
		"RandLAPACK::BQRRP_GPU.call returned ", info);

    Tensor A_fact = A_cm.transpose(0, 1).contiguous();  // (m, n) CUDA, row-major
    A = A_fact;

    return std::make_tuple(A, tau, J);
}
