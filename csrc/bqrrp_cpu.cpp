#include <torch/extension.h>
#include <ATen/ATen.h>

#include <tuple>

// RandLAPACK CPU driver
#include "rl_bqrrp.hh"     // RandLAPACK::BQRRP<T,RNG>
#include <RandBLAS.hh>
#include <Random123/philox.h>

using torch::Tensor;

std::tuple<Tensor, Tensor, Tensor> bqrrp_cpu(
    const Tensor& A_in,
    int64_t block_size,
    int64_t d)
{
    TORCH_CHECK(!A_in.is_cuda(), "bqrrp_cpu: A must be a CPU tensor");
    TORCH_CHECK(A_in.dim() == 2, "bqrrp_cpu: A must be 2D");

    const int64_t m = A_in.size(0);
    const int64_t n = A_in.size(1);

    TORCH_CHECK(m >= n, "bqrrp_cpu: currently expects m >= n");

    // Work copy of A – BQRRP overwrites it with Q/R (GEQP3-style)
    Tensor A = A_in.clone().contiguous();

    auto opts = A.options();
    Tensor tau = torch::empty({n}, opts);
    Tensor J   = torch::empty({n}, opts.dtype(torch::kInt64));

    AT_DISPATCH_FLOATING_TYPES(
        A.scalar_type(),
        "bqrrp_cpu",
        [&]() {
            using scalar_t = scalar_t;
            using RNG      = r123::Philox4x32;

            scalar_t* A_ptr   = A.data_ptr<scalar_t>();
            scalar_t* tau_ptr = tau.data_ptr<scalar_t>();
            int64_t*  J_ptr   = J.data_ptr<int64_t>();

            int64_t lda = m;

            RandLAPACK::BQRRP<scalar_t, RNG> alg(/*time_subroutines=*/false,
                                                 block_size);

            // d_factor = d / block_size (as in RandLAPACK tests)
            scalar_t d_factor =
                static_cast<scalar_t>(
                    static_cast<double>(d) /
                    static_cast<double>(alg.block_size));

            RandBLAS::RNGState<RNG> state;

            int info = alg.call(
                m,          // m
                n,          // n
                A_ptr,      // A
                lda,        // lda
                d_factor,   // d_factor
                tau_ptr,    // tau
                J_ptr,      // J
                state       // RNG state
            );

            TORCH_CHECK(info == 0, "RandLAPACK::BQRRP.call returned ", info);
        });

    return std::make_tuple(A, tau, J);
}
