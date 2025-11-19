"""
bqrrp_demo.py

Simple sanity-check example for the RandLAPACK BQRRP binding.
"""

import torch
from torch_bqrrp import bqrrp


def main():
    use_cuda_flag = os.environ.get("USE_CUDA", "1") == "1"
    if use_cuda_flag and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    dtype = torch.float64

    m, n = 5000, 2800
    block_size = 900
    d_factor = 1.0
    d = int(block_size * d_factor)
    print(f"Device: {device}, dtype: {dtype}")
    print(f"Matrix size: m={m}, n={n}, block_size={block_size}, d={d}")

    A = torch.randn(m, n, device=device, dtype=dtype)

    A_factored, tau, J = bqrrp(A, block_size=block_size, d=d)

    print("BQRRP completed.")
    print(f"A_factored shape: {A_factored.shape}")
    print(f"tau shape: {tau.shape}, J shape: {J.shape}")
    print(f"First 10 pivot indices J: {J[:10].tolist()}")


if __name__ == "__main__":
    main()
