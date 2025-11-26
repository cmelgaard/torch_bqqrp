"""
bqrrp_demo.py

Simple sanity-check example for the RandLAPACK BQRRP binding.
"""

import torch, os
import numpy as np
from scipy.linalg import lapack as lapack_sc
from torch_bqrrp import bqrrp

def explicit_q_from_lapack(A_factored: torch.Tensor,
                           tau: torch.Tensor):
    """
    Faster Q reconstruction using LAPACK's ORGQR via SciPy.

    A_factored: m x n torch tensor containing LAPACK-style QR factors
    tau: k vector of Householder scalars

    Returns:
      Q_torch: m x k (thin Q) as torch tensor
      R_torch: k x k upper-triangular torch tensor
    """
    assert A_factored.dtype in (torch.float32, torch.float64), "Real dtypes only"
    device = A_factored.device
    m, n = A_factored.shape
    k = min(m, n)

    # Compute R before we let LAPACK overwrite A
    R_torch = torch.triu(A_factored[:k, :k].clone())

    # Move to CPU/NumPy for LAPACK
    A_np = A_factored.detach().cpu().numpy().astype(np.float64, copy=True)
    tau_np = tau.detach().cpu().numpy().astype(np.float64, copy=True)

    # Get the appropriate ORGQR routine (sorgqr/dorgqr) based on dtype
    orgqr, = lapack_sc.get_lapack_funcs(('orgqr',), (A_np,))

    # Call LAPACK: overwrite_a=False to keep A_np if you want; True is fine too
    Q_np, work, info = orgqr(A_np, tau_np, overwrite_a=True)
    if info != 0:
        raise RuntimeError(f"LAPACK orgqr failed with info={info}")

    # Take thin Q (m x k) and convert back to torch/device
    Q_torch = torch.from_numpy(Q_np[:, :k]).to(device=device,
                                               dtype=A_factored.dtype)

    return Q_torch, R_torch

def main():
    use_cuda_flag = os.environ.get("USE_CUDA", "1") == "1"
    if use_cuda_flag and torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    dtype = torch.float32

    m, n = 5000, 2800
    block_size = 900
    d_factor = 1.0
    d = int(block_size * d_factor)
    print(f"Device: {device}, dtype: {dtype}")
    print(f"Matrix size: m={m}, n={n}, block_size={block_size}, d={d}")

    A = torch.randn(m, n, device=device, dtype=dtype)

    #A_factored, tau, J = bqrrp(A, block_size=block_size, d=d,)
    A_factored = torch.zeros_like(A, device=device)
    tau = torch.zeros(n, dtype=dtype, device=device)
    J = torch.ones(n, dtype=torch.int64, device=device)
    J -= 1

    print("BQRRP completed.")
    print(f"A_factored shape: {A_factored.shape}")
    print(f"tau shape: {tau.shape}, J shape: {J.shape}")
    print(f"First 10 pivot indices J: {J[:10].tolist()}")

    Q, R = explicit_q_from_lapack(A_factored, tau)
    
    AP = A[:, J]
    QR = Q @ R

    rel_err = torch.linalg.norm(AP - QR) / torch.linalg.norm(A)
    orth_err = torch.linalg.norm(Q.T @ Q - torch.eye(Q.shape[1], device=device, dtype=dtype))

    print("\n=== BQRRP Error Metrics ===")
    print(f"REL NORM OF A P - Q R:   {rel_err.item():.3e}")
    print(f"FRO NORM OF (Q^T Q - I): {orth_err.item():.3e}")
    print("===========================")

    del A, A_factored, tau, J, Q, R, AP, QR, rel_err, orth_err


if __name__ == "__main__":
    main()
    import gc
    gc.collect()
    import time
    time.sleep(2)
    main()
    import gc
    gc.collect()
    import time
    time.sleep(2)
    main()
