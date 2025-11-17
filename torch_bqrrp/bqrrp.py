import torch
from . import _bqrrp


def bqrrp(
    A: torch.Tensor,
    block_size: int,
    d: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    RandLAPACK BQRRP / BQRRP_GPU wrapper.

    Parameters
    ----------
    A : (m, n) tensor
        Input matrix (CPU or CUDA, float32 or float64).
    block_size : int
        BQRRP block size (b_sz in RandLAPACK).
    d : int
        Sketch dimension. In the RandLAPACK CPU implementation, d = d_factor * block_size.

    Returns
    -------
    A_factored : (m, n) tensor
        In-place factorized version of A (implicit Q, explicit R).
    tau : (n,) tensor
        Householder scalars.
    J : (n,) int64 tensor
        Column pivot indices.
    """
    if A.dim() != 2:
        raise ValueError(f"bqrrp expects a 2D matrix, got {A.shape}")
    if not A.is_floating_point():
        raise TypeError("bqrrp expects a floating-point tensor")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if d <= 0:
        raise ValueError("d must be positive")

    A_out, tau, J = _bqrrp.bqrrp(A, int(block_size), int(d))
    return A_out, tau, J
