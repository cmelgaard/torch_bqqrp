import torch
import torch.linalg as tll
import torch.distributed as dist

from torch_bqrrp import bqrrp as bqrrp_factor


def lisao_via_QR(G):
    """
    LISAO core transform: QR/QRCP-style regularization using RandLAPACK BQRRP.

    Conceptually:
        - Factor G (or G^T if G is wide) using BQRRP: X ≈ Q R P^T
        - Work with R, normalize its rows, regularize small directions
        - Solve a triangular system to build a preconditioning matrix X
        - Apply X on the right (or left, depending on aspect ratio)

    This replaces the original torch.geqrf-based implementation in Lisao.
    """
    assert G.ndim >= 2  # last two dims define the matrix

    tol, tol2 = 1.0e-4, 1.0e-8

    # Use the last two dimensions as the matrix; we assume non-batched usage
    X = G
    m, n = X.size(-2), X.size(-1)

    # BQRRP prefers tall matrices (m >= n). If we're wide, factor G^T instead.
    tall = m >= n
    if tall:
        mat = X
    else:
        mat = X.mT  # shape: (n, m)
        m, n = mat.shape

    # Choose a reasonable block size and sketch dimension
    block_size = min(128, n)
    if block_size <= 0:
        block_size = n
    d = block_size  # d_factor = 1.0

    # Force to float32/float64 for LAPACK-style routines
    if mat.dtype not in (torch.float32, torch.float64):
        mat = mat.to(torch.float32)

    # BQRRP factorization: mat ≈ Q R P^T, R stored in upper triangle of A_factored
    A_factored, tau, J = bqrrp_factor(mat, block_size=block_size, d=d)

    # Extract R (n x n) from the first n rows
    R = torch.triu(A_factored[:n, :n], diagonal=0)
    R = R + tol2 * torch.eye(n, dtype=R.dtype, device=R.device)

    # Rowwise norms of R
    D = torch.sqrt(torch.sum(R**2, dim=1))
    mask = D < tol
    D = D.masked_fill(mask, tol)

    # T = R normalized row-wise (T has row norm 1)
    T = R / D.unsqueeze(1)

    # Xmat = T / D (further regularization)
    Xmat = T / D.unsqueeze(1)

    # Solve T * X = Xmat, upper-triangular system
    # (n x n) (n x n) = (n x n)
    Xsolve = tll.solve_triangular(T, Xmat, upper=True)

    # Apply transform depending on aspect ratio
    if tall:
        # G: (m x n), Xsolve: (n x n)
        return G @ Xsolve
    else:
        # G: (m x n), mat = G^T: (n x m), Xsolve: (m x m)
        return Xsolve.mT @ G


def lisao_update(grad, momentum, beta=0.95, nesterov=True):
    """
    LISAO update: momentum + QR-regularization via BQRRP.
    """
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum

    # For conv filters: view as 2D (out_channels, -1)
    if update.ndim == 4:
        update = update.view(len(update), -1)

    update = lisao_via_QR(update)

    # Simple scaling as in the original Lisao implementation
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class Lisao(torch.optim.Optimizer):
    """
    Lisao / LISAO - MomentUm regularized by QR/QRCP (here via BQRRP).

    This is the distributed variant using torch.distributed, mirroring the
    original Lisao design but calling RandLAPACK BQRRP under the hood.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(len(params))[::world_size]:
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)  # force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = lisao_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size],
                                params_pad[base_i + rank])

        return loss


class SingleDeviceLisao(torch.optim.Optimizer):
    """
    Single-device Lisao variant (no torch.distributed).
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = lisao_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


# The Aux-Adam variants below are left structurally identical to your original
# lisao.py; they now indirectly use BQRRP via lisao_update above.

def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


class LisaoWithAuxAdam(torch.optim.Optimizer):
    """
    Distributed Lisao variant that can be used for all parameters in the network,
    running internal AdamW for non-Lisao-compatible params.
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_lisao" in group
            if group["use_lisao"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_lisao"])
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_lisao"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_lisao"]:
                params = group["params"]
                world_size = dist.get_world_size()
                rank = dist.get_rank()
                params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
                for base_i in range(len(params))[::world_size]:
                    if base_i + rank < len(params):
                        p = params[base_i + rank]
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = lisao_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + world_size],
                                    params_pad[base_i + rank])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceLisaoWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of LisaoWithAuxAdam.
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_lisao" in group
            if group["use_lisao"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_lisao"])
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_lisao"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_lisao"]:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = lisao_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
