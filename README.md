# torch_bqrrp + LISAO

LISAO (Lisao) is a family of optimizers where **momentum updates are regularized
via QR/QRCP-style factorizations**. In this repo, the QR/QRCP step is implemented
using **RandLAPACK's BQRRP** (blocked randomized QR with pivoting), and exposed
to PyTorch as:

- a low-level binding: `torch_bqrrp.bqrrp`
- high-level optimizers: `Lisao`, `SingleDeviceLisao`, and `LisaoWithAuxAdam`
  in `lisao.py`.

The project provides:

- a **C++/CUDA extension** that calls RandLAPACK `BQRRP` / `BQRRP_GPU`
- Python-side wrappers
- example scripts for:
  - a simple 5000×2800 matrix factorization demo
  - CIFAR-10 training using `SingleDeviceLisao`

## Repository Layout

```text
torch_bqrrp/
├── README.md
├── setup.py
├── lisao.py                  # LISAO optimizers (QR-regularized momentum using BQRRP)
├── torch_bqrrp/
│   ├── __init__.py
│   └── bqrrp.py              # Python wrapper around C++/CUDA RandLAPACK BQRRP
├── csrc/
│   ├── bqrrp_binding.cpp     # PyTorch/pybind11 front-end, dispatch CPU vs CUDA
│   ├── bqrrp_cpu.cpp         # CPU BQRRP using RandLAPACK::BQRRP
│   └── bqrrp_cuda.cu         # GPU BQRRP using RandLAPACK::BQRRP_GPU
└── examples/
    ├── bqrrp_demo.py         # Simple 5000x2800 demo of BQRRP binding
    └── airbench94_lisao.py   # CIFAR10 training script using SingleDeviceLisao
```

## Dependencies

You will need:

- **Python ≥ 3.8**
- **PyTorch** with C++ extension support
- **CMake**, a C++17-capable compiler
- **RandBLAS** and **RandLAPACK**, built and installed
- A **BLAS/LAPACK backend** (e.g., OpenBLAS, MKL, BLAS++/LAPACK++)
- **(Optional, GPU)**: NVIDIA **CUDA**, **cuBLAS**, and **cuSOLVER**

> ⚠️ **Note on GPUs:**  
> RandLAPACK’s `BQRRP_GPU` backend is **CUDA-only**.  
> This means the GPU path works **only on NVIDIA GPUs**.  
> On Apple M-series (MPS) or any non-CUDA GPU, only the **CPU BQRRP** path is available.

---

## Installing RandBLAS and RandLAPACK

For full installation details (including BLAS++/LAPACK++), see:  
👉 **[INSTALL.md](./INSTALL.md)**

The exact commands may vary by system, but the typical build steps are:

### 1. Install RandBLAS

```bash
git clone https://github.com/BallisticLA/RandBLAS.git
cd RandBLAS
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
sudo make install     # or set CMAKE_INSTALL_PREFIX to a custom path
```

### 2. Install RandLAPACK

```bash
git clone https://github.com/BallisticLA/RandLAPACK.git
cd RandLAPACK
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
sudo make install
```

Make sure the install prefixes for both libraries are visible to your compiler
and linker (e.g., via `CMAKE_PREFIX_PATH`, `LD_LIBRARY_PATH`, etc.).

---

## Building the PyTorch Extension

From the repository root:

```bash
python setup.py develop
# or
python setup.py install
```

If RandLAPACK/RandBLAS were installed in a non-standard location, update  
`include_dirs` and `library_dirs` in `setup.py` to point to your install paths.

---

## Usage

### 1. Low-level BQRRP binding

The simplest entry point is the `bqrrp` function:

```python
import torch
from torch_bqrrp import bqrrp

m, n = 5000, 2800
block_size = 900
d = block_size

A = torch.randn(m, n, device="cuda", dtype=torch.float64)  # or device="cpu"

A_factored, tau, J = bqrrp(A, block_size=block_size, d=d)
```

To run the included demo:

```bash
python examples/bqrrp_demo.py
```

This constructs a `5000 × 2800` matrix, runs BQRRP on CPU or GPU (depending on
availability), and prints basic stats (shapes, pivots, etc.).

---

### 2. LISAO Optimizer (`SingleDeviceLisao`)

`SingleDeviceLisao` behaves like the original Lisao optimizer, but internally:

- Uses **RandLAPACK BQRRP** instead of `torch.geqrf`
- Regularizes the momentum update via a QR/QRCP-style preconditioner

Basic usage:

```python
from lisao import SingleDeviceLisao

model = ...
params = [p for p in model.parameters() if p.requires_grad]
optimizer = SingleDeviceLisao(params, lr=0.02, weight_decay=0.01, momentum=0.95)

for x, y in dataloader:
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()
    optimizer.step()
```

---

### 3. CIFAR-10 / Airbench-style Example

The example script:

```
examples/airbench94_lisao.py
```

trains a CIFAR-10 classifier using `SingleDeviceLisao`:

```bash
python examples/airbench94_lisao.py
```

Environment variables:

- `CIFAR_DATA` (optional): path for CIFAR-10 dataset (default: `./data`)

This script will:

- Download CIFAR-10 if necessary
- Build a small convolutional network
- Train it using LISAO (and optionally compare with SGD)
- Print epoch-by-epoch train and test accuracy
