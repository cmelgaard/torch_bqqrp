# Installation Guide

This project builds the `torch_bqrrp` PyTorch extension and depends on several C++ libraries:

- **Random123**
- **BLAS++**
- **LAPACK++**
- **RandBLAS**
- **RandLAPACK**
- **(Optional) CUDA** for GPU kernels

All libraries are automatically cloned into `deps/src/` and installed into `deps/install/`.  
Nothing is installed system-wide except basic compiler packages.

---

# Repository Layout

```text
torch_bqrrp/
│
├── Makefile
├── install.sh
├── INSTALL.md
├── README.md
├── setup.py
├── lisao.py
│
├── torch_bqrrp/
│   ├── bqrrp.py
│   └── bqrrp*.so
│
├── examples/
│   ├── bqrrp_demo.py
│   └── airbench94_lisao.py
│
└── deps/                 # auto-generated
    ├── src/             # cloned C++ deps
    └── install/         # built headers + libs
```

`deps/` is fully recreated by `install.sh`.

---

# 1. Recommended Installation (Makefile)

Most users should run:

```bash
make install
```

This:

1. Installs required system packages (OpenBLAS, LAPACK, LAPACKE, compilers)
2. Clones all C++ deps into `deps/src/`
3. Builds and installs them into `deps/install/`
4. Builds the PyTorch extension
5. Runs `examples/bqrrp_demo.py` as a sanity check

---

# 1.1 CPU-only Build

```bash
make install-cpu
```

This is equivalent to:

```bash
USE_CUDA=0 ./install.sh
```

CUDA is disabled; CPU-only kernels are compiled.

---

# 1.2 Running Examples

```bash
make test
```

Runs:

- `examples/bqrrp_demo.py`
- `examples/airbench94_lisao.py`

---

# 1.3 Cleaning

```bash
make clean-deps      # removes deps/src + deps/install
make clean-build     # removes build artifacts
make clean-all       # full reset
```

---

# 2. Python Virtual Environment (Recommended)

Ubuntu 24.04+ uses PEP 668 (“externally managed”), which blocks global installs.  
Use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install torch torchvision
```
Possible fix: 
```bash
cd /home/brosef/repos/torch_bqqrp
rm -rf .venv

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install "numpy==2.3.5"
# or a compatible torch + numpy combo:
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
```

Activate it in every new terminal:

```bash
source .venv/bin/activate
```

Now run:

```bash
make install
```

---

# 3. CUDA Behavior (`USE_CUDA`)

- Default: CUDA enabled (`USE_CUDA=1`)
- Requires `nvcc` in PATH
- BLAS++ uses cuBLAS
- RandLAPACK builds GPU kernels

Force CPU-only:

```bash
USE_CUDA=0 make install
```

---

# 4. Verifying Installation

## BQRRP Demo

```bash
python3 examples/bqrrp_demo.py
```

Runs a medium-size matrix through BQRRP.  
If successful, your install is correct.

## CIFAR-10 LISAO Example

```bash
python3 examples/airbench94_lisao.py
```

(Optional dataset dir:)

```bash
export CIFAR_DATA=./data
```

---

# 5. PyTorch Compatibility

### CUDA build:
```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
```

### CPU-only:
```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

Verify installation:

```bash
python3 - << 'EOF'
import torch, os
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF
```

---

# 6. Troubleshooting (Concise)

### Missing `nvcc`
```bash
which nvcc
```
If not found → install CUDA or use CPU-only mode.

### “externally managed environment”
Means you installed without a venv.  
Solution: create + activate a venv.

### Shared library load errors
```bash
export LD_LIBRARY_PATH=$PWD/deps/install/lib:$LD_LIBRARY_PATH
```

### Rebuild from scratch
```bash
make clean-all
make install
```

---

# 7. Direct `install.sh` Usage (Optional)

```bash
USE_CUDA=1 ./install.sh     # default
USE_CUDA=0 ./install.sh     # CPU-only
```

We recommend using the Makefile instead.

---

# 8. Reporting Issues

Include:

- OS version  
- CUDA version (if used)  
- Python + PyTorch versions  
- venv or system Python  
- Exact command you ran  
- Terminal logs

---

# End of INSTALL.md
