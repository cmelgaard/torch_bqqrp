# Torch-BQRRP — GPU Build & Install Guide (CUDA Required)

This repository builds a PyTorch extension that depends on:
- Random123  
- BLAS++  
- LAPACK++  
- RandBLAS  
- RandLAPACK  
- CUDA (mandatory)  

Everything is built locally inside `deps/`.  
A single command performs a full install:

```
make install
```

---

# 1. Requirements

- Linux (Ubuntu recommended)
- Python 3.10–3.12
- PyTorch ≥ 2.5.0 (built with CUDA 12.1, recommended)
- CUDA toolkit 12.0 or 12.1 installed
- NVIDIA driver supporting CUDA ≥ 12.2
- A GPU with compute capability ≥ **5.2** (GTX TITAN-X is OK)

---

# 2. Quick Start

```
git clone <your repo>
cd torch_bqrrp
make install
```

This will:
1. Create `deps/`
2. Compile all required dependencies
3. Build the CUDA extension
4. Install your package in editable mode

---

# 3. Repository Layout

```
torch_bqrrp/
│
├── csrc/                  # C++/CUDA source
├── torch_bqrrp/           # Python package
│   ├── __init__.py
│   └── bqrrp.py
│
├── deps/
│   ├── src/               # Full source clones
│   └── install/           # Installed headers and libs
│
├── install.sh             # Dependency installer
├── setup.py               # PyTorch CUDA extension builder
├── Makefile               # Defines make install/deps/clean
└── install.md
```

---

# 4. Installing Python Environment

```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

# 5. Running the Full Build

```
make install
```

---

# 6. Cleaning Everything

```
make clean
make clean-deps
```

`clean-deps` removes the entire `deps/` tree.

---

# 7. Dependency Notes

### CUDA  
PyTorch you installed is built with `CUDA 12.1`.  
Your toolkit is CUDA 12.0 — this is *fine* because we use:
- `CUDAHOSTCXX=g++-12`
- `-Xcompiler=-mno-avx512fp16`
- explicit arch `(compute_52)` for TITAN-X

---

# 8. Rebuilding Only the Extension

```
make build
```

---

# 9. Testing Import

```
python3 -c "import torch_bqrrp; print('BQRRP OK')"
```

---

# 10. Troubleshooting

### Q: NVCC errors mentioning AVX512 FP16?  
A: Your CPU supports AVX2 only. We disable AVX512 FP16 via:
```
-Xcompiler=-mno-avx512fp16
```

### Q: Link errors “cannot find -lRandBLAS or -lRandLAPACK”?  
A: Your deps weren’t built:  
Run:
```
make clean-deps
make install
```

### Q: PTX or “unsupported gpu architecture”?  
A: TITAN-X = compute_52.  
Your extension sets:
```
-gencode=arch=compute_52,code=sm_52
```

### Q: PyTorch complains about ABI?  
A: You must keep:
```
-D_GLIBCXX_USE_CXX11_ABI=0
```

---

# 11. Summary

- **Single command install:** `make install`  
- Fully self-contained dependencies in `deps/`  
- CUDA-compatible extension build  
- Works with PyTorch 2.5.1 + CUDA 12.1  
- Tested on compute capability **5.2**  
