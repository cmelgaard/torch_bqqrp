# Installation Guide

This project depends on several C++ numerical libraries:

- **Random123**  
- **BLAS++**  
- **LAPACK++**  
- **RandBLAS**  
- **RandLAPACK**  
- **(Optional) CUDA** for GPU-accelerated `BQRRP_GPU`

To maintain **reproducibility**, **isolation**, and **zero system contamination**, all dependencies are:

- cloned into: `deps/src/`
- built and installed into: `deps/install/`

Everything stays inside this repository.  
To reset everything: **delete the repo** or just delete `deps/` and rerun `install.sh`.

---

# Repository Layout (Developer-Friendly)

Below is the complete layout *after* running `install.sh`, including the new `Makefile`:

```text
torch_bqrrp/
│
├── .gitignore
├── Makefile
├── install.sh
├── INSTALL.md
├── README.md
├── setup.py
├── lisao.py
│
├── torch_bqrrp/
│   ├── __init__.py
│   ├── bqrrp.py
│   └── bqrrp*.so                # compiled PyTorch extension (generated)
│
├── examples/
│   ├── bqrrp_demo.py
│   └── airbench94_lisao.py
│
└── deps/                        # CREATED BY install.sh (ignored by git)
    ├── src/                     # dependency sources
    │   ├── random123/
    │   ├── blaspp/
    │   ├── lapackpp/
    │   ├── RandBLAS/
    │   └── RandLAPACK/
    │
    └── install/                 # installed headers + libraries
        ├── include/
        │   ├── Random123/
        │   ├── blaspp/
        │   ├── lapackpp/
        │   ├── RandBLAS/
        │   └── RandLAPACK/
        │
        ├── lib/
        │   ├── libblaspp.so
        │   ├── liblapackpp.so
        │   ├── libRandBLAS.so
        │   ├── libRandLAPACK.so
        │   └── (optional CUDA libs)
        │
        └── cmake/
            ├── blaspp/
            ├── lapackpp/
            ├── RandBLAS/
            └── RandLAPACK/
```

The entire dependency tree lives in `deps/` and is **ignored** by git.  
This structure is commonly used in scientific/HPC projects.

---

# 1. Installing with `install.sh` (Recommended)

From the root of the repo:

```bash
chmod +x install.sh
./install.sh
```

This performs:

1. Installs system packages via `apt`
2. Clones all libraries into `deps/src/`
3. Builds and installs them into `deps/install/`
4. Configures environment variables for compilers + linkers
5. Builds the PyTorch extension (`python3 setup.py develop`)
6. Runs the BQRRP demo to verify the install

Everything stays in `deps/`, and **nothing is installed globally**.

---

# 2. CUDA Support (`USE_CUDA` Flag)

This project supports two modes:

| Mode | Description |
|------|-------------|
| **Default (USE_CUDA=1)** | CUDA is **required**, and installer **fails** if CUDA is missing |
| **CPU-only (USE_CUDA=0)** | Build everything without CUDA |

### Default behavior (CUDA required)

```bash
./install.sh
```

- `USE_CUDA` defaults to `1`
- If `nvcc` is missing → **installation fails**
- If CUDA exists:
  - BLAS++ is built with cuBLAS
  - RandLAPACK is built with GPU support
  - PyTorch extension exposes GPU BQRRP

### CPU-only install

```bash
USE_CUDA=0 ./install.sh
```

This builds:

- BLAS++ without cuBLAS
- RandLAPACK CPU-only
- CPU-only BQRRP extension

Use this mode on systems without NVIDIA GPUs.

---

# 3. Checking CUDA Version

```bash
nvcc --version
```

or

```bash
nvidia-smi | grep "CUDA Version"
```

or

```bash
cat /usr/local/cuda/version.txt
```

CUDA **12.2** is fully compatible with this project.

---

# 4. Verifying Installation

## 4.1 BQRRP demo

```bash
python3 examples/bqrrp_demo.py
```

This:

- Builds a 5000×2800 matrix  
- Runs BQRRP (CPU or GPU)  
- Prints pivot info  

If it runs without error, installation succeeded.

---

## 4.2 CIFAR-10 + LISAO Example

```bash
python3 examples/airbench94_lisao.py
```

Optionally:

```bash
export CIFAR_DATA=./data
```

This:

- Downloads CIFAR-10  
- Builds a simple CNN  
- Trains using SGD + LISAO  
- Prints accuracy and timing  

---

# 5. Makefile Usage (Optional but Recommended)

This repo includes a **Makefile** providing convenient shortcuts.

## 5.1 Install (CUDA required)

```bash
make install
```

Equivalent to:

```bash
./install.sh
```

## 5.2 CPU-only install

```bash
make install-cpu
```

Equivalent to:

```bash
USE_CUDA=0 ./install.sh
```

## 5.3 Run both examples

```bash
make test
```

## 5.4 Clean dependency tree (remove deps/)

```bash
make clean-deps
```

## 5.5 Clean all build artifacts

```bash
make clean-build
```

## 5.6 Clean everything

```bash
make clean-all
```

---

# 6. Manual Installation (Advanced Only)

If you truly want to replicate the script manually:

1. `apt install` required dev packages  
2. Clone all dependencies into `deps/src/`  
3. Use CMake to install each into `deps/install/`  
4. Export environment variables:
   ```bash
   export CMAKE_PREFIX_PATH=$PWD/deps/install:$CMAKE_PREFIX_PATH
   export LD_LIBRARY_PATH=$PWD/deps/install/lib:$LD_LIBRARY_PATH
   export LIBRARY_PATH=$PWD/deps/install/lib:$LIBRARY_PATH
   export CPATH=$PWD/deps/install/include:$CPATH
   ```
5. Build extension:
   ```bash
   python3 setup.py develop
   ```

But **99% of users should never do this** — use `install.sh`.

---

# 7. Troubleshooting

## CUDA requested but missing

```
ERROR: CUDA was requested (USE_CUDA != 0) but 'nvcc' is not in PATH.
```

Fix:

- Install CUDA properly, **or**
- Use CPU-only mode:

```bash
USE_CUDA=0 ./install.sh
```

---

## Missing BLAS++ or LAPACK++

Delete `deps/` and reinstall:

```bash
rm -rf deps
./install.sh
```

---

## Shared library load errors

```bash
export LD_LIBRARY_PATH=$PWD/deps/install/lib:$LD_LIBRARY_PATH
```

---

# 8. Getting Help

When opening an issue, include:

- OS version  
- CUDA version (if any)  
- Python + PyTorch versions  
- Full output of:

```bash
./install.sh
```

---

# END OF INSTALL.md
