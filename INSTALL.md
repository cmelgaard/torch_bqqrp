# Installation Guide

This project depends on several C++ libraries:

- Random123  
- BLAS++ and LAPACK++  
- RandBLAS  
- RandLAPACK  
- Optional CUDA (for GPU BQRRP)  

To make installation simple and self-contained, everything is downloaded, built, and installed into:

```
./deps/
    src/      (cloned sources)
    install/  (headers + libs)
```

Nothing outside this repo is touched (except for apt packages).

If you ever want to completely reset the system, just delete the entire repo.

---

# 1. Using `install.sh` (Recommended)

From the root of this repo:

```bash
chmod +x install.sh
./install.sh
```

This script will:

1. Install system packages (compiler, cmake, python headers, OpenBLAS, etc.)
2. Clone all required C++ libraries into `deps/src/`
3. Build and install them into `deps/install/`
4. Build the PyTorch extension (`python setup.py develop`)
5. Run a BQRRP smoke test (`examples/bqrrp_demo.py`)

Everything is local to the repo.

---

# 2. CUDA Support (`USE_CUDA` flag)

The project supports two installation modes:

### **Default: CUDA REQUIRED**

If you simply run:

```bash
./install.sh
```

Then:

- `USE_CUDA` defaults to `1`
- CUDA **must** be installed (`nvcc` must be in PATH)
- GPU version of BQRRP (`BQRRP_GPU`) will be compiled
- If CUDA is missing, the script exits with an **error**  

No silent CPU fallback happens.

---

### **CPU-only Build (explicit opt-out)**

To disable CUDA entirely:

```bash
USE_CUDA=0 ./install.sh
```

This will:

- Build BLAS++ without cuBLAS
- Build RandLAPACK without CUDA
- Build a CPU-only PyTorch extension

Use this when building on machines without NVIDIA GPUs.

---

# 3. Checking Your CUDA Version

If you plan on using CUDA:

```bash
nvcc --version
```

or:

```bash
nvidia-smi | grep "CUDA Version"
```

or:

```bash
cat /usr/local/cuda/version.txt
```

CUDA **12.2** works fine with this project.

---

# 4. Verifying the Installation

### **4.1 BQRRP Demo**

```bash
python3 examples/bqrrp_demo.py
```

This constructs a 5000×2800 matrix and runs BQRRP using either CPU or GPU depending on your build.

If this script succeeds, the core extension works.

---

### **4.2 CIFAR-10 / LISAO Training Example**

```bash
python3 examples/airbench94_lisao.py
```

Optional:

```bash
export CIFAR_DATA=./data
```

This:

- Downloads CIFAR-10  
- Builds a simple CNN  
- Trains using SGD and LISAO  
- Prints accuracy + timing

---

# 5. Manual Installation (Optional)

Normally you do **not** need this, but here is what `install.sh` does:

1. Install system packages with `apt`
2. Clone:
   - Random123  
   - blaspp  
   - lapackpp  
   - RandBLAS  
   - RandLAPACK (`--recursive`)  
3. Configure them with:
   - `-DCMAKE_INSTALL_PREFIX=./deps/install`
   - CUDA flags depending on `USE_CUDA`
4. Export:
   ```bash
   export CMAKE_PREFIX_PATH=./deps/install:$CMAKE_PREFIX_PATH
   export LD_LIBRARY_PATH=./deps/install/lib:$LD_LIBRARY_PATH
   export LIBRARY_PATH=./deps/install/lib:$LIBRARY_PATH
   export CPATH=./deps/install/include:$CPATH
   ```
5. Run:
   ```bash
   python3 setup.py develop
   ```

---

# 6. Troubleshooting

### **CUDA requested but nvcc missing**

If you see:

```
ERROR: CUDA was requested but 'nvcc' not found
```

Either:

- Install CUDA  
or  
- Use CPU-only mode:

```bash
USE_CUDA=0 ./install.sh
```

---

### **Missing BLAS/LAPACK at build time**

Check:

```
./deps/install/lib/cmake/blaspp
./deps/install/lib/cmake/lapackpp
```

If missing, delete `deps/` and rerun install.sh.

---

### **PyTorch extension cannot find libs at runtime**

Export these (install.sh already prints them):

```bash
export LD_LIBRARY_PATH=./deps/install/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=./deps/install/lib:$LIBRARY_PATH
export CPATH=./deps/install/include:$CPATH
```

---

# 7. Getting Help

If something breaks, open an issue and include:

- OS version  
- CUDA version (if any)  
- Python version  
- Full output of `install.sh`  

Issues are usually very quick to diagnose.

---
