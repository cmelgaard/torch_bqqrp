# Installation Guide

This project depends on several C++ numerical libraries:

- **Random123**  
- **BLAS++** and **LAPACK++**  
- **RandBLAS**  
- **RandLAPACK**  
- **(Optional) CUDA** for GPU-accelerated `BQRRP_GPU`

To keep things **self-contained and reproducible**, all of these are:

- cloned into: `deps/src/`
- built and installed into: `deps/install/`

Everything stays inside this repo.  
If you want to reset the build, just delete `deps/` (or the entire repo) and reinstall.

---

## Repository Layout

After installing, the repo looks roughly like:

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
│   ├── bqrrp_demo.py            # BQRRP demo on a 5000×2800 matrix
│   └── airbench94_lisao.py      # CIFAR-10 + LISAO optimizer example
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
        ├── lib/
        └── cmake/
```

The `deps/` directory is generated; it is **not tracked by git**.

---

# 1. Recommended Installation: Using the Makefile

The **supported and recommended** way to install is via the `Makefile`.  
You should almost never need to call `install.sh` directly.

From the repo root:

```bash
make install
```

This will:

1. Install required system packages via `apt` (compiler, cmake, OpenBLAS, etc.).
2. Clone all C++ dependencies into `deps/src/`.
3. Build and install them into `deps/install/`.
4. Build the PyTorch extension (`python3 setup.py develop`).
5. Run the BQRRP demo (`examples/bqrrp_demo.py`) to verify it works.

By default, `make install` enables **CUDA mode**, and **CUDA is required**.  
If CUDA is not installed or `nvcc` is not in your `PATH`, it will fail with a clear error.

---

## 1.1 CPU-only install

If you want a **CPU-only** build (no CUDA at all), run:

```bash
make install-cpu
```

This is equivalent to:

```bash
USE_CUDA=0 ./install.sh
```

and will:

- Build BLAS++ without cuBLAS
- Build RandLAPACK with `RequireCUDA=OFF`
- Produce a CPU-only BQRRP extension

Use this on machines without NVIDIA GPUs or where CUDA is not available.

---

## 1.2 Running tests

After installation finishes, you can run:

```bash
make test
```

This runs:

- `python3 examples/bqrrp_demo.py`
- `python3 examples/airbench94_lisao.py`

to verify the low-level BQRRP binding and the LISAO optimizer on CIFAR-10.

---

## 1.3 Cleaning

To remove just the dependency tree:

```bash
make clean-deps
```

This deletes `deps/` (all cloned sources and built libs), but leaves your Python code and examples.

To clean build artifacts (CMake/Python/build outputs, compiled extensions, etc.):

```bash
make clean-build
```

To nuke **everything generated** (both `deps/` and build artifacts):

```bash
make clean-all
```

---

# 2. Optional but Recommended: Python Virtual Environment

On modern Linux systems (e.g., Ubuntu 24.04), system Python is “externally managed” and does not like global `pip` usage.  
The safest way to work is inside a **virtual environment**.

From the repo root:

```bash
cd /path/to/torch_bqrrp

# Create a venv (only needed once)
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Inside the venv, install Python deps you need:
pip install --upgrade pip setuptools wheel
pip install torch torchvision  # or your preferred versions
```

Then:

```bash
make install
```

or for CPU-only:

```bash
make install-cpu
```

The compiled extension will be built against the Python and PyTorch in **this venv**, not system Python.

If you ever start a new shell, don’t forget to re-activate the venv:

```bash
cd /path/to/torch_bqrrp
source .venv/bin/activate
```

before running `make`, `python`, etc.

---

# 3. CUDA Behavior (`USE_CUDA`)

The CUDA behavior is controlled by `USE_CUDA`, but you normally never set it directly — the **Makefile does it for you**.

- `make install`  
  → runs `USE_CUDA=1 ./install.sh` internally  
  → **CUDA required**

- `make install-cpu`  
  → runs `USE_CUDA=0 ./install.sh` internally  
  → **CPU-only build**

Internally:

- `USE_CUDA=1` (default for `make install`):
  - `nvcc` must be in `PATH`
  - BLAS++ is built with cuBLAS
  - RandLAPACK is built with `RequireCUDA=ON`
- `USE_CUDA=0` (`make install-cpu`):
  - CUDA is not used at all
  - BLAS++ and RandLAPACK are built CPU-only

---

# 4. Verifying the Installation

After `make install` or `make install-cpu` completes successfully:

## 4.1 BQRRP demo

```bash
python3 examples/bqrrp_demo.py
```

This:

- Builds a 5000×2800 matrix
- Runs BQRRP (and BQRRP_GPU if built with CUDA)
- Prints pivot info / shapes

If this runs without error, the core extension is working.

---

## 4.2 CIFAR-10 + LISAO

```bash
python3 examples/airbench94_lisao.py
```

Optional: specify CIFAR data directory:

```bash
export CIFAR_DATA=./data
```

This:

- Downloads CIFAR-10 (if not present)
- Builds a small CNN
- Trains using SGD and LISAO
- Prints accuracy and timing info

---

# 5. Advanced: Using `install.sh` Directly (Not Recommended)

The primary supported interface is:

- `make install`
- `make install-cpu`

`install.sh` is a **lower-level helper** and should only be used if you know what you’re doing or are debugging install issues.

From the repo root:

```bash
chmod +x install.sh

# CUDA required
./install.sh

# CPU-only
USE_CUDA=0 ./install.sh
```

The script:

1. Installs system dev packages via `apt`.
2. Clones all dependencies into `deps/src/`.
3. Builds and installs them under `deps/install/`.
4. Exports environment variables for the compiler/linker.
5. Runs `python3 setup.py develop`.
6. Executes `python3 examples/bqrrp_demo.py`.

Again, for normal users: **use the Makefile**.

---

# 6. Troubleshooting

### 6.1 “CUDA was requested but 'nvcc' is not in PATH”

If you run `make install` and see an error about CUDA:

- Make sure CUDA toolkit is installed
- Make sure `nvcc` is visible:

  ```bash
  which nvcc
  nvcc --version
  ```

- Or explicitly choose CPU mode:

  ```bash
  make install-cpu
  ```

---

### 6.2 Externally-managed Python environment (Ubuntu 24.04)

If you see `error: externally-managed-environment` related to `pip`, it means you’re trying to modify system Python.  
Fix: create and use a **venv** (see Section 2 above).

---

### 6.3 Shared library load errors

If Python can’t find C++ shared libraries at runtime, try:

```bash
export LD_LIBRARY_PATH=$PWD/deps/install/lib:$LD_LIBRARY_PATH
```

and re-run your command.

---

### 6.4 Nuking and starting over

If the build gets into a weird state:

```bash
make clean-all
make install        # or make install-cpu
```

---

# 7. Getting Help

If you’re stuck, please include:

- OS (e.g., Ubuntu 24.04)
- CUDA version (if applicable)
- Python + PyTorch versions
- Whether you used a venv
- Exact command you ran (`make install`, `make install-cpu`, etc.)
- Full install log (copy-paste from your terminal)

when opening an issue.

---
