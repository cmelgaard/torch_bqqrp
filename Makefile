# Makefile for torch_bqrrp
#
# Common targets:
#   make install        # install with CUDA required (default)
#   make install-cpu    # install CPU-only (no CUDA)
#   make test           # run demo + CIFAR/LISAO example
#   make clean-deps     # remove entire deps/ tree
#   make clean-r123     # remove only Random123 bits
#   make clean-build    # remove Python/CMake build artifacts
#   make clean-all      # clean everything built/generated

# Default target
.PHONY: all
all: install

# ----------------------------------------------------------
# Install targets
# ----------------------------------------------------------

.PHONY: install
install:
	chmod +x install.sh
	./install.sh

.PHONY: install-cpu
install-cpu:
	chmod +x install.sh
	USE_CUDA=0 ./install.sh

# ----------------------------------------------------------
# Test targets
# ----------------------------------------------------------

.PHONY: test
test:
	python3 examples/bqrrp_demo.py
	python3 examples/airbench94_lisao.py

# ----------------------------------------------------------
# Clean targets
# ----------------------------------------------------------

# Nuke the entire deps tree (all sources + installed libs/headers)
.PHONY: clean-deps
clean-deps:
	rm -rf deps

# Just clean Random123 (optional lighter reset)
.PHONY: clean-r123
clean-r123:
	rm -rf deps/src/random123 deps/install/include/Random123
	@echo "Removed Random123 sources and installed headers."

# Clean Python/CMake/build artifacts, but keep deps/
.PHONY: clean-build
clean-build:
	rm -rf build dist
	rm -rf *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.so" -type f -delete
	find . -name "CMakeFiles" -type d -exec rm -rf {} +
	find . -name "CMakeCache.txt" -type f -delete
	find . -name "cmake_install.cmake" -type f -delete
	find . -name "Makefile" -type f -path "*/deps/src/*/build/*" -delete
	find . -name "*.ninja" -type f -delete
	find . -name ".ninja_log" -type f -delete
	find . -name ".ninja_deps" -type f -delete

# Clean absolutely everything generated
.PHONY: clean-all
clean-all: clean-deps clean-build
