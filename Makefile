PYTHON = python3

all: install

###########################################
# Build Dependencies
###########################################
deps:
	@bash install.sh

clean-deps:
	rm -rf deps

###########################################
# Build Python Extension
###########################################
build:
	$(PYTHON) setup.py build

install:
	@make deps
	$(PYTHON) -m pip install -e . --no-build-isolation

clean:
	# Remove build artifacts, but keep deps/ and all .so files
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +
	rm -f torch_bqrrp/_bqrrp*.so

clean-all: clean clean-deps
