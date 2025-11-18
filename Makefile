PYTHON = python3

all: install

###########################################
# Build Dependencies
###########################################
deps:
	@bash install.sh

clean-deps:
	rm -rf deps/src deps/install

###########################################
# Build Python Extension
###########################################
build:
	$(PYTHON) setup.py build

install:
	@make deps
	$(PYTHON) -m pip install -e .

clean:
	rm -rf build dist *.egg-info
	find . -name "*.so" -delete
