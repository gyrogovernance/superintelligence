# BabyLM Project Makefile

# Variables
PYTHON = python3
VENV = .venv
PIP = $(VENV)/bin/pip
ACTIVATE = source $(VENV)/bin/activate

# Use Homebrew LLVM for C/C++ compilation
LLVM_PREFIX=$(shell brew --prefix llvm)
export CC=$(LLVM_PREFIX)/bin/clang
export CXX=$(LLVM_PREFIX)/bin/clang++

.PHONY: help venv install system-deps setup clean run-wikipedia check-imports test gyrolabe-c gyrolabe-bench

help:
	@echo "Available targets:"
	@echo "  make venv         - Create a Python virtual environment"
	@echo "  make install      - Install Python dependencies into venv"
	@echo "  make system-deps  - Install system dependencies (Homebrew, llvm)"
	@echo "  make setup        - Full setup: system deps + venv + install"
	@echo "  make clean        - Remove __pycache__ and build artifacts"
	@echo "  make check-imports - Check that all imports use src.kernel (not router)"
	@echo "  make test         - Run all tests"
	@echo "  make run-wikipedia - Run the Wikipedia training script"
	@echo "  make gyrolabe-c   - Build GyroLabe C library (gyrolabe_codec.c + gyrolabe_mul.c)"
	@echo "  make gyrolabe-bench - Phase 4: benchmark C vs Python primitives"

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

system-deps:
	@echo "Installing system dependencies (requires Homebrew)..."
	@if ! command -v brew >/dev/null 2>&1; then \
	  echo 'Homebrew not found! Please install Homebrew first: https://brew.sh/'; \
	  exit 1; \
	fi
	brew install llvm

setup: system-deps install

clean:
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type f -name '*.log' -delete
	find . -type f -name '*.tmp' -delete

check-imports:
	@echo "Checking import namespace consistency..."
	$(PYTHON) scripts/check_imports.py

test:
	@echo "Running AIR CLI tests..."
	$(PYTHON) tests/test_aci_cli.py

test-pytest:
	@echo "Running all tests with pytest..."
	$(PYTHON) -m pytest tests/ -v

run-wikipedia:
	LLVM_PREFIX=$(shell brew --prefix llvm) && \
	export CC=$$LLVM_PREFIX/bin/clang && \
	export CXX=$$LLVM_PREFIX/bin/clang++ && \
	export NUMBA_OPT=3 && \
	export NUMBA_CACHE_DIR=$$HOME/.cache/numba && \
	$(ACTIVATE) && $(PYTHON) toys/training/wikipedia_eng.py --manual-download

gyrolabe-c:
	@echo "Building GyroLabe C library..."
	$(PYTHON) scripts/build_gyrolabe.py

gyrolabe-bench:
	@echo "Running GyroLabe Phase 4 benchmarks..."
	$(PYTHON) -m src.tools.gyrolabe.helpers.benchmark --report docs/reports/GyroLabe_Phase4_Benchmark_Report.md
