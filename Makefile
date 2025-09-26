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

.PHONY: help venv install system-deps setup clean run-wikipedia

help:
	@echo "Available targets:"
	@echo "  make venv         - Create a Python virtual environment"
	@echo "  make install      - Install Python dependencies into venv"
	@echo "  make system-deps  - Install system dependencies (Homebrew, llvm)"
	@echo "  make setup        - Full setup: system deps + venv + install"
	@echo "  make clean        - Remove __pycache__ and build artifacts"
	@echo "  make run-wikipedia - Run the Wikipedia training script"

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

run-wikipedia:
	LLVM_PREFIX=$(shell brew --prefix llvm) && \
	export CC=$$LLVM_PREFIX/bin/clang && \
	export CXX=$$LLVM_PREFIX/bin/clang++ && \
	export NUMBA_OPT=3 && \
	export NUMBA_CACHE_DIR=$$HOME/.cache/numba && \
	$(ACTIVATE) && $(PYTHON) toys/training/wikipedia_eng.py --manual-download
