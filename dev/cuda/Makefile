# Makefile for CUDA Temperature + Top-NSigma Kernel
#
# Quick commands:
#   make build    - Build the PyTorch extension
#   make test     - Run tests
#   make bench    - Run benchmarks
#   make profile  - Profile with Nsight Compute
#   make clean    - Clean build artifacts

# Configuration
PYTHON := python3
TORCH_CUDA_ARCH_LIST ?= 8.0;8.6;8.9;9.0

# Phony targets
.PHONY: all build test bench profile clean help

all: build test

help:
	@echo "Available targets:"
	@echo "  build    - Build the PyTorch CUDA extension"
	@echo "  test     - Run correctness tests"
	@echo "  bench    - Run performance benchmarks"
	@echo "  profile  - Profile with NVIDIA Nsight Compute"
	@echo "  clean    - Remove build artifacts"
	@echo ""
	@echo "Environment variables:"
	@echo "  TORCH_CUDA_ARCH_LIST - CUDA architectures (default: 8.0;8.6;8.9;9.0)"
	@echo "  PYTHON - Python interpreter (default: python3)"

build:
	@echo "Building CUDA extension..."
	@echo "CUDA architectures: $(TORCH_CUDA_ARCH_LIST)"
	TORCH_CUDA_ARCH_LIST=$(TORCH_CUDA_ARCH_LIST) $(PYTHON) setup.py build_ext --inplace
	@echo "✅ Build complete!"

test: build
	@echo "Running tests..."
	$(PYTHON) test_cuda_kernel.py

bench: build
	@echo "Running benchmarks..."
	$(PYTHON) test_cuda_kernel.py

profile: build
	@echo "Profiling with NVIDIA Nsight Compute..."
	@echo "This will generate profile.ncu-rep"
	ncu --set full -o profile $(PYTHON) test_cuda_kernel.py
	@echo "View results with: ncu-ui profile.ncu-rep"

profile-nsys: build
	@echo "Profiling with NVIDIA Nsight Systems..."
	@echo "This will generate profile.nsys-rep"
	nsys profile -o profile $(PYTHON) test_cuda_kernel.py
	@echo "View results with: nsys-ui profile.nsys-rep"

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -f *.so
	rm -f *.pyd
	rm -f *.ncu-rep
	rm -f *.nsys-rep
	rm -rf __pycache__/
	@echo "✅ Clean complete!"

# Development helpers
format:
	@echo "Formatting code..."
	clang-format -i *.cu *.cpp *.h

check-cuda:
	@echo "Checking CUDA installation..."
	@nvcc --version || echo "❌ nvcc not found!"
	@nvidia-smi || echo "❌ nvidia-smi not found!"

info:
	@echo "System Information:"
	@echo "===================="
	@echo ""
	@echo "Python:"
	@$(PYTHON) --version
	@echo ""
	@echo "PyTorch:"
	@$(PYTHON) -c "import torch; print(f'Version: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'cuDNN: {torch.backends.cudnn.version()}')"
	@echo ""
	@echo "CUDA:"
	@nvcc --version
	@echo ""
	@echo "GPU:"
	@nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
