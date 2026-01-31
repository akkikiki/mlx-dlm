#!/bin/bash
# Compile the bandwidth-optimized CUDA kernel

set -e  # Exit on error

echo "=========================================="
echo "COMPILING BANDWIDTH-OPTIMIZED KERNEL"
echo "=========================================="
echo ""

# Check if nvcc is available
if ! command -v nvcc &> /dev/null; then
    echo "❌ Error: nvcc not found"
    echo "Please install CUDA Toolkit or add it to PATH"
    exit 1
fi

# Show CUDA version
echo "CUDA version:"
nvcc --version | grep "release"
echo ""

# Compilation flags
NVCC_FLAGS="-O3 --use_fast_math -Xcompiler -fPIC --shared"

# Compile the bandwidth-optimized kernel
echo "Compiling bandwidth_optimized_kernel.cu..."
nvcc $NVCC_FLAGS \
    -o bandwidth_optimized_kernel.so \
    bandwidth_optimized_kernel.cu

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo ""
    echo "Output: bandwidth_optimized_kernel.so"
    ls -lh bandwidth_optimized_kernel.so
else
    echo "❌ Compilation failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "READY TO BENCHMARK"
echo "=========================================="
echo ""
echo "Run the benchmark with:"
echo "  python3 benchmark_bandwidth.py"
echo ""
