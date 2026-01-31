# Build Troubleshooting Guide

Common issues when building the CUDA extension and how to fix them.

## Issue 1: "could not find ninja"

```
UserWarning: Attempted to use ninja as the BuildExtension backend but we could not find ninja.
Falling back to using the slow distutils backend.
```

**Impact**: Build is slower but still works
**Solution** (optional - speeds up builds):

```bash
# Ubuntu/Debian
sudo apt-get install ninja-build

# macOS
brew install ninja

# pip
pip install ninja
```

## Issue 2: "getCurrentCUDAStream is not a member of at::cuda"

```
error: 'getCurrentCUDAStream' is not a member of 'at::cuda'
```

**Cause**: API changed in PyTorch 2.x
**Status**: ✅ FIXED in the updated code

**Fix applied**:
```cpp
// Old (PyTorch 1.x)
cudaStream_t stream = at::cuda::getCurrentCUDAStream();

// New (PyTorch 1.x and 2.x compatible)
cudaStream_t stream = c10::cuda::getCurrentCUDAStream(logits.device().index());
```

## Issue 3: CUDA version mismatch

```
UserWarning: There are no x86_64-linux-gnu-g++ version bounds defined for CUDA version 12.4
```

**Impact**: Warning only, usually safe to ignore
**Meaning**: PyTorch was compiled with different CUDA version than you have

**Check versions**:
```bash
# Your CUDA version
nvcc --version

# PyTorch's CUDA version
python -c "import torch; print(torch.version.cuda)"
```

**If they differ significantly** (e.g., 11.8 vs 12.4):
- Usually works fine (CUDA is backward compatible)
- If you get runtime errors, reinstall PyTorch matching your CUDA:

```bash
# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Issue 4: "nvcc not found"

```
error: nvcc not found
```

**Cause**: CUDA toolkit not in PATH

**Fix**:
```bash
# Find CUDA
ls /usr/local/cuda*/bin/nvcc

# Add to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Reload
source ~/.bashrc  # or source ~/.zshrc
```

## Issue 5: Compiler version incompatible

```
error: unsupported GNU version! gcc versions later than X are not supported!
```

**Cause**: GCC version too new for your CUDA version

**CUDA compatibility**:
- CUDA 11.x: GCC <= 10
- CUDA 12.x: GCC <= 12

**Fix**:
```bash
# Install older GCC
sudo apt-get install gcc-10 g++-10

# Use specific version
export CC=gcc-10
export CXX=g++-10

# Then build
python setup.py build_ext --inplace
```

## Issue 6: "unsupported GPU architecture"

```
error: unsupported gpu architecture 'compute_XX'
```

**Cause**: TORCH_CUDA_ARCH_LIST doesn't match your GPU

**Fix**:
```bash
# Check your GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Example output: 8.6 (for RTX 3090)

# Set correct architecture
export TORCH_CUDA_ARCH_LIST="8.6"

# For multiple GPUs
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"

# Build
python setup.py build_ext --inplace
```

**Common compute capabilities**:
- RTX 40xx (4090, etc.): 8.9
- RTX 30xx (3090, 3080, etc.): 8.6
- A100: 8.0
- V100: 7.0
- RTX 20xx (2080 Ti, etc.): 7.5

## Issue 7: Out of memory during compilation

```
c++: fatal error: Killed signal terminated program cc1plus
```

**Cause**: Not enough RAM for parallel compilation

**Fix**:
```bash
# Limit parallel jobs
MAX_JOBS=1 python setup.py build_ext --inplace

# Or set permanently
export MAX_JOBS=2
python setup.py build_ext --inplace
```

## Issue 8: torch/extension.h not found

```
fatal error: torch/extension.h: No such file or directory
```

**Cause**: PyTorch not properly installed

**Fix**:
```bash
# Reinstall PyTorch
pip uninstall torch
pip install torch

# Or install from source
pip install torch --no-binary torch
```

## Issue 9: Build works but import fails

```python
>>> import temperature_topnsigma_cuda
ImportError: cannot import name 'temperature_topnsigma_cuda'
```

**Cause**: Module not in Python path or wrong Python version

**Fix**:
```bash
# Check build output location
ls -la *.so

# Should see: temperature_topnsigma_cuda.cpython-311-x86_64-linux-gnu.so

# Make sure you're in the same directory
cd dev/cuda
python -c "import temperature_topnsigma_cuda; print('Success!')"
```

## Issue 10: Runtime CUDA errors

```
CUDA kernel failed: invalid configuration argument
```

**Possible causes**:
1. GPU out of memory
2. Invalid tensor dimensions
3. Tensor not on GPU

**Debug**:
```python
import torch
import temperature_topnsigma_cuda

# Check tensor
logits = torch.randn(8, 32, 156896, device='cuda')
print(f"Device: {logits.device}")  # Should be cuda:0
print(f"Contiguous: {logits.is_contiguous()}")  # Should be True
print(f"Dtype: {logits.dtype}")  # Should be torch.float32

# Try
try:
    result = temperature_topnsigma_cuda.temperature_topnsigma(
        logits, 1.5, 2.0
    )
    print("Success!")
except RuntimeError as e:
    print(f"Error: {e}")
```

## Quick Build Commands

```bash
# Clean build
rm -rf build/ *.so
python setup.py build_ext --inplace

# With specific CUDA architecture
TORCH_CUDA_ARCH_LIST="8.6" python setup.py build_ext --inplace

# Debug build (with symbols)
DEBUG=1 python setup.py build_ext --inplace

# Using Makefile
make clean
make build
```

## Verify Installation

```bash
# Run tests
python test_cuda_kernel.py

# Quick check
python -c "
import torch
import temperature_topnsigma_cuda

x = torch.randn(2, 4, 1000, device='cuda')
y = temperature_topnsigma_cuda.temperature_topnsigma(x, 1.5, 2.0)
print('✅ CUDA extension working!')
"
```

## Still Having Issues?

1. **Check system info**:
   ```bash
   make info
   # Or manually:
   python --version
   python -c "import torch; print(torch.__version__, torch.version.cuda)"
   nvcc --version
   nvidia-smi
   ```

2. **Clean everything**:
   ```bash
   make clean
   # Or manually:
   rm -rf build/ dist/ *.egg-info/ *.so __pycache__/
   ```

3. **Try minimal test**:
   ```bash
   # Test PyTorch CUDA
   python -c "import torch; print(torch.cuda.is_available())"

   # Test basic compilation
   echo '#include <cuda_runtime.h>
   __global__ void test() {}
   int main() { return 0; }' > test.cu
   nvcc test.cu -o test
   ./test
   ```

4. **Check PyTorch installation**:
   ```bash
   python -c "
   import torch
   print(f'PyTorch: {torch.__version__}')
   print(f'CUDA available: {torch.cuda.is_available()}')
   print(f'CUDA version: {torch.version.cuda}')
   print(f'cuDNN version: {torch.backends.cudnn.version()}')
   print(f'GPU: {torch.cuda.get_device_name(0)}')
   "
   ```

## Getting Help

If none of these solutions work:

1. Collect diagnostic info:
   ```bash
   make info > diagnostic.txt
   python test_cuda_kernel.py 2>&1 >> diagnostic.txt
   ```

2. Check PyTorch forums: https://discuss.pytorch.org/

3. Include in your question:
   - Full error message
   - PyTorch version
   - CUDA version
   - GPU model
   - OS and distribution
