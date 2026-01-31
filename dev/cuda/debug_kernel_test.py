#!/usr/bin/env python3
"""
Debug the CUDA kernel step by step to find the bug.
"""

import torch
import numpy as np
import subprocess
import sys

# ============================================================================
# Step 1: Compile Debug Kernel
# ============================================================================

def compile_debug_kernel():
    """Compile the debug kernel as a standalone .so file."""
    print("Compiling debug kernel...")

    cmd = [
        "nvcc",
        "-shared",
        "-Xcompiler", "-fPIC",
        "-o", "debug_kernel.so",
        "debug_kernel.cu",
        "-O3",
        "--use_fast_math",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("❌ Compilation failed!")
        print(result.stderr)
        return False

    print("✅ Debug kernel compiled")
    return True


# ============================================================================
# Step 2: Load and Test Debug Kernel
# ============================================================================

import ctypes

def load_debug_kernel():
    """Load the compiled debug kernel."""
    lib = ctypes.CDLL("./debug_kernel.so")

    # Define function signature
    lib.launch_temperature_topnsigma_debug.argtypes = [
        ctypes.c_void_p,  # input
        ctypes.c_void_p,  # output
        ctypes.c_void_p,  # debug_info
        ctypes.c_int,     # batch_size
        ctypes.c_int,     # seq_len
        ctypes.c_int,     # vocab_size
        ctypes.c_float,   # temperature
        ctypes.c_float,   # top_nsigma
        ctypes.c_void_p,  # stream
    ]

    return lib


def test_small_case():
    """Test with a small case to see intermediate values."""
    print("=" * 80)
    print("SMALL TEST CASE")
    print("=" * 80)
    print()

    # Very small test
    batch, seq, vocab = 2, 4, 100
    temperature = 1.5
    top_nsigma = 2.0

    torch.manual_seed(42)
    logits = torch.randn(batch, seq, vocab, device='cuda')

    print(f"Input shape: {logits.shape}")
    print(f"Temperature: {temperature}")
    print(f"Top-nsigma: {top_nsigma}")
    print()

    # PyTorch reference
    print("PyTorch computation:")
    logits_scaled = logits / temperature
    print(f"  First few values: {logits_scaled[0, 0, :5].tolist()}")

    maximum = logits_scaled.max(dim=-1, keepdim=True)[0]
    std = logits_scaled.std(dim=-1, keepdim=True)
    threshold = maximum - top_nsigma * std

    print(f"  Position [0,0]:")
    print(f"    Max: {maximum[0, 0, 0].item():.6f}")
    print(f"    Std: {std[0, 0, 0].item():.6f}")
    print(f"    Threshold: {threshold[0, 0, 0].item():.6f}")

    filtered = torch.where(
        logits_scaled >= threshold,
        logits_scaled,
        torch.tensor(float('-inf'), device='cuda')
    )

    inf_count = torch.isinf(filtered[0, 0]).sum().item()
    print(f"    Infs in position [0,0]: {inf_count}")
    print(f"    Value at [0,0,35]: {filtered[0, 0, 35].item():.6f}")
    print()

    # CUDA debug kernel
    print("CUDA kernel computation:")
    print("-" * 40)

    lib = load_debug_kernel()

    output = torch.empty_like(logits)
    debug_info = torch.zeros(batch * seq, 4, device='cuda')

    # Call kernel
    lib.launch_temperature_topnsigma_debug(
        ctypes.c_void_p(logits.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(debug_info.data_ptr()),
        batch, seq, vocab,
        ctypes.c_float(temperature),
        ctypes.c_float(top_nsigma),
        ctypes.c_void_p(0)  # default stream
    )

    torch.cuda.synchronize()

    print("-" * 40)
    print()

    # Compare
    print("Comparison:")
    cuda_inf_count = torch.isinf(output[0, 0]).sum().item()
    print(f"  PyTorch infs: {inf_count}")
    print(f"  CUDA infs: {cuda_inf_count}")
    print(f"  Match: {inf_count == cuda_inf_count}")
    print()

    # Debug info from kernel
    print("Debug info from kernel (position 0):")
    print(f"  Max: {debug_info[0, 0].item():.6f}")
    print(f"  Mean: {debug_info[0, 1].item():.6f}")
    print(f"  Std: {debug_info[0, 2].item():.6f}")
    print(f"  Threshold: {debug_info[0, 3].item():.6f}")
    print()

    # Check specific value
    print(f"Value at [0,0,35]:")
    print(f"  PyTorch: {filtered[0, 0, 35].item():.6f}")
    print(f"  CUDA: {output[0, 0, 35].item():.6f}")
    print()


def compare_basic_vs_optimized():
    """Compare basic kernel from extension vs debug kernel."""
    print("=" * 80)
    print("COMPARING EXTENSION KERNELS")
    print("=" * 80)
    print()

    try:
        import temperature_topnsigma_cuda
    except ImportError:
        print("❌ Extension not built. Run: python setup.py build_ext --inplace")
        return

    batch, seq, vocab = 2, 4, 100
    temperature = 1.5
    top_nsigma = 2.0

    torch.manual_seed(42)
    logits = torch.randn(batch, seq, vocab, device='cuda')

    # PyTorch
    logits_scaled = logits / temperature
    maximum = logits_scaled.max(dim=-1, keepdim=True)[0]
    std = logits_scaled.std(dim=-1, keepdim=True)
    threshold = maximum - top_nsigma * std
    pytorch_result = torch.where(
        logits_scaled >= threshold,
        logits_scaled,
        torch.tensor(float('-inf'), device='cuda')
    )

    # Extension basic
    basic_result = temperature_topnsigma_cuda.temperature_topnsigma(
        logits.clone(), temperature, top_nsigma
    )

    # Extension optimized
    opt_result = temperature_topnsigma_cuda.temperature_topnsigma_optimized(
        logits.clone(), temperature, top_nsigma
    )

    # Debug kernel
    lib = load_debug_kernel()
    debug_result = torch.empty_like(logits)
    debug_info = torch.zeros(batch * seq, 4, device='cuda')

    lib.launch_temperature_topnsigma_debug(
        ctypes.c_void_p(logits.data_ptr()),
        ctypes.c_void_p(debug_result.data_ptr()),
        ctypes.c_void_p(debug_info.data_ptr()),
        batch, seq, vocab,
        ctypes.c_float(temperature),
        ctypes.c_float(top_nsigma),
        ctypes.c_void_p(0)
    )
    torch.cuda.synchronize()

    # Compare inf counts
    print("Inf counts for position [0, 0]:")
    print(f"  PyTorch:   {torch.isinf(pytorch_result[0, 0]).sum().item()}")
    print(f"  Basic:     {torch.isinf(basic_result[0, 0]).sum().item()}")
    print(f"  Optimized: {torch.isinf(opt_result[0, 0]).sum().item()}")
    print(f"  Debug:     {torch.isinf(debug_result[0, 0]).sum().item()}")
    print()

    # Check if results match
    print("Results match PyTorch:")
    print(f"  Basic:     {torch.allclose(basic_result, pytorch_result, equal_nan=True)}")
    print(f"  Optimized: {torch.allclose(opt_result, pytorch_result, equal_nan=True)}")
    print(f"  Debug:     {torch.allclose(debug_result, pytorch_result, equal_nan=True)}")
    print()


def main():
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  CUDA KERNEL DEBUGGING".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return

    # Compile debug kernel
    if not compile_debug_kernel():
        return

    # Run tests
    test_small_case()
    compare_basic_vs_optimized()

    print()
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("Look at the CUDA kernel output above.")
    print("Compare the intermediate values with PyTorch.")
    print("The bug should be visible in the printed values!")
    print()


if __name__ == "__main__":
    main()
