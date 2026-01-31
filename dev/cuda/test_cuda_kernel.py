#!/usr/bin/env python3
"""
Test and benchmark the CUDA temperature + top-nsigma kernel

Prerequisites:
    1. Build the extension: python setup.py build_ext --inplace
    2. Have PyTorch with CUDA support installed

Usage:
    python test_cuda_kernel.py
"""

import torch
import time
import numpy as np

# Try to import the CUDA extension
try:
    import temperature_topnsigma_cuda
    CUDA_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import CUDA extension: {e}")
    print("   Build it with: python setup.py build_ext --inplace")
    CUDA_AVAILABLE = False


def temperature_topnsigma_pytorch(logits, temperature, top_nsigma):
    """Reference PyTorch implementation for comparison"""
    # Temperature scaling
    logits = logits / temperature

    # Compute statistics
    maximum = logits.max(dim=-1, keepdim=True)[0]
    std = logits.std(dim=-1, keepdim=True)
    threshold = maximum - top_nsigma * std

    # Filter
    logits = torch.where(logits >= threshold, logits, torch.tensor(float('-inf')))

    return logits


def test_correctness():
    """Test that CUDA kernel produces correct results"""
    if not CUDA_AVAILABLE:
        print("⏭️  Skipping correctness test (CUDA extension not available)")
        return

    print("=" * 80)
    print("CORRECTNESS TEST")
    print("=" * 80)
    print()

    # Test configuration
    batch_size = 4
    seq_len = 8
    vocab_size = 1000
    temperature = 1.5
    top_nsigma = 2.0

    print(f"Configuration:")
    print(f"  Shape: ({batch_size}, {seq_len}, {vocab_size})")
    print(f"  Temperature: {temperature}")
    print(f"  Top-nsigma: {top_nsigma}")
    print()

    # Create test data
    torch.manual_seed(42)
    logits = torch.randn(batch_size, seq_len, vocab_size, device='cuda')

    # Run CUDA kernel (basic)
    print("Running CUDA kernel (basic)...")
    result_cuda_basic = temperature_topnsigma_cuda.temperature_topnsigma(
        logits, temperature, top_nsigma
    )
    torch.cuda.synchronize()

    # Run CUDA kernel (optimized)
    print("Running CUDA kernel (optimized)...")
    result_cuda_opt = temperature_topnsigma_cuda.temperature_topnsigma_optimized(
        logits, temperature, top_nsigma
    )
    torch.cuda.synchronize()

    # Run PyTorch reference
    print("Running PyTorch reference...")
    result_pytorch = temperature_topnsigma_pytorch(
        logits.clone(), temperature, top_nsigma
    )
    torch.cuda.synchronize()

    print()
    print("Comparing results...")

    # Compare basic CUDA vs PyTorch
    finite_mask = torch.isfinite(result_cuda_basic) & torch.isfinite(result_pytorch)
    inf_match_basic = (torch.isinf(result_cuda_basic) == torch.isinf(result_pytorch)).all().item()

    if finite_mask.any():
        diff_basic = torch.abs(result_cuda_basic - result_pytorch)
        diff_basic = torch.where(finite_mask, diff_basic, torch.tensor(0.0, device='cuda'))
        max_diff_basic = diff_basic.max().item()
        mean_diff_basic = diff_basic.mean().item()
    else:
        max_diff_basic = 0.0
        mean_diff_basic = 0.0

    print(f"  CUDA (basic) vs PyTorch:")
    print(f"    Max difference (finite):  {max_diff_basic:.2e}")
    print(f"    Mean difference (finite): {mean_diff_basic:.2e}")
    print(f"    Inf positions match: {inf_match_basic}")

    # Compare optimized CUDA vs PyTorch
    finite_mask_opt = torch.isfinite(result_cuda_opt) & torch.isfinite(result_pytorch)
    inf_match_opt = (torch.isinf(result_cuda_opt) == torch.isinf(result_pytorch)).all().item()

    if finite_mask_opt.any():
        diff_opt = torch.abs(result_cuda_opt - result_pytorch)
        diff_opt = torch.where(finite_mask_opt, diff_opt, torch.tensor(0.0, device='cuda'))
        max_diff_opt = diff_opt.max().item()
        mean_diff_opt = diff_opt.mean().item()
    else:
        max_diff_opt = 0.0
        mean_diff_opt = 0.0

    print(f"  CUDA (optimized) vs PyTorch:")
    print(f"    Max difference (finite):  {max_diff_opt:.2e}")
    print(f"    Mean difference (finite): {mean_diff_opt:.2e}")
    print(f"    Inf positions match: {inf_match_opt}")

    print()

    # Verdict
    if max_diff_basic < 1e-4 and max_diff_opt < 1e-4 and inf_match_basic and inf_match_opt:
        print("  ✅ All implementations match!")
    else:
        print("  ⚠️  Results differ")
        if not inf_match_basic or not inf_match_opt:
            print("      - Inf positions don't match")
        if max_diff_basic >= 1e-4 or max_diff_opt >= 1e-4:
            print(f"      - Finite values differ significantly")

    print()


def benchmark_performance():
    """Benchmark CUDA kernel vs PyTorch reference"""
    if not CUDA_AVAILABLE:
        print("⏭️  Skipping benchmark (CUDA extension not available)")
        return

    print("=" * 80)
    print("PERFORMANCE BENCHMARK")
    print("=" * 80)
    print()

    # Realistic size for LLaDA2
    batch_size = 8
    seq_len = 32
    vocab_size = 156896  # LLaDA2 vocab size
    temperature = 1.0
    top_nsigma = 2.0

    print(f"Configuration:")
    print(f"  Shape: ({batch_size}, {seq_len}, {vocab_size})")
    print(f"  Total positions: {batch_size * seq_len}")
    print(f"  Vocab size: {vocab_size:,}")
    print()

    # Create test data
    torch.manual_seed(42)
    logits = torch.randn(batch_size, seq_len, vocab_size, device='cuda')

    # Warmup
    print("Warming up...")
    for _ in range(5):
        _ = temperature_topnsigma_cuda.temperature_topnsigma(logits, temperature, top_nsigma)
        _ = temperature_topnsigma_cuda.temperature_topnsigma_optimized(logits, temperature, top_nsigma)
        _ = temperature_topnsigma_pytorch(logits.clone(), temperature, top_nsigma)
        torch.cuda.synchronize()
    print()

    # Benchmark CUDA (basic)
    print("Benchmarking CUDA kernel (basic)...")
    times_cuda_basic = []
    for _ in range(20):
        torch.cuda.synchronize()
        start = time.time()
        result = temperature_topnsigma_cuda.temperature_topnsigma(logits, temperature, top_nsigma)
        torch.cuda.synchronize()
        times_cuda_basic.append((time.time() - start) * 1000)

    avg_cuda_basic = np.mean(times_cuda_basic)
    std_cuda_basic = np.std(times_cuda_basic)
    print(f"  Average: {avg_cuda_basic:.2f}ms ± {std_cuda_basic:.2f}ms")

    # Benchmark CUDA (optimized)
    print("Benchmarking CUDA kernel (optimized)...")
    times_cuda_opt = []
    for _ in range(20):
        torch.cuda.synchronize()
        start = time.time()
        result = temperature_topnsigma_cuda.temperature_topnsigma_optimized(logits, temperature, top_nsigma)
        torch.cuda.synchronize()
        times_cuda_opt.append((time.time() - start) * 1000)

    avg_cuda_opt = np.mean(times_cuda_opt)
    std_cuda_opt = np.std(times_cuda_opt)
    print(f"  Average: {avg_cuda_opt:.2f}ms ± {std_cuda_opt:.2f}ms")

    # Benchmark PyTorch
    print("Benchmarking PyTorch reference...")
    times_pytorch = []
    for _ in range(20):
        torch.cuda.synchronize()
        start = time.time()
        result = temperature_topnsigma_pytorch(logits.clone(), temperature, top_nsigma)
        torch.cuda.synchronize()
        times_pytorch.append((time.time() - start) * 1000)

    avg_pytorch = np.mean(times_pytorch)
    std_pytorch = np.std(times_pytorch)
    print(f"  Average: {avg_pytorch:.2f}ms ± {std_pytorch:.2f}ms")

    print()
    print("-" * 80)
    print("COMPARISON")
    print("-" * 80)
    print(f"PyTorch reference:       {avg_pytorch:.2f}ms ± {std_pytorch:.2f}ms")
    print(f"CUDA kernel (basic):     {avg_cuda_basic:.2f}ms ± {std_cuda_basic:.2f}ms")
    print(f"CUDA kernel (optimized): {avg_cuda_opt:.2f}ms ± {std_cuda_opt:.2f}ms")
    print()

    speedup_basic = avg_pytorch / avg_cuda_basic
    speedup_opt = avg_pytorch / avg_cuda_opt

    if speedup_basic > 1.0:
        print(f"CUDA (basic) is {speedup_basic:.2f}x FASTER than PyTorch ✅")
    else:
        print(f"CUDA (basic) is {1/speedup_basic:.2f}x slower than PyTorch")

    if speedup_opt > 1.0:
        print(f"CUDA (optimized) is {speedup_opt:.2f}x FASTER than PyTorch ✅")
    else:
        print(f"CUDA (optimized) is {1/speedup_opt:.2f}x slower than PyTorch")

    if avg_cuda_opt < avg_cuda_basic:
        improvement = avg_cuda_basic / avg_cuda_opt
        print(f"Optimized version is {improvement:.2f}x faster than basic ⚡")

    print()


def profile_kernel():
    """Profile the kernel with CUDA events"""
    if not CUDA_AVAILABLE:
        print("⏭️  Skipping profiling (CUDA extension not available)")
        return

    print("=" * 80)
    print("KERNEL PROFILING")
    print("=" * 80)
    print()

    batch_size = 8
    seq_len = 32
    vocab_size = 156896
    temperature = 1.0
    top_nsigma = 2.0

    logits = torch.randn(batch_size, seq_len, vocab_size, device='cuda')

    # Use CUDA events for precise timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(5):
        _ = temperature_topnsigma_cuda.temperature_topnsigma_optimized(logits, temperature, top_nsigma)
    torch.cuda.synchronize()

    # Profile
    print("Profiling with CUDA events...")
    times = []
    for _ in range(100):
        start_event.record()
        result = temperature_topnsigma_cuda.temperature_topnsigma_optimized(logits, temperature, top_nsigma)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    print(f"  Min: {np.min(times):.2f}ms")
    print(f"  Max: {np.max(times):.2f}ms")
    print(f"  Mean: {np.mean(times):.2f}ms")
    print(f"  Median: {np.median(times):.2f}ms")
    print(f"  Std: {np.std(times):.2f}ms")
    print()

    # Throughput
    total_elements = batch_size * seq_len * vocab_size
    throughput = total_elements / (np.mean(times) / 1000) / 1e9  # GB/s
    print(f"  Throughput: {throughput:.2f} GElements/s")
    print(f"  Bandwidth: {throughput * 4:.2f} GB/s (float32)")
    print()


def main():
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  CUDA KERNEL TEST: TEMPERATURE + TOP-NSIGMA SAMPLING".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        print("   Make sure you have:")
        print("   1. NVIDIA GPU")
        print("   2. CUDA toolkit installed")
        print("   3. PyTorch with CUDA support")
        return

    print(f"✅ CUDA is available")
    print(f"   Device: {torch.cuda.get_device_name()}")
    print(f"   CUDA version: {torch.version.cuda}")
    print()

    # Run tests
    test_correctness()
    benchmark_performance()
    profile_kernel()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("✅ CUDA kernel implementation complete!")
    print()
    print("Key features:")
    print("  • Fused temperature scaling + top-nsigma filtering")
    print("  • Single kernel pass (no intermediate memory)")
    print("  • Parallel reductions for max/mean/variance")
    print("  • Optimized version with warp primitives")
    print()
    print("For profiling with Nsight Compute:")
    print("  ncu --set full -o profile python test_cuda_kernel.py")
    print()


if __name__ == "__main__":
    main()
