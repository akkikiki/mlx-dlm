#!/usr/bin/env python3
"""
Test ONLY the basic kernel to see if it's actually fast when working correctly.
"""

import torch
import time
import numpy as np

try:
    import temperature_topnsigma_cuda
    CUDA_AVAILABLE = True
except ImportError:
    print("❌ Extension not built")
    CUDA_AVAILABLE = False
    exit(1)


def benchmark_basic_kernel():
    """Benchmark just the basic kernel."""
    print("=" * 80)
    print("BASIC KERNEL PERFORMANCE TEST")
    print("=" * 80)
    print()

    batch_size = 8
    seq_len = 32
    vocab_size = 156896
    temperature = 1.0
    top_nsigma = 2.0

    torch.manual_seed(42)
    logits = torch.randn(batch_size, seq_len, vocab_size, device='cuda')

    print(f"Configuration:")
    print(f"  Shape: ({batch_size}, {seq_len}, {vocab_size})")
    print(f"  Temperature: {temperature}")
    print(f"  Top-nsigma: {top_nsigma}")
    print()

    # Warmup
    print("Warming up...")
    for _ in range(10):
        _ = temperature_topnsigma_cuda.temperature_topnsigma(
            logits.clone(), temperature, top_nsigma
        )
        torch.cuda.synchronize()
    print()

    # Benchmark
    print("Benchmarking basic kernel (100 iterations)...")
    times = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(100):
        start_event.record()
        result = temperature_topnsigma_cuda.temperature_topnsigma(
            logits.clone(), temperature, top_nsigma
        )
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    times = np.array(times)

    print(f"  Mean: {np.mean(times):.2f}ms ± {np.std(times):.2f}ms")
    print(f"  Min: {np.min(times):.2f}ms")
    print(f"  Median: {np.median(times):.2f}ms")
    print()

    # Check correctness
    print("Checking correctness...")
    ref_logits = logits / temperature
    ref_max = ref_logits.max(dim=-1, keepdim=True)[0]
    ref_std = ref_logits.std(dim=-1, keepdim=True)
    ref_thresh = ref_max - top_nsigma * ref_std
    ref_result = torch.where(
        ref_logits >= ref_thresh,
        ref_logits,
        torch.tensor(float('-inf'), device='cuda')
    )

    result = temperature_topnsigma_cuda.temperature_topnsigma(
        logits.clone(), temperature, top_nsigma
    )

    inf_match = (torch.isinf(ref_result) == torch.isinf(result)).all().item()
    finite_mask = torch.isfinite(ref_result) & torch.isfinite(result)

    if finite_mask.any():
        diff = torch.abs(ref_result - result)
        diff = torch.where(finite_mask, diff, torch.tensor(0.0, device='cuda'))
        max_diff = diff.max().item()
    else:
        max_diff = 0.0

    print(f"  Inf positions match: {inf_match}")
    print(f"  Max difference (finite): {max_diff:.2e}")

    if inf_match and max_diff < 1e-4:
        print("  ✅ Basic kernel is CORRECT!")
    else:
        print("  ❌ Basic kernel has issues")

    print()

    # Bandwidth
    total_elements = batch_size * seq_len * vocab_size
    bytes_total = total_elements * 4 * 5  # Rough estimate
    bandwidth = bytes_total / (np.mean(times) / 1000) / 1e9

    print(f"Estimated bandwidth: {bandwidth:.1f} GB/s")
    print(f"Peak bandwidth (RTX 4000 Ada): 560 GB/s")
    print(f"Efficiency: {bandwidth / 560 * 100:.1f}%")
    print()


def profile_with_nsight():
    """Instructions for profiling with Nsight."""
    print("=" * 80)
    print("PROFILING WITH NSIGHT COMPUTE")
    print("=" * 80)
    print()
    print("To see why the kernel is slow, profile it:")
    print()
    print("  ncu --set full --target-processes all \\")
    print("      --kernel-name temperature_topnsigma_kernel \\")
    print("      python test_basic_only.py")
    print()
    print("This will show:")
    print("  - Memory throughput")
    print("  - Compute utilization")
    print("  - Warp stalls")
    print("  - Occupancy")
    print()


def main():
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  BASIC KERNEL ONLY - CORRECTNESS & PERFORMANCE".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return

    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    benchmark_basic_kernel()
    profile_with_nsight()


if __name__ == "__main__":
    main()
