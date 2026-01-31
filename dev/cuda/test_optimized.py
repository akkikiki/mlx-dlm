#!/usr/bin/env python3
"""
Test and benchmark the optimized fused kernel
"""

import torch
import time
import numpy as np

# Import all versions
try:
    import temperature_topnsigma_cuda
    ORIGINAL_AVAILABLE = True
except ImportError:
    ORIGINAL_AVAILABLE = False
    print("⚠️  Original kernel not available")

try:
    import temperature_topnsigma_optimized as opt
    OPTIMIZED_AVAILABLE = True
except ImportError:
    OPTIMIZED_AVAILABLE = False
    print("⚠️  Optimized kernel not available")
    print("   Build with: python setup_optimized.py build_ext --inplace")


def test_correctness():
    """Verify optimized kernel produces correct results."""
    print("=" * 80)
    print("CORRECTNESS TEST")
    print("=" * 80)
    print()

    if not OPTIMIZED_AVAILABLE:
        print("❌ Optimized kernel not available")
        return

    batch, seq, vocab = 4, 8, 1000
    temperature = 1.5
    top_nsigma = 2.0

    torch.manual_seed(42)
    logits = torch.randn(batch, seq, vocab, device='cuda')

    # PyTorch reference
    scaled = logits / temperature
    maximum = scaled.max(dim=-1, keepdim=True)[0]
    std = scaled.std(dim=-1, keepdim=True)
    threshold = maximum - top_nsigma * std
    ref = torch.where(scaled >= threshold, scaled, torch.tensor(float('-inf'), device='cuda'))

    # Optimized kernel (basic)
    result_fused = opt.fused(logits.clone(), temperature, top_nsigma)

    # Optimized kernel (warp)
    result_fused_warp = opt.fused_warp(logits.clone(), temperature, top_nsigma)

    # Compare
    def check_match(name, result):
        ref_infs = torch.isinf(ref).sum().item()
        result_infs = torch.isinf(result).sum().item()
        diff = abs(ref_infs - result_infs)

        finite_mask = torch.isfinite(ref) & torch.isfinite(result)
        if finite_mask.any():
            max_diff = torch.abs(ref - result)[finite_mask].max().item()
        else:
            max_diff = 0.0

        status = "✅" if diff <= 2 and max_diff < 1e-4 else "❌"
        print(f"{status} {name:25s}: infs diff={diff:>4d}, max_diff={max_diff:.2e}")

    print(f"Reference (PyTorch): {torch.isinf(ref).sum().item()} infs")
    check_match("Optimized (fused)", result_fused)
    check_match("Optimized (fused_warp)", result_fused_warp)

    print()


def benchmark_all():
    """Benchmark all versions."""
    print("=" * 80)
    print("PERFORMANCE BENCHMARK")
    print("=" * 80)
    print()

    batch, seq, vocab = 8, 32, 156896
    temperature = 1.5  # Use non-1.0 for fair comparison
    top_nsigma = 2.0

    torch.manual_seed(42)
    logits = torch.randn(batch, seq, vocab, device='cuda')

    results = []

    # torch.compile
    @torch.compile
    def compiled_version(logits, temp, nsigma):
        scaled = logits / temp
        maximum = scaled.max(dim=-1, keepdim=True)[0]
        std = scaled.std(dim=-1, keepdim=True)
        threshold = maximum - nsigma * std
        return torch.where(scaled >= threshold, scaled, torch.tensor(float('-inf'), device='cuda'))

    # Warmup
    for _ in range(5):
        _ = compiled_version(logits.clone(), temperature, top_nsigma)
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(50):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = compiled_version(logits.clone(), temperature, top_nsigma)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    results.append(("torch.compile", np.mean(times), np.std(times)))

    # Original CUDA kernel
    if ORIGINAL_AVAILABLE:
        for _ in range(5):
            _ = temperature_topnsigma_cuda.temperature_topnsigma(logits.clone(), temperature, top_nsigma)
            torch.cuda.synchronize()

        times = []
        for _ in range(50):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = temperature_topnsigma_cuda.temperature_topnsigma(logits.clone(), temperature, top_nsigma)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

        results.append(("CUDA (original)", np.mean(times), np.std(times)))

    # Optimized fused kernel
    if OPTIMIZED_AVAILABLE:
        for _ in range(5):
            _ = opt.fused(logits.clone(), temperature, top_nsigma)
            torch.cuda.synchronize()

        times = []
        for _ in range(50):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = opt.fused(logits.clone(), temperature, top_nsigma)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

        results.append(("CUDA (fused)", np.mean(times), np.std(times)))

        # Warp-optimized version
        for _ in range(5):
            _ = opt.fused_warp(logits.clone(), temperature, top_nsigma)
            torch.cuda.synchronize()

        times = []
        for _ in range(50):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = opt.fused_warp(logits.clone(), temperature, top_nsigma)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

        results.append(("CUDA (fused+warp)", np.mean(times), np.std(times)))

    # Print results
    print(f"{'Implementation':<25s} {'Time':>12s} {'Speedup':>10s}")
    print("-" * 80)

    baseline = results[0][1]  # torch.compile
    for name, mean_time, std_time in sorted(results, key=lambda x: x[1]):
        speedup = baseline / mean_time
        print(f"{name:<25s} {mean_time:>8.2f}ms ± {std_time:>4.2f}ms   {speedup:>6.2f}x")

    print()

    # Memory bandwidth analysis
    total_elements = batch * seq * vocab
    bytes_per_element = 4

    print("Memory Bandwidth Analysis:")
    print("-" * 80)

    for name, mean_time, _ in results:
        if "fused" in name.lower():
            # Optimized: 2 reads + 1 write = 3 ops
            memory_ops = 3
        elif "original" in name.lower():
            # Original: 5 reads + 2 writes = 7 ops
            memory_ops = 7
        else:
            # torch.compile: estimate ~4 ops
            memory_ops = 4

        total_bytes = total_elements * bytes_per_element * memory_ops
        bandwidth = total_bytes / (mean_time / 1000) / 1e9

        print(f"{name:<25s}: {bandwidth:>6.1f} GB/s ({memory_ops} memory ops)")

    print()
    print(f"RTX 4000 Ada peak bandwidth: 560 GB/s")
    print()


def main():
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  OPTIMIZED FUSED KERNEL BENCHMARK".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return

    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    test_correctness()
    benchmark_all()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Optimizations applied:")
    print("  ✓ Fused temperature scaling with statistics computation")
    print("  ✓ Computed mean and variance using E[X²] - E[X]²")
    print("  ✓ Eliminated intermediate write of scaled values")
    print("  ✓ Reduced memory operations: 7 → 3 (2 reads + 1 write)")
    print()
    print("Expected performance:")
    print("  • Should match or beat torch.compile!")
    print("  • ~2.3x speedup over original CUDA kernel")
    print()


if __name__ == "__main__":
    main()
