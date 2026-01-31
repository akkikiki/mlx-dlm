#!/usr/bin/env python3
"""
Investigate why the CUDA kernel is slower than torch.compile.
"""

import torch
import time
import numpy as np

try:
    import temperature_topnsigma_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("❌ CUDA extension not available")
    exit(1)


def benchmark_fair_comparison():
    """Fair comparison with temperature != 1.0."""
    print("=" * 80)
    print("FAIR COMPARISON (temperature != 1.0)")
    print("=" * 80)
    print()

    batch, seq, vocab = 8, 32, 156896

    # Test with different temperatures
    temperatures = [1.0, 1.5, 2.0]
    top_nsigma = 2.0

    for temp in temperatures:
        print(f"Temperature: {temp}")
        print("-" * 40)

        torch.manual_seed(42)
        logits = torch.randn(batch, seq, vocab, device='cuda')

        # Eager PyTorch
        times = []
        for _ in range(20):
            torch.cuda.synchronize()
            start = time.perf_counter()

            scaled = logits / temp
            maximum = scaled.max(dim=-1, keepdim=True)[0]
            std = scaled.std(dim=-1, keepdim=True)
            threshold = maximum - top_nsigma * std
            result = torch.where(scaled >= threshold, scaled, torch.tensor(float('-inf'), device='cuda'))

            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

        eager_time = np.mean(times)
        print(f"  Eager PyTorch: {eager_time:.2f}ms")

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
            _ = compiled_version(logits.clone(), temp, top_nsigma)
            torch.cuda.synchronize()

        times = []
        for _ in range(20):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = compiled_version(logits.clone(), temp, top_nsigma)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

        compiled_time = np.mean(times)
        print(f"  torch.compile:  {compiled_time:.2f}ms")

        # CUDA kernel
        times = []
        for _ in range(20):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = temperature_topnsigma_cuda.temperature_topnsigma(logits.clone(), temp, top_nsigma)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

        cuda_time = np.mean(times)
        print(f"  CUDA kernel:    {cuda_time:.2f}ms")

        print(f"  Speedup (compile vs CUDA): {cuda_time/compiled_time:.2f}x")
        print()


def measure_kernel_launch_overhead():
    """Measure if kernel launch overhead is significant."""
    print("=" * 80)
    print("KERNEL LAUNCH OVERHEAD")
    print("=" * 80)
    print()

    # Very small input - kernel should be fast, overhead visible
    batch, seq, vocab = 1, 1, 1000
    temperature = 1.5
    top_nsigma = 2.0

    logits = torch.randn(batch, seq, vocab, device='cuda')

    # Warmup
    for _ in range(10):
        _ = temperature_topnsigma_cuda.temperature_topnsigma(logits, temperature, top_nsigma)
        torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(100):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        result = temperature_topnsigma_cuda.temperature_topnsigma(logits, temperature, top_nsigma)
        end_event.record()
        torch.cuda.synchronize()

        times.append(start_event.elapsed_time(end_event))

    print(f"Small input (1, 1, 1000):")
    print(f"  Mean: {np.mean(times):.3f}ms")
    print(f"  Min:  {np.min(times):.3f}ms")
    print()

    # Compare to simple PyTorch operation
    times_pytorch = []
    for _ in range(100):
        start_event.record()
        result = logits / temperature  # Simple division
        end_event.record()
        torch.cuda.synchronize()
        times_pytorch.append(start_event.elapsed_time(end_event))

    print(f"PyTorch simple division:")
    print(f"  Mean: {np.mean(times_pytorch):.3f}ms")
    print()

    overhead = np.mean(times) - np.mean(times_pytorch)
    print(f"Estimated overhead: {overhead:.3f}ms")
    print()


def analyze_memory_access():
    """Analyze memory access pattern."""
    print("=" * 80)
    print("MEMORY ACCESS ANALYSIS")
    print("=" * 80)
    print()

    batch, seq, vocab = 8, 32, 156896
    num_positions = batch * seq
    num_elements = num_positions * vocab

    print(f"Configuration:")
    print(f"  Positions: {num_positions}")
    print(f"  Vocab: {vocab:,}")
    print(f"  Total elements: {num_elements:,}")
    print()

    bytes_per_element = 4  # float32

    print("Memory operations in CUDA kernel:")
    print("  1. Read input (temperature scaling):     1x read")
    print("  2. Write output (temperature scaling):   1x write")
    print("  3. Read output (max reduction):          1x read")
    print("  4. Read output (sum reduction):          1x read")
    print("  5. Read output (variance reduction):     1x read")
    print("  6. Read output (threshold filter):       1x read")
    print("  7. Write output (threshold filter):      1x write")
    print()

    total_reads = 5  # Steps 1, 3, 4, 5, 6
    total_writes = 2  # Steps 2, 7

    total_bytes = num_elements * bytes_per_element * (total_reads + total_writes)
    print(f"Total memory traffic: {total_bytes / 1e9:.2f} GB")
    print()

    # Measured performance
    cuda_time = 4.63  # ms from earlier benchmark
    compile_time = 2.61  # ms from earlier benchmark

    cuda_bandwidth = total_bytes / (cuda_time / 1000) / 1e9
    compile_bandwidth = total_bytes / (compile_time / 1000) / 1e9

    print(f"CUDA kernel:")
    print(f"  Time: {cuda_time:.2f}ms")
    print(f"  Bandwidth: {cuda_bandwidth:.1f} GB/s")
    print()

    print(f"torch.compile:")
    print(f"  Time: {compile_time:.2f}ms")
    print(f"  Bandwidth: {compile_bandwidth:.1f} GB/s")
    print()

    print(f"torch.compile is {cuda_bandwidth / compile_bandwidth:.2f}x more memory efficient")
    print("This suggests torch.compile is doing fewer memory operations!")
    print()


def profile_with_pytorch_profiler():
    """Use PyTorch profiler to see what's happening."""
    print("=" * 80)
    print("PYTORCH PROFILER")
    print("=" * 80)
    print()

    batch, seq, vocab = 8, 32, 156896
    temperature = 1.5
    top_nsigma = 2.0

    torch.manual_seed(42)
    logits = torch.randn(batch, seq, vocab, device='cuda')

    print("Profiling CUDA kernel...")

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(10):
            _ = temperature_topnsigma_cuda.temperature_topnsigma(logits.clone(), temperature, top_nsigma)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print()


def check_gpu_utilization():
    """Check if we're saturating the GPU."""
    print("=" * 80)
    print("GPU UTILIZATION CHECK")
    print("=" * 80)
    print()

    batch, seq, vocab = 8, 32, 156896
    num_blocks = batch * seq

    print(f"Kernel launch configuration:")
    print(f"  Grid size: {num_blocks} blocks")
    print(f"  Block size: 256 threads")
    print(f"  Total threads: {num_blocks * 256} = {num_blocks * 256:,}")
    print()

    # RTX 4000 Ada specs
    print(f"RTX 4000 Ada Generation specs:")
    print(f"  CUDA cores: 6,144")
    print(f"  SMs: 48")
    print(f"  Max threads per SM: 1,536")
    print(f"  Max blocks per SM: 16-32 (varies)")
    print()

    threads_launched = num_blocks * 256
    max_concurrent_threads = 48 * 1536  # 48 SMs * 1536 threads/SM

    utilization = min(100, (threads_launched / max_concurrent_threads) * 100)

    print(f"Thread utilization:")
    print(f"  Launched: {threads_launched:,} threads")
    print(f"  GPU can handle: {max_concurrent_threads:,} concurrent threads")
    print(f"  Utilization: {utilization:.1f}%")
    print()

    if utilization < 50:
        print("⚠️  LOW GPU UTILIZATION!")
        print("   The GPU is underutilized. Consider:")
        print("   - Processing multiple positions per block")
        print("   - Using larger batch sizes")
        print()
    else:
        print("✅ GPU utilization looks reasonable")
        print()


def main():
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  WHY IS THE CUDA KERNEL SLOW?".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    benchmark_fair_comparison()
    measure_kernel_launch_overhead()
    analyze_memory_access()
    check_gpu_utilization()
    # profile_with_pytorch_profiler()  # Commented out - can be slow

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Possible reasons for slow performance:")
    print()
    print("1. Memory access pattern - CUDA does 7 passes, torch.compile may fuse better")
    print("2. GPU utilization - Only 256 blocks might not saturate RTX 4000 Ada")
    print("3. Kernel fusion - torch.compile fuses operations more aggressively")
    print("4. Memory layout - torch.compile may use better memory organization")
    print()
    print("To improve CUDA kernel:")
    print("  • Fuse more operations (reduce passes over memory)")
    print("  • Process multiple positions per block")
    print("  • Use Triton instead (easier optimization)")
    print()


if __name__ == "__main__":
    main()
