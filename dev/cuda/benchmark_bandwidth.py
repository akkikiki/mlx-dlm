#!/usr/bin/env python3
"""
Benchmark bandwidth optimizations for custom CUDA kernel
"""

import torch
import time
import subprocess
import os

# Compile the bandwidth-optimized kernel
print("=" * 80)
print("COMPILING BANDWIDTH-OPTIMIZED KERNEL")
print("=" * 80)
print()

compile_cmd = [
    "nvcc",
    "-O3",
    "--use_fast_math",
    "-Xcompiler", "-fPIC",
    "--shared",
    "-o", "bandwidth_optimized_kernel.so",
    "bandwidth_optimized_kernel.cu"
]

try:
    result = subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
    print("✅ Compilation successful")
except subprocess.CalledProcessError as e:
    print(f"❌ Compilation failed:")
    print(e.stderr)
    exit(1)

print()

# Load the compiled kernel
import ctypes
lib = ctypes.CDLL("./bandwidth_optimized_kernel.so")

# Define function signatures
lib.launch_temperature_topnsigma_bandwidth_optimized_bs256.argtypes = [
    ctypes.c_void_p,  # input
    ctypes.c_void_p,  # output
    ctypes.c_int,     # batch_size
    ctypes.c_int,     # seq_len
    ctypes.c_int,     # vocab_size
    ctypes.c_float,   # temperature
    ctypes.c_float,   # top_nsigma
    ctypes.c_void_p   # stream
]

lib.launch_temperature_topnsigma_bandwidth_optimized_bs512.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_void_p
]

lib.launch_temperature_topnsigma_bandwidth_optimized_bs1024.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_void_p
]

# Test configuration
batch_size = 8
seq_len = 32
vocab_size = 156896
temperature = 1.5
top_nsigma = 2.0
num_warmup = 10
num_iter = 100

print("=" * 80)
print("BENCHMARK CONFIGURATION")
print("=" * 80)
print(f"Batch size: {batch_size}")
print(f"Sequence length: {seq_len}")
print(f"Vocabulary size: {vocab_size}")
print(f"Temperature: {temperature}")
print(f"Top-nsigma: {top_nsigma}")
print(f"Warmup iterations: {num_warmup}")
print(f"Benchmark iterations: {num_iter}")
print()

# Create test data
logits = torch.randn(batch_size, seq_len, vocab_size, device='cuda', dtype=torch.float32)
output = torch.empty_like(logits)

# Calculate data transfer sizes
input_size_mb = logits.numel() * 4 / (1024 * 1024)
output_size_mb = output.numel() * 4 / (1024 * 1024)
total_data_mb = 2 * input_size_mb + output_size_mb  # 2 reads + 1 write

print(f"Input size: {input_size_mb:.2f} MB")
print(f"Output size: {output_size_mb:.2f} MB")
print(f"Total data transfer: {total_data_mb:.2f} MB (2 reads + 1 write)")
print()

# Get stream
stream = torch.cuda.current_stream().cuda_stream

def benchmark_kernel(launch_func, block_size_name):
    """Benchmark a kernel configuration"""
    # Warmup
    for _ in range(num_warmup):
        launch_func(
            logits.data_ptr(),
            output.data_ptr(),
            batch_size,
            seq_len,
            vocab_size,
            temperature,
            top_nsigma,
            stream
        )
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iter):
        launch_func(
            logits.data_ptr(),
            output.data_ptr(),
            batch_size,
            seq_len,
            vocab_size,
            temperature,
            top_nsigma,
            stream
        )
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time_ms = (end - start) / num_iter * 1000
    bandwidth_gb_s = total_data_mb / 1024 / (avg_time_ms / 1000)

    return avg_time_ms, bandwidth_gb_s

# Benchmark different block sizes
print("=" * 80)
print("BANDWIDTH BENCHMARK RESULTS")
print("=" * 80)
print()

configs = [
    (lib.launch_temperature_topnsigma_bandwidth_optimized_bs256, "BS=256"),
    (lib.launch_temperature_topnsigma_bandwidth_optimized_bs512, "BS=512"),
    (lib.launch_temperature_topnsigma_bandwidth_optimized_bs1024, "BS=1024"),
]

results = []
for launch_func, name in configs:
    print(f"Testing {name}...")
    try:
        time_ms, bandwidth = benchmark_kernel(launch_func, name)
        results.append((name, time_ms, bandwidth))
        print(f"  Time: {time_ms:.3f} ms")
        print(f"  Bandwidth: {bandwidth:.1f} GB/s")
        print()
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        print()

# Find best configuration
if results:
    best = max(results, key=lambda x: x[2])  # Max bandwidth
    print("=" * 80)
    print("BEST CONFIGURATION")
    print("=" * 80)
    print(f"Configuration: {best[0]}")
    print(f"Time: {best[1]:.3f} ms")
    print(f"Bandwidth: {best[2]:.1f} GB/s")
    print()

# Compare with baseline (if available)
print("=" * 80)
print("COMPARISON WITH BASELINES")
print("=" * 80)
print()

baselines = {
    "torch.compile": (2.62, 307),
    "Custom CUDA (original)": (2.63, 183),
}

print(f"{'Method':<30} {'Time (ms)':<12} {'Bandwidth (GB/s)':<18} {'Speedup':<10}")
print("-" * 80)

for method, (time, bw) in baselines.items():
    print(f"{method:<30} {time:<12.3f} {bw:<18.1f} {'1.00x':<10}")

if results:
    for name, time, bw in results:
        speedup = baselines["Custom CUDA (original)"][0] / time
        print(f"Optimized {name:<21} {time:<12.3f} {bw:<18.1f} {speedup:<10.2f}x")

print()
print("=" * 80)
print("ANALYSIS")
print("=" * 80)
print()

if results:
    best_time, best_bw = best[1], best[2]
    original_bw = baselines["Custom CUDA (original)"][1]
    torch_bw = baselines["torch.compile"][1]

    bw_improvement = (best_bw / original_bw - 1) * 100
    print(f"Bandwidth improvement: {bw_improvement:+.1f}% vs original custom CUDA")

    if best_bw >= torch_bw:
        print(f"✅ MATCHED torch.compile bandwidth! ({best_bw:.1f} vs {torch_bw:.1f} GB/s)")
    else:
        gap = (torch_bw / best_bw - 1) * 100
        print(f"Gap to torch.compile: {gap:.1f}% ({best_bw:.1f} vs {torch_bw:.1f} GB/s)")

    print()
    print("Optimizations applied:")
    print("  ✅ Vectorized loads (float4) - 4× memory throughput per load")
    print("  ✅ Fused reductions - reduced syncthreads overhead")
    print("  ✅ Cache hints (__ldg) - better L2 cache utilization")
    print("  ✅ Block size tuning - tested 256, 512, 1024 threads")
    print()

print("=" * 80)
print("NEXT STEPS FOR FURTHER OPTIMIZATION")
print("=" * 80)
print()
print("If bandwidth is still below 300 GB/s, try:")
print("  1. Software pipelining (prefetch next iteration)")
print("  2. Occupancy analysis (cudaOccupancyMaxActiveBlocksPerMultiprocessor)")
print("  3. Non-temporal stores (__stwt) for compute capability 8.0+")
print("  4. Profile with nsys/ncu to identify bottlenecks")
print()
