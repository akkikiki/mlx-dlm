#!/usr/bin/env python3
"""
Manual Metal kernel implementation for temperature + top-nsigma sampling.

This demonstrates writing the kernel by hand instead of using @mx.compile.
You get full visibility and control over the Metal code!
"""

import mlx.core as mx
import mlx.core.fast as fast
import time


# ============================================================================
# METAL KERNEL: Temperature + Top-NSigma Filter
# ============================================================================

METAL_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

METAL_SOURCE = """
// Get thread info
uint tid = thread_position_in_threadgroup.x;
uint bid = threadgroup_position_in_grid.x;
uint threads_per_group = threads_per_threadgroup.x;

// Get parameters (temperature, top_nsigma, vocab_size)
float temperature = params[0];
float top_nsigma = params[1];
uint vocab_size = uint(params[2]);

// Shared memory for reductions (max 256 threads)
threadgroup float shared[256];

// Each threadgroup handles one position
uint offset = bid * vocab_size;

// ========================================================================
// STEP 1: Temperature Scaling
// ========================================================================
// Each thread processes multiple elements
for (uint i = tid; i < vocab_size; i += threads_per_group) {
    float val = input0[offset + i];
    output0[offset + i] = val / temperature;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// ========================================================================
// STEP 2: Find Maximum (parallel reduction)
// ========================================================================
float local_max = -INFINITY;
for (uint i = tid; i < vocab_size; i += threads_per_group) {
    local_max = max(local_max, output0[offset + i]);
}
shared[tid] = local_max;
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint stride = threads_per_group / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
        shared[tid] = max(shared[tid], shared[tid + stride]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
float maximum = shared[0];
threadgroup_barrier(mem_flags::mem_threadgroup);

// ========================================================================
// STEP 3: Compute Mean (parallel reduction)
// ========================================================================
float sum = 0.0f;
for (uint i = tid; i < vocab_size; i += threads_per_group) {
    sum += output0[offset + i];
}
shared[tid] = sum;
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint stride = threads_per_group / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
        shared[tid] += shared[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
float mean = shared[0] / float(vocab_size);
threadgroup_barrier(mem_flags::mem_threadgroup);

// ========================================================================
// STEP 4: Compute Variance (parallel reduction)
// ========================================================================
float var_sum = 0.0f;
for (uint i = tid; i < vocab_size; i += threads_per_group) {
    float diff = output0[offset + i] - mean;
    var_sum += diff * diff;
}
shared[tid] = var_sum;
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint stride = threads_per_group / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
        shared[tid] += shared[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
float variance = shared[0] / float(vocab_size);
float std_dev = sqrt(variance);
threadgroup_barrier(mem_flags::mem_threadgroup);

// ========================================================================
// STEP 5: Apply Top-NSigma Filter
// ========================================================================
float threshold = maximum - top_nsigma * std_dev;

for (uint i = tid; i < vocab_size; i += threads_per_group) {
    float val = output0[offset + i];
    output0[offset + i] = (val >= threshold) ? val : -INFINITY;
}
"""


def create_temperature_topnsigma_kernel():
    """
    Create the custom Metal kernel for temperature + top-nsigma filtering.

    Returns the kernel function.
    """
    kernel = fast.metal_kernel(
        name="temperature_topnsigma_filter",
        input_names=["input0", "params"],
        output_names=["output0"],
        source=METAL_SOURCE,
        header=METAL_HEADER
    )

    return kernel


# ============================================================================
# PYTHON WRAPPER
# ============================================================================

def temperature_topnsigma_filter_manual(
    logits: mx.array,
    temperature: float,
    top_nsigma: float
) -> mx.array:
    """
    Apply temperature scaling + top-nsigma filtering using custom Metal kernel.

    Args:
        logits: Input logits (batch_size, seq_len, vocab_size) or (batch*seq, vocab_size)
        temperature: Temperature value
        top_nsigma: Top-nsigma threshold

    Returns:
        Filtered logits
    """
    # Reshape to 2D if needed
    original_shape = logits.shape
    if len(logits.shape) == 3:
        batch_size, seq_len, vocab_size = logits.shape
        logits_2d = logits.reshape(-1, vocab_size)
    else:
        logits_2d = logits

    num_positions = logits_2d.shape[0]
    vocab_size = logits_2d.shape[1]

    # Create kernel
    kernel = create_temperature_topnsigma_kernel()

    # Parameters array (temperature, top_nsigma, vocab_size)
    params = mx.array([temperature, top_nsigma, float(vocab_size)], dtype=mx.float32)

    # Grid and threadgroup sizes
    # Grid = total number of threads across all threadgroups
    # Threadgroup = threads per threadgroup
    threads_per_group = min(256, vocab_size)
    total_threads = num_positions * threads_per_group

    grid = (total_threads, 1, 1)
    threadgroup = (threads_per_group, 1, 1)

    # Call kernel (returns list of outputs)
    results = kernel(
        inputs=[logits_2d, params],
        output_shapes=[logits_2d.shape],
        output_dtypes=[logits_2d.dtype],
        grid=grid,
        threadgroup=threadgroup
    )

    # Extract first (and only) output
    result = results[0]

    # Reshape back if needed
    if len(original_shape) == 3:
        result = result.reshape(original_shape)

    return result


# ============================================================================
# COMPARISON WITH AUTO-GENERATED VERSION
# ============================================================================

def temperature_topnsigma_filter_auto(
    logits: mx.array,
    temperature: float,
    top_nsigma: float
) -> mx.array:
    """Auto-generated version using @mx.compile (for comparison)."""
    logits = logits / temperature
    maximum = mx.max(logits, axis=-1, keepdims=True)
    std = mx.std(logits, axis=-1, keepdims=True)
    threshold = maximum - top_nsigma * std
    logits = mx.where(logits >= threshold, logits, mx.full(logits.shape, float("-inf")))
    return logits


# ============================================================================
# TESTS AND BENCHMARKS
# ============================================================================

def test_correctness():
    """Test that manual kernel produces correct results."""
    print("=" * 80)
    print("CORRECTNESS TEST")
    print("=" * 80)
    print()

    # Create test data
    batch_size = 4
    seq_len = 8
    vocab_size = 1000

    logits = mx.random.normal((batch_size, seq_len, vocab_size))
    temperature = 1.5
    top_nsigma = 2.0

    print(f"Input shape: {logits.shape}")
    print(f"Temperature: {temperature}")
    print(f"Top-nsigma: {top_nsigma}")
    print()

    # Run manual kernel
    print("Running manual Metal kernel...")
    result_manual = temperature_topnsigma_filter_manual(logits, temperature, top_nsigma)
    mx.eval(result_manual)

    # Run auto-generated version
    print("Running auto-generated (@mx.compile) version...")
    result_auto = temperature_topnsigma_filter_auto(logits, temperature, top_nsigma)
    mx.eval(result_auto)

    # Compare results
    print()
    print("Comparing results...")

    # Compare finite values and inf positions separately to avoid NaN
    finite_mask = mx.isfinite(result_manual) & mx.isfinite(result_auto)
    inf_mask_manual = mx.isinf(result_manual)
    inf_mask_auto = mx.isinf(result_auto)

    # Check that inf positions match
    inf_positions_match = mx.all(inf_mask_manual == inf_mask_auto).item()

    # Compute difference for finite values only
    if mx.any(finite_mask).item():
        finite_diff = mx.where(finite_mask, mx.abs(result_manual - result_auto), mx.array(0.0))
        max_diff = mx.max(finite_diff).item()
        mean_diff = mx.mean(finite_diff).item()
    else:
        max_diff = 0.0
        mean_diff = 0.0

    print(f"  Max difference (finite values): {max_diff:.2e}")
    print(f"  Mean difference (finite values): {mean_diff:.2e}")
    print(f"  Inf positions match: {inf_positions_match}")

    if max_diff < 1e-4 and inf_positions_match:
        print("  ✅ Results match!")
    else:
        print("  ⚠️  Results differ")
        if not inf_positions_match:
            print("      - Inf positions don't match")
        if max_diff >= 1e-4:
            print(f"      - Finite values differ by {max_diff:.2e}")

    print()


def benchmark_performance():
    """Benchmark manual vs auto-generated kernel."""
    print("=" * 80)
    print("PERFORMANCE BENCHMARK")
    print("=" * 80)
    print()

    # Realistic size for LLaDA2
    batch_size = 8
    seq_len = 32
    vocab_size = 156896

    logits = mx.random.normal((batch_size, seq_len, vocab_size))
    temperature = 1.0
    top_nsigma = 2.0

    print(f"Configuration:")
    print(f"  Shape: {logits.shape}")
    print(f"  Total positions: {batch_size * seq_len}")
    print(f"  Vocab size: {vocab_size}")
    print()

    # Warmup
    print("Warming up...")
    for _ in range(3):
        result_manual = temperature_topnsigma_filter_manual(logits, temperature, top_nsigma)
        mx.eval(result_manual)

        result_auto = temperature_topnsigma_filter_auto(logits, temperature, top_nsigma)
        mx.eval(result_auto)
    print()

    # Benchmark manual kernel
    print("Benchmarking manual Metal kernel...")
    times_manual = []
    for _ in range(10):
        start = time.time()
        result = temperature_topnsigma_filter_manual(logits, temperature, top_nsigma)
        mx.eval(result)
        times_manual.append((time.time() - start) * 1000)

    avg_manual = sum(times_manual) / len(times_manual)
    print(f"  Average: {avg_manual:.2f}ms")

    # Benchmark auto-generated
    print("Benchmarking auto-generated kernel...")
    times_auto = []
    for _ in range(10):
        start = time.time()
        result = temperature_topnsigma_filter_auto(logits, temperature, top_nsigma)
        mx.eval(result)
        times_auto.append((time.time() - start) * 1000)

    avg_auto = sum(times_auto) / len(times_auto)
    print(f"  Average: {avg_auto:.2f}ms")

    print()
    print("-" * 80)
    print("COMPARISON")
    print("-" * 80)
    print(f"Manual kernel:      {avg_manual:.2f}ms")
    print(f"Auto-generated:     {avg_auto:.2f}ms")

    if avg_manual < avg_auto:
        speedup = avg_auto / avg_manual
        print(f"Manual is {speedup:.2f}x FASTER ✅")
    else:
        slowdown = avg_manual / avg_auto
        print(f"Manual is {slowdown:.2f}x slower")
        print("(Auto-generated has better optimization)")

    print()


def show_kernel_source():
    """Display the actual Metal kernel source."""
    print("=" * 80)
    print("ACTUAL METAL KERNEL SOURCE")
    print("=" * 80)
    print()
    print("This is the ACTUAL Metal code that runs on the GPU:")
    print()
    print("-" * 80)
    print("HEADER (Helper Functions):")
    print("-" * 80)
    print(METAL_HEADER)
    print()
    print("-" * 80)
    print("KERNEL BODY (Main Function):")
    print("-" * 80)
    print(METAL_SOURCE)
    print()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys

    print("\n" * 2)
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  MANUAL METAL KERNEL FOR TEMPERATURE + TOP-NSIGMA SAMPLING".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    print("This demonstrates writing custom Metal kernels by hand,")
    print("giving you full visibility and control over the GPU code!")
    print()

    if len(sys.argv) > 1 and sys.argv[1] == "--show-source":
        show_kernel_source()
    else:
        test_correctness()
        benchmark_performance()

        print("=" * 80)
        print("TIP: Run with --show-source to see the Metal kernel code")
        print("=" * 80)
        print()

        print("Key Takeaways:")
        print()
        print("✅ You CAN write custom Metal kernels in MLX")
        print("✅ Full visibility - you see every line of GPU code")
        print("✅ Full control - optimize however you want")
        print("✅ Same API as PyTorch custom CUDA kernels")
        print()
        print("Trade-offs:")
        print("  • Manual: More work, but full control")
        print("  • Auto (@mx.compile): Less work, MLX optimizes for you")
        print()
        print("For production: Use @mx.compile (easier, maintained)")
        print("For learning/research: Write custom kernels (educational)")
        print()
