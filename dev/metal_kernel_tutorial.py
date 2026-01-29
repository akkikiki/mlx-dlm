#!/usr/bin/env python3
"""
Example of writing custom Metal kernels in MLX.

This demonstrates that MLX DOES support custom kernels,
just like PyTorch supports custom CUDA kernels!

Documentation: https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
"""

import mlx.core as mx
import mlx.core.fast as fast


# ============================================================================
# Example 1: Simple Element-Wise Operation
# ============================================================================

def create_custom_relu():
    """Create a custom ReLU kernel in Metal."""

    # Metal kernel source code
    metal_source = """
    uint idx = thread_position_in_grid.x;
    T val = input0[idx];
    output0[idx] = val > 0 ? val : 0;
    """

    # Create the kernel
    kernel = fast.metal_kernel(
        name="custom_relu",
        input_names=["input0"],
        output_names=["output0"],
        source=metal_source
    )

    return kernel


# ============================================================================
# Example 2: Custom Fused Operation (Like Our Sampling Kernel!)
# ============================================================================

def create_temperature_scale_kernel():
    """
    Custom kernel for temperature scaling.
    This is what you COULD do if you wanted manual control!
    """

    metal_source = """
    // Thread index
    uint idx = thread_position_in_grid.x;

    // Load input
    float logit = input0[idx];
    float temperature = params[0];

    // Apply temperature scaling
    output0[idx] = logit / temperature;
    """

    kernel = fast.metal_kernel(
        name="temperature_scale",
        input_names=["input0", "params"],
        output_names=["output0"],
        source=metal_source
    )

    return kernel


# ============================================================================
# Example 3: Complex Kernel with Header (Helper Functions)
# ============================================================================

def create_softmax_kernel():
    """Custom softmax with helper functions."""

    # Header with helper functions
    metal_header = """
    #include <metal_stdlib>
    using namespace metal;

    // Helper: Find maximum using parallel reduction
    float parallel_max(
        device const float* data,
        uint size,
        uint tid,
        uint threads,
        threadgroup float* shared
    ) {
        float local_max = -INFINITY;
        for (uint i = tid; i < size; i += threads) {
            local_max = max(local_max, data[i]);
        }
        shared[tid] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Reduction
        for (uint stride = threads / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                shared[tid] = max(shared[tid], shared[tid + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        return shared[0];
    }
    """

    # Main kernel body
    metal_source = """
    // This would be the actual softmax implementation
    // using the helper functions from the header
    uint idx = thread_position_in_grid.x;

    // Simplified: just pass through for demonstration
    output0[idx] = input0[idx];
    """

    kernel = fast.metal_kernel(
        name="custom_softmax",
        input_names=["input0"],
        output_names=["output0"],
        source=metal_source,
        header=metal_header  # Include helper functions!
    )

    return kernel


# ============================================================================
# Example 4: The Temperature + Top-NSigma Kernel (Manual Implementation)
# ============================================================================

def create_manual_temperature_topnsigma_kernel():
    """
    This is how you COULD manually implement the fixed-algorithm kernel!

    Instead of using @mx.compile (which auto-generates Metal),
    you write the Metal code yourself for full control.
    """

    metal_header = """
    #include <metal_stdlib>
    using namespace metal;

    // Helper: Parallel max reduction
    float reduce_max(
        device const float* data,
        uint size,
        uint tid,
        threadgroup float* shared
    ) {
        float local_max = -INFINITY;
        for (uint i = tid; i < size; i += 256) {
            local_max = max(local_max, data[i]);
        }
        shared[tid] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint s = 128; s > 0; s >>= 1) {
            if (tid < s) {
                shared[tid] = max(shared[tid], shared[tid + s]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        return shared[0];
    }

    // Helper: Parallel standard deviation
    float reduce_std(
        device const float* data,
        uint size,
        float mean,
        uint tid,
        threadgroup float* shared
    ) {
        float local_sum = 0.0f;
        for (uint i = tid; i < size; i += 256) {
            float diff = data[i] - mean;
            local_sum += diff * diff;
        }
        shared[tid] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint s = 128; s > 0; s >>= 1) {
            if (tid < s) {
                shared[tid] += shared[tid + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        return sqrt(shared[0] / size);
    }
    """

    metal_source = """
    // Parameters from Python
    float temperature = params[0];
    float top_nsigma = params[1];
    uint vocab_size = shape0[0];  // Auto-passed by MLX

    // Thread info
    uint tid = thread_position_in_threadgroup.x;

    // Shared memory for reductions
    threadgroup float shared[256];

    // Step 1: Temperature scaling (in-place)
    for (uint i = tid; i < vocab_size; i += 256) {
        input0[i] = input0[i] / temperature;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Find max
    float maximum = reduce_max(input0, vocab_size, tid, shared);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Compute mean
    float sum = 0.0f;
    for (uint i = tid; i < vocab_size; i += 256) {
        sum += input0[i];
    }
    shared[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = shared[0] / vocab_size;

    // Step 4: Compute std dev
    float std_dev = reduce_std(input0, vocab_size, mean, tid, shared);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 5: Apply top-nsigma filter
    float threshold = maximum - top_nsigma * std_dev;
    for (uint i = tid; i < vocab_size; i += 256) {
        float val = input0[i];
        output0[i] = (val >= threshold) ? val : -INFINITY;
    }
    """

    kernel = fast.metal_kernel(
        name="manual_temperature_topnsigma",
        input_names=["input0", "params"],
        output_names=["output0"],
        source=metal_source,
        header=metal_header
    )

    return kernel


# ============================================================================
# Test the Kernels
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MLX Custom Metal Kernels Demo")
    print("=" * 80)
    print()
    print("YES! MLX supports custom Metal kernels, just like PyTorch")
    print("supports custom CUDA kernels!")
    print()

    # Test custom ReLU
    print("Testing custom ReLU kernel...")
    custom_relu = create_custom_relu()

    x = mx.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = custom_relu(x)
    mx.eval(result)

    print(f"Input:  {x}")
    print(f"Output: {result}")
    print("✓ Custom Metal kernel works!")
    print()

    print("=" * 80)
    print("Key Points")
    print("=" * 80)
    print()
    print("MLX provides TWO ways to write GPU code:")
    print()
    print("1. Auto-generated (high-level):")
    print("   @mx.compile")
    print("   def func(x): return mx.softmax(x)")
    print("   → MLX generates Metal automatically")
    print("   → Can't see generated code")
    print("   → Easy to use")
    print()
    print("2. Custom kernels (low-level):")
    print("   metal_kernel(source='your Metal code')")
    print("   → You write Metal code yourself")
    print("   → Full control and visibility")
    print("   → More work")
    print()
    print("This is EXACTLY like PyTorch:")
    print("  - torch.compile (auto-generated, hidden)")
    print("  - Custom CUDA extensions (manual, visible)")
    print()
    print("=" * 80)
    print()
    print("Documentation:")
    print("  https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html")
    print("=" * 80)
