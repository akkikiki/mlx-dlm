#!/usr/bin/env python3
"""
Debug the manual Metal kernel to find the NaN issue.
"""

import mlx.core as mx
import mlx.core.fast as fast


# Simplified kernel for debugging
SIMPLE_METAL_SOURCE = """
// Get thread info
uint tid = thread_position_in_threadgroup.x;
uint bid = threadgroup_position_in_grid.x;
uint threads_per_group = threads_per_threadgroup.x;

// Get parameters
float temperature = params[0];
float top_nsigma = params[1];
uint vocab_size = uint(params[2]);

uint offset = bid * vocab_size;
device const float* input_logits = input0 + offset;
device float* output_logits = output0 + offset;

// Just do temperature scaling first (test if this works)
for (uint i = tid; i < vocab_size; i += threads_per_group) {
    float val = input_logits[i];
    output_logits[i] = val / temperature;
}
"""

def test_simple_kernel():
    """Test just temperature scaling."""
    print("Testing simple temperature scaling only...")

    kernel = fast.metal_kernel(
        name="simple_temp_scale",
        input_names=["input0", "params"],
        output_names=["output0"],
        source=SIMPLE_METAL_SOURCE
    )

    # Small test
    logits = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=mx.float32)
    temperature = 2.0
    params = mx.array([temperature, 2.0, 5.0], dtype=mx.float32)

    print(f"Input: {logits}")
    print(f"Temperature: {temperature}")

    result = kernel(
        inputs=[logits, params],
        output_shapes=[logits.shape],
        output_dtypes=[logits.dtype],
        grid=(1, 1, 1),
        threadgroup=(256, 1, 1)
    )[0]

    mx.eval(result)
    print(f"Output: {result}")
    print(f"Expected: {logits / temperature}")

    # Check
    expected = logits / temperature
    if mx.allclose(result, expected):
        print("✅ Temperature scaling works!")
        return True
    else:
        print("❌ Temperature scaling broken!")
        print(f"Max diff: {mx.max(mx.abs(result - expected)).item()}")
        return False


def test_auto_version():
    """Test the auto-generated version for comparison."""
    print("\nTesting auto-generated version...")

    logits = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=mx.float32)
    temperature = 2.0
    top_nsigma = 2.0

    # Auto version
    scaled = logits / temperature
    maximum = mx.max(scaled, axis=-1, keepdims=True)
    std = mx.std(scaled, axis=-1, keepdims=True)
    threshold = maximum - top_nsigma * std
    result = mx.where(scaled >= threshold, scaled, mx.array(-float('inf')))

    print(f"Input: {logits}")
    print(f"Scaled: {scaled}")
    print(f"Max: {maximum}")
    print(f"Std: {std}")
    print(f"Threshold: {threshold}")
    print(f"Result: {result}")

    # Check for NaN
    if mx.any(mx.isnan(result)):
        print("❌ Auto version has NaN!")
    else:
        print("✅ Auto version OK!")


if __name__ == "__main__":
    print("=" * 80)
    print("DEBUGGING MANUAL KERNEL")
    print("=" * 80)
    print()

    test_auto_version()
    print()
    print("-" * 80)
    print()
    test_simple_kernel()
