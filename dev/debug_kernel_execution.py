#!/usr/bin/env python3
"""Debug kernel execution with print statements."""

import mlx.core as mx
import mlx.core.fast as fast

# Simple kernel that just copies input to output with thread/block info
DEBUG_KERNEL = """
uint tid = thread_position_in_threadgroup.x;
uint bid = threadgroup_position_in_grid.x;
uint vocab_size = uint(params[0]);
uint offset = bid * vocab_size;

// Just copy input to output (to verify threadgroups are running)
for (uint i = tid; i < vocab_size; i += threads_per_threadgroup.x) {
    output0[offset + i] = input0[offset + i] + float(bid);  // Add bid to verify
}
"""

def test_grid():
    """Test if grid parameter works correctly for multiple threadgroups."""
    print("Testing grid execution...")

    kernel = fast.metal_kernel(
        name="debug_copy",
        input_names=["input0", "params"],
        output_names=["output0"],
        source=DEBUG_KERNEL
    )

    # Test with 4 rows, 10 columns
    batch_size = 4
    vocab_size = 10

    logits = mx.ones((batch_size, vocab_size), dtype=mx.float32)
    params = mx.array([float(vocab_size)], dtype=mx.float32)

    print(f"Input shape: {logits.shape}")
    print(f"Input:\n{logits}")

    # Grid: Total number of threads (batch_size * threads_per_batch)
    # Threadgroup: threads per threadgroup
    threads_per_batch = vocab_size
    total_threads = batch_size * threads_per_batch

    print(f"Grid (total threads): {total_threads}")
    print(f"Threadgroup size: {threads_per_batch}")
    print(f"Number of threadgroups: {total_threads // threads_per_batch}")

    result = kernel(
        inputs=[logits, params],
        output_shapes=[logits.shape],
        output_dtypes=[logits.dtype],
        grid=(total_threads, 1, 1),
        threadgroup=(threads_per_batch, 1, 1)
    )[0]

    mx.eval(result)
    print(f"\nOutput:\n{result}")
    print("\nExpected:")
    print("  Row 0: all 1.0 (1.0 + 0)")
    print("  Row 1: all 2.0 (1.0 + 1)")
    print("  Row 2: all 3.0 (1.0 + 2)")
    print("  Row 3: all 4.0 (1.0 + 3)")

    # Check
    expected = mx.array([
        [1.0] * vocab_size,
        [2.0] * vocab_size,
        [3.0] * vocab_size,
        [4.0] * vocab_size
    ])

    if mx.allclose(result, expected):
        print("\n✅ Grid works correctly!")
    else:
        print("\n❌ Grid NOT working!")
        print(f"Expected:\n{expected}")

if __name__ == "__main__":
    test_grid()
