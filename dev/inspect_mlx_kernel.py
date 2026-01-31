#!/usr/bin/env python3
"""
Attempt to inspect MLX's internal kernel compilation.

Note: MLX doesn't expose generated Metal source code directly,
but we can see what gets compiled and make educated guesses.
"""

import mlx.core as mx
from functools import partial


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def temperature_topnsigma_sample(logits, temperature, top_nsigma):
    """Our fixed-algorithm kernel."""
    logits = logits / temperature
    maximum = mx.max(logits, axis=-1, keepdims=True)
    std = mx.std(logits, axis=-1, keepdims=True)
    threshold = maximum - top_nsigma * std
    logits = mx.where(logits >= threshold, logits, mx.full(logits.shape, float("-inf")))
    probs = mx.softmax(logits, axis=-1)
    orig_shape = probs.shape[:-1]
    flat_probs = probs.reshape(-1, probs.shape[-1])
    flat_tokens = mx.random.categorical(mx.log(flat_probs + 1e-10))
    tokens = flat_tokens.reshape(orig_shape)
    token_probs = mx.take_along_axis(probs, mx.expand_dims(tokens, axis=-1), axis=-1).squeeze(-1)
    return tokens, token_probs


print("=" * 80)
print("Inspecting MLX Kernel Compilation")
print("=" * 80)
print()

# Create input
logits = mx.random.normal((8, 32, 156896))

print("Python function:")
print("-" * 80)
print("""
@mx.compile
def temperature_topnsigma_sample(logits, temperature, top_nsigma):
    logits = logits / temperature
    maximum = mx.max(logits, axis=-1, keepdims=True)
    std = mx.std(logits, axis=-1, keepdims=True)
    threshold = maximum - top_nsigma * std
    logits = mx.where(logits >= threshold, logits, -inf)
    probs = mx.softmax(logits, axis=-1)
    # ... sampling ...
""")
print("-" * 80)
print()

# Call the function
print("Calling function (this triggers compilation)...")
result = temperature_topnsigma_sample(logits, 1.0, 2.0)
mx.eval(result[0])
print("✓ Compiled and executed")
print()

print("=" * 80)
print("What MLX Does Internally")
print("=" * 80)
print()
print("1. Parse Python code → Build computation graph")
print("   - Node 1: divide (logits, temperature)")
print("   - Node 2: max (divide_output, axis=-1)")
print("   - Node 3: std (divide_output, axis=-1)")
print("   - Node 4: subtract (max_output, nsigma * std_output)")
print("   - Node 5: where (divide_output >= threshold, ...)")
print("   - Node 6: softmax (where_output)")
print("   - Node 7: categorical (log(softmax_output))")
print()
print("2. Optimize graph → Identify fusion opportunities")
print("   - Fuse: divide + max + std + threshold + where → single kernel")
print("   - Fuse: softmax (already a fused primitive)")
print()
print("3. Generate Metal code → Compile with Metal compiler")
print("   - Write Metal Shading Language (.metal) source")
print("   - Call metal compiler (runtime)")
print("   - Cache compiled binary in memory")
print()
print("4. Execute on GPU")
print("   - Bind buffers")
print("   - Launch kernel")
print("   - Return results")
print()

print("=" * 80)
print("Where is the generated Metal code?")
print("=" * 80)
print()
print("❌ NOT accessible from Python API")
print("❌ NOT saved to disk")
print("✅ Generated in C++ (mlx/core/compile.cpp)")
print("✅ Compiled by Metal compiler at runtime")
print("✅ Cached in process memory as binary")
print()
print("To see actual Metal code, you would need to:")
print("  1. Modify MLX C++ source to dump generated .metal files")
print("  2. Set Metal environment variables to log shader compilation")
print("  3. Use Xcode Instruments to capture Metal trace")
print()

print("=" * 80)
print("Alternative: Set Metal Debug Environment")
print("=" * 80)
print()
print("You can enable Metal shader debugging:")
print()
print("  export MTL_DEBUG_LAYER=1")
print("  export MTL_SHADER_VALIDATION=1")
print("  python your_script.py")
print()
print("This enables validation but doesn't expose source code.")
print()

print("=" * 80)
print("What the kernel CONCEPTUALLY looks like (next section)")
print("=" * 80)
