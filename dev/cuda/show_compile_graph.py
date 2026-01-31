#!/usr/bin/env python3
"""
Show the computation graph and kernel fusion decisions from torch.compile
"""

import torch
import torch._dynamo as dynamo
import torch._inductor.config as inductor_config

print("=" * 80)
print("TORCH.COMPILE GRAPH ANALYSIS")
print("=" * 80)
print()


def temperature_topnsigma(logits, temperature, top_nsigma):
    """The function to analyze."""
    scaled = logits / temperature
    maximum = scaled.max(dim=-1, keepdim=True)[0]
    std = scaled.std(dim=-1, keepdim=True)
    threshold = maximum - top_nsigma * std
    return torch.where(scaled >= threshold, scaled, torch.tensor(float('-inf'), device='cuda'))


# Enable graph visualization
inductor_config.debug = True
inductor_config.trace.enabled = True

# Compile
print("Compiling...")
compiled_fn = torch.compile(temperature_topnsigma, backend='inductor')

# Test inputs
batch, seq, vocab = 2, 4, 1000  # Smaller for clearer output
logits = torch.randn(batch, seq, vocab, device='cuda')

# Run to trigger compilation
print("Running compiled function...")
print()
result = compiled_fn(logits, 1.5, 2.0)
torch.cuda.synchronize()

print()
print("=" * 80)
print("GRAPH INFORMATION")
print("=" * 80)
print()

# Try to get graph info
try:
    graphs = dynamo.list_backends()
    print(f"Available backends: {graphs}")
except:
    pass

print()
print("To see detailed graphs, check the output above for:")
print("  • Captured FX graph")
print("  • Inductor graph")
print("  • Kernel fusion decisions")
print()

# Show what operations are in the original function
print("=" * 80)
print("ORIGINAL OPERATIONS")
print("=" * 80)
print()
print("1. scaled = logits / temperature           (pointwise)")
print("2. maximum = scaled.max(dim=-1)            (reduction)")
print("3. std = scaled.std(dim=-1)                (reduction)")
print("4. threshold = maximum - nsigma * std      (pointwise)")
print("5. result = where(scaled >= threshold)     (pointwise)")
print()
print("Potential fusion opportunities:")
print("  • Fuse ops 1, 4, 5 (all pointwise)")
print("  • Keep reductions separate (ops 2, 3)")
print()

print("=" * 80)
print("EXPECTED TORCH.COMPILE STRATEGY")
print("=" * 80)
print()
print("Likely generates ~3-4 kernels:")
print()
print("Kernel 1: Pointwise fusion")
print("  - Division (scaled = logits / temp)")
print("  - Broadcast threshold")
print("  - Comparison and selection")
print()
print("Kernel 2: Max reduction")
print("  - Reduce to find maximum")
print()
print("Kernel 3: Std reduction")
print("  - Compute mean")
print("  - Compute variance/std")
print()
print("Or possibly fused into 2 kernels if it combines reductions!")
print()
