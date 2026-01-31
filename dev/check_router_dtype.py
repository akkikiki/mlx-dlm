#!/usr/bin/env python3
"""
Check the dtype of router/gate weights in the 8-bit LLaDA2 model.
"""

import mlx.core as mx
from mlx_lm import load

print("Loading 8-bit LLaDA2 model...")
model, tokenizer = load("mlx-community/LLaDA2.0-mini-8bit")
print("Model loaded!\n")

print("="*80)
print("Checking layer dtypes...")
print("="*80)

# Collect layer info
gate_layers = []
expert_layers = []
norm_layers = []
other_layers = []

for name, module in model.named_modules():
    if hasattr(module, 'weight'):
        dtype = module.weight.dtype
        shape = module.weight.shape

        if "gate" in name.lower():
            gate_layers.append((name, dtype, shape))
        elif "expert" in name.lower():
            expert_layers.append((name, dtype, shape))
        elif "norm" in name.lower():
            norm_layers.append((name, dtype, shape))
        else:
            other_layers.append((name, dtype, shape))

# Print gate/router layers
print("\n🎯 GATE/ROUTER LAYERS:")
print("-"*80)
if gate_layers:
    for name, dtype, shape in gate_layers[:5]:  # Show first 5
        print(f"{name}")
        print(f"  dtype: {dtype}")
        print(f"  shape: {shape}")
        print()
    if len(gate_layers) > 5:
        print(f"... and {len(gate_layers) - 5} more gate layers")
else:
    print("No gate layers found!")

# Print some expert layers for comparison
print("\n⚙️  EXPERT LAYERS (for comparison):")
print("-"*80)
if expert_layers:
    for name, dtype, shape in expert_layers[:3]:  # Show first 3
        print(f"{name}")
        print(f"  dtype: {dtype}")
        print(f"  shape: {shape}")
        print()
    if len(expert_layers) > 3:
        print(f"... and {len(expert_layers) - 3} more expert layers")

# Print norm layers
print("\n📏 NORM LAYERS:")
print("-"*80)
if norm_layers:
    for name, dtype, shape in norm_layers[:3]:  # Show first 3
        print(f"{name}")
        print(f"  dtype: {dtype}")
        print(f"  shape: {shape}")
        print()
    if len(norm_layers) > 3:
        print(f"... and {len(norm_layers) - 3} more norm layers")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

# Count dtypes
dtype_counts = {}
for layers in [gate_layers, expert_layers, norm_layers, other_layers]:
    for _, dtype, _ in layers:
        dtype_str = str(dtype)
        dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1

print("\nDtype distribution:")
for dtype, count in sorted(dtype_counts.items()):
    print(f"  {dtype}: {count} layers")

# Check if gates are fp32
gate_dtypes = set(str(dtype) for _, dtype, _ in gate_layers)
print(f"\nGate/Router dtypes: {gate_dtypes}")

if gate_dtypes == {"float32"}:
    print("✅ Routers are FP32 (as expected from config.json)")
elif gate_dtypes == {"float16"}:
    print("⚠️  Routers are FP16 (not FP32 as config suggests)")
elif any("uint" in str(d) for d in gate_dtypes):
    print("❌ Routers are QUANTIZED (unexpected!)")
else:
    print(f"❓ Routers have unexpected dtypes: {gate_dtypes}")

# Check if experts are quantized
expert_dtypes = set(str(dtype) for _, dtype, _ in expert_layers)
print(f"\nExpert dtypes: {expert_dtypes}")

if any("uint" in str(d) for d in expert_dtypes):
    print("✅ Experts are QUANTIZED (as expected for 8-bit model)")
else:
    print("⚠️  Experts are NOT quantized")

print("="*80)
