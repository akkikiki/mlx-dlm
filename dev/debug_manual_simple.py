#!/usr/bin/env python3
"""Simple debug test for manual kernel."""

import mlx.core as mx
from manual_metal_kernel import temperature_topnsigma_filter_manual, temperature_topnsigma_filter_auto

# Test with batch size (like the failing test)
# Original failing test uses (4, 8, 1000) which reshapes to (32, 1000)
logits = mx.random.normal((4, 1000), dtype=mx.float32)
temperature = 1.5
top_nsigma = 2.0

print("=" * 80)
print("SIMPLE DEBUG TEST")
print("=" * 80)
print()
print(f"Input: {logits}")
print(f"Temperature: {temperature}")
print(f"Top-nsigma: {top_nsigma}")
print()

# Run auto version (reference)
print("Auto version (reference):")
result_auto = temperature_topnsigma_filter_auto(logits, temperature, top_nsigma)
mx.eval(result_auto)
print(f"Result: {result_auto}")
print()

# Compute steps manually to verify
scaled = logits / temperature
print(f"1. Scaled: {scaled}")

maximum = mx.max(scaled, axis=-1, keepdims=True)
print(f"2. Maximum: {maximum}")

std = mx.std(scaled, axis=-1, keepdims=True)
print(f"3. Std: {std}")

threshold = maximum - top_nsigma * std
print(f"4. Threshold: {threshold}")

filtered = mx.where(scaled >= threshold, scaled, mx.array(float("-inf")))
print(f"5. Filtered: {filtered}")
print()

# Run manual kernel
print("Manual Metal kernel:")
result_manual = temperature_topnsigma_filter_manual(logits, temperature, top_nsigma)
mx.eval(result_manual)
print(f"Result: {result_manual}")
print()

# Compare
print("Comparison:")
print(f"Auto:   {result_auto}")
print(f"Manual: {result_manual}")

finite_mask = mx.isfinite(result_auto) & mx.isfinite(result_manual)
if mx.any(finite_mask):
    diff = mx.where(finite_mask, mx.abs(result_auto - result_manual), mx.array(0.0))
    max_diff = mx.max(diff).item()
    print(f"Max difference (finite): {max_diff}")
else:
    print("No finite values to compare")

inf_match = mx.all(mx.isinf(result_auto) == mx.isinf(result_manual)).item()
print(f"Inf positions match: {inf_match}")

if max_diff < 1e-5 and inf_match:
    print("✅ PASS")
else:
    print("❌ FAIL")
