#!/usr/bin/env python3
"""
Test the debug kernel (standalone) on large input to see if it works.
"""

import torch
import ctypes
import subprocess

# Compile debug kernel
print("Compiling debug kernel...")
cmd = [
    "nvcc", "-shared", "-Xcompiler", "-fPIC",
    "-o", "debug_kernel.so", "debug_kernel.cu",
    "-O3", "--use_fast_math"
]
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print("❌ Failed:", result.stderr)
    exit(1)
print("✅ Compiled\n")

# Load
lib = ctypes.CDLL("./debug_kernel.so")
lib.launch_temperature_topnsigma_debug.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_float, ctypes.c_float, ctypes.c_void_p,
]

# Test on LARGE input
batch, seq, vocab = 8, 32, 156896
temperature = 1.0
top_nsigma = 2.0

print(f"Testing on LARGE input: ({batch}, {seq}, {vocab})")
print()

torch.manual_seed(42)
logits = torch.randn(batch, seq, vocab, device='cuda')

# PyTorch reference
print("PyTorch:")
ref = logits / temperature
ref_max = ref.max(dim=-1, keepdim=True)[0]
ref_std = ref.std(dim=-1, keepdim=True)
ref_thresh = ref_max - top_nsigma * ref_std
ref_result = torch.where(ref >= ref_thresh, ref, torch.tensor(float('-inf'), device='cuda'))
print(f"  Total infs: {torch.isinf(ref_result).sum().item()}")
print(f"  First position infs: {torch.isinf(ref_result[0,0]).sum().item()}")
print()

# Debug kernel
print("Debug kernel:")
output = torch.empty_like(logits)
debug_info = torch.zeros(batch * seq, 4, device='cuda')

lib.launch_temperature_topnsigma_debug(
    ctypes.c_void_p(logits.data_ptr()),
    ctypes.c_void_p(output.data_ptr()),
    ctypes.c_void_p(debug_info.data_ptr()),
    batch, seq, vocab,
    ctypes.c_float(temperature),
    ctypes.c_float(top_nsigma),
    ctypes.c_void_p(0)
)
torch.cuda.synchronize()

print(f"  Total infs: {torch.isinf(output).sum().item()}")
print(f"  First position infs: {torch.isinf(output[0,0]).sum().item()}")
print()

# Compare
print("Comparison:")
inf_match = (torch.isinf(ref_result) == torch.isinf(output)).all().item()
print(f"  Inf positions match: {inf_match}")

if inf_match:
    print("  ✅ Debug kernel works on LARGE input!")
else:
    print("  ❌ Debug kernel also fails on large input")
    print(f"     Expected: {torch.isinf(ref_result).sum().item()}")
    print(f"     Got: {torch.isinf(output).sum().item()}")
