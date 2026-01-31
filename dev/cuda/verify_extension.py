#!/usr/bin/env python3
"""
Verify the extension is built with the latest code.
"""

import torch
import os
import subprocess
from datetime import datetime

print("=" * 80)
print("EXTENSION VERIFICATION")
print("=" * 80)
print()

# Check if .so file exists
so_files = [f for f in os.listdir('.') if f.endswith('.so') and 'temperature' in f]

if not so_files:
    print("❌ No .so file found! Extension not built.")
    print("   Run: python setup.py build_ext --inplace")
    exit(1)

so_file = so_files[0]
print(f"Found: {so_file}")

# Check modification time
mtime = os.path.getmtime(so_file)
mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
print(f"Last modified: {mtime_str}")

# Check .cu file modification time
cu_file = "temperature_topnsigma_kernel.cu"
if os.path.exists(cu_file):
    cu_mtime = os.path.getmtime(cu_file)
    cu_mtime_str = datetime.fromtimestamp(cu_mtime).strftime('%Y-%m-%d %H:%M:%S')
    print(f".cu file modified: {cu_mtime_str}")

    if cu_mtime > mtime:
        print()
        print("⚠️  WARNING: .cu file is NEWER than .so file!")
        print("   The extension needs to be rebuilt!")
        print("   Run: python setup.py build_ext --inplace")
        print()
    else:
        print("✅ Extension is up to date")
else:
    print("❌ .cu file not found")

print()

# Try to import
try:
    import temperature_topnsigma_cuda
    print("✅ Extension imports successfully")

    # Try a simple test
    x = torch.randn(1, 1, 100, device='cuda')
    try:
        result = temperature_topnsigma_cuda.temperature_topnsigma(x, 1.5, 2.0)
        print("✅ Extension runs without errors")
    except Exception as e:
        print(f"❌ Extension errors when run: {e}")

except ImportError as e:
    print(f"❌ Cannot import extension: {e}")

print()
print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print()
print("To ensure you're testing the latest code:")
print()
print("  1. Clean old builds:")
print("     rm -rf build/ *.so")
print()
print("  2. Rebuild:")
print("     python setup.py build_ext --inplace")
print()
print("  3. Test:")
print("     python test_basic_only.py")
print()
