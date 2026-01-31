#!/usr/bin/env python3
"""Check if the optimized kernel was built correctly."""

import os
import sys

print("=" * 80)
print("CHECKING BUILD STATUS")
print("=" * 80)
print()

# List .so files
so_files = [f for f in os.listdir('.') if f.endswith('.so')]
print(f"Found .so files:")
for f in so_files:
    size = os.path.getsize(f)
    print(f"  {f} ({size:,} bytes)")
print()

# Try to import
print("Trying to import...")
try:
    import temperature_topnsigma_optimized
    print("✅ Import successful!")
    print(f"Module: {temperature_topnsigma_optimized}")
    print(f"Available functions: {dir(temperature_topnsigma_optimized)}")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print()
    print("Trying manual load...")

    # Find the .so file
    opt_so = [f for f in so_files if 'optimized' in f]
    if opt_so:
        so_file = opt_so[0]
        print(f"Found: {so_file}")

        # Try loading with ctypes
        import ctypes
        try:
            lib = ctypes.CDLL(f"./{so_file}")
            print(f"✅ ctypes load successful")
        except Exception as e2:
            print(f"❌ ctypes load failed: {e2}")
    else:
        print("❌ No optimized .so file found!")
        print("   Expected: temperature_topnsigma_optimized*.so")

print()
print("Python path:")
for p in sys.path[:5]:
    print(f"  {p}")
