#!/usr/bin/env python3
"""
Inspect what code torch.compile generates for temperature + top-nsigma
"""

import torch
import os
import tempfile
import shutil

# Enable verbose compilation logging
os.environ['TORCH_LOGS'] = '+dynamo,+aot,+inductor'
os.environ['TORCHINDUCTOR_DUMP_TRITON_CODE'] = '1'

# Create temp directory for output
output_dir = tempfile.mkdtemp(prefix='torch_compile_output_')
os.environ['TORCHINDUCTOR_CACHE_DIR'] = output_dir

print("=" * 80)
print("TORCH.COMPILE CODE INSPECTION")
print("=" * 80)
print()
print(f"Output directory: {output_dir}")
print()


def temperature_topnsigma(logits, temperature, top_nsigma):
    """The function we want to compile."""
    scaled = logits / temperature
    maximum = scaled.max(dim=-1, keepdim=True)[0]
    std = scaled.std(dim=-1, keepdim=True)
    threshold = maximum - top_nsigma * std
    return torch.where(scaled >= threshold, scaled, torch.tensor(float('-inf'), device='cuda'))


# Compile it
print("Compiling function...")
print("-" * 80)
compiled_fn = torch.compile(temperature_topnsigma, mode='max-autotune')

# Run it to trigger compilation
batch, seq, vocab = 8, 32, 156896
temperature = 1.5
top_nsigma = 2.0

logits = torch.randn(batch, seq, vocab, device='cuda')
result = compiled_fn(logits, temperature, top_nsigma)
torch.cuda.synchronize()

print()
print("=" * 80)
print("COMPILATION COMPLETE")
print("=" * 80)
print()

# Find generated files
print("Generated files:")
print("-" * 80)

triton_files = []
for root, dirs, files in os.walk(output_dir):
    for f in files:
        full_path = os.path.join(root, f)
        rel_path = os.path.relpath(full_path, output_dir)

        if f.endswith('.py') and 'triton' in f.lower():
            triton_files.append(full_path)
            print(f"  📄 {rel_path}")
        elif f.endswith('.cpp') or f.endswith('.cu'):
            print(f"  📄 {rel_path}")

print()

# Display Triton kernels
if triton_files:
    print("=" * 80)
    print("GENERATED TRITON KERNELS")
    print("=" * 80)
    print()

    for i, triton_file in enumerate(triton_files[:3]):  # Show first 3
        print(f"Kernel {i+1}: {os.path.basename(triton_file)}")
        print("-" * 80)

        try:
            with open(triton_file, 'r') as f:
                content = f.read()

            # Print first 100 lines
            lines = content.split('\n')
            for j, line in enumerate(lines[:100]):
                print(f"{j+1:4d} | {line}")

            if len(lines) > 100:
                print(f"     | ... ({len(lines) - 100} more lines)")

            print()
        except Exception as e:
            print(f"Could not read file: {e}")
            print()
else:
    print("⚠️  No Triton files found.")
    print("   torch.compile might be using a different backend.")
    print()

print("=" * 80)
print("ANALYSIS")
print("=" * 80)
print()

# Try to analyze what operations were fused
print("Looking for optimization patterns...")
print()

if triton_files:
    for triton_file in triton_files:
        with open(triton_file, 'r') as f:
            content = f.read()

        # Check for common patterns
        has_max = 'max' in content.lower() or 'maximum' in content.lower()
        has_std = 'std' in content.lower() or 'variance' in content.lower()
        has_div = 'div' in content.lower() or '/' in content
        has_where = 'where' in content.lower() or 'select' in content.lower()

        print(f"Kernel: {os.path.basename(triton_file)}")
        print(f"  Contains max operation: {has_max}")
        print(f"  Contains std/variance: {has_std}")
        print(f"  Contains division: {has_div}")
        print(f"  Contains conditional: {has_where}")
        print()

print()
print("=" * 80)
print("INSTRUCTIONS")
print("=" * 80)
print()
print("To view all generated files:")
print(f"  ls -lah {output_dir}")
print()
print("To view a specific Triton kernel:")
print(f"  cat {output_dir}/<kernel_file>.py")
print()
print("To keep the files, copy them:")
print(f"  cp -r {output_dir} ./torch_compile_kernels/")
print()
print("Files will be deleted when this script exits unless you copy them!")
print()

# Ask if user wants to keep files
import sys
if sys.stdin.isatty():
    response = input("Keep generated files? [y/N]: ").strip().lower()
    if response == 'y':
        dest = './torch_compile_kernels'
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(output_dir, dest)
        print(f"✅ Files saved to: {dest}")
    else:
        print("Files will be cleaned up.")
else:
    # Non-interactive, keep files
    dest = './torch_compile_kernels'
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(output_dir, dest)
    print(f"✅ Files saved to: {dest}")
