#!/usr/bin/env python3
"""
Detailed comparison to see exactly where PyTorch and CUDA diverge.
"""

import torch
import temperature_topnsigma_cuda

def detailed_compare():
    """Compare step by step."""
    print("=" * 80)
    print("DETAILED STEP-BY-STEP COMPARISON")
    print("=" * 80)
    print()

    # Medium size for easier debugging
    batch, seq, vocab = 2, 2, 1000
    temperature = 1.5
    top_nsigma = 2.0

    torch.manual_seed(42)
    logits = torch.randn(batch, seq, vocab, device='cuda')

    print(f"Input: ({batch}, {seq}, {vocab})")
    print()

    # PyTorch step by step
    print("PyTorch computation:")
    print("-" * 40)

    scaled = logits / temperature
    print(f"1. After scaling: min={scaled.min().item():.4f}, max={scaled.max().item():.4f}")

    maximum = scaled.max(dim=-1, keepdim=True)[0]
    print(f"2. Maximum: {maximum[0, 0, 0].item():.6f}")

    std = scaled.std(dim=-1, keepdim=True)
    print(f"3. Std: {std[0, 0, 0].item():.6f}")

    threshold = maximum - top_nsigma * std
    print(f"4. Threshold: {threshold[0, 0, 0].item():.6f}")

    result_pytorch = torch.where(scaled >= threshold, scaled, torch.tensor(float('-inf'), device='cuda'))

    inf_count = torch.isinf(result_pytorch[0, 0]).sum().item()
    print(f"5. Infs in [0,0]: {inf_count}")

    # Check specific values
    above_thresh = (scaled[0, 0] >= threshold[0, 0, 0]).sum().item()
    below_thresh = (scaled[0, 0] < threshold[0, 0, 0]).sum().item()
    print(f"6. Values >= threshold: {above_thresh}")
    print(f"   Values < threshold: {below_thresh}")

    print()

    # CUDA kernel
    print("CUDA kernel computation:")
    print("-" * 40)

    result_cuda = temperature_topnsigma_cuda.temperature_topnsigma(
        logits.clone(), temperature, top_nsigma
    )

    cuda_inf_count = torch.isinf(result_cuda[0, 0]).sum().item()
    print(f"Infs in [0,0]: {cuda_inf_count}")

    # Check if any values are different
    pytorch_infs = torch.isinf(result_pytorch[0, 0])
    cuda_infs = torch.isinf(result_cuda[0, 0])

    only_pytorch = pytorch_infs & ~cuda_infs
    only_cuda = cuda_infs & ~pytorch_infs

    print(f"Infs only in PyTorch: {only_pytorch.sum().item()}")
    print(f"Infs only in CUDA: {only_cuda.sum().item()}")

    if only_pytorch.any():
        idx = torch.where(only_pytorch)[0][0].item()
        print(f"\nExample (should be -inf but isn't):")
        print(f"  Index: {idx}")
        print(f"  PyTorch: {result_pytorch[0, 0, idx].item()}")
        print(f"  CUDA: {result_cuda[0, 0, idx].item()}")
        print(f"  Original (scaled): {scaled[0, 0, idx].item():.6f}")
        print(f"  Threshold: {threshold[0, 0, 0].item():.6f}")
        print(f"  Should be inf: {scaled[0, 0, idx].item() < threshold[0, 0, 0].item()}")

    if only_cuda.any():
        idx = torch.where(only_cuda)[0][0].item()
        print(f"\nExample (shouldn't be -inf but is):")
        print(f"  Index: {idx}")
        print(f"  PyTorch: {result_pytorch[0, 0, idx].item()}")
        print(f"  CUDA: {result_cuda[0, 0, idx].item()}")
        print(f"  Original (scaled): {scaled[0, 0, idx].item():.6f}")
        print(f"  Threshold: {threshold[0, 0, 0].item():.6f}")

    print()

    # Overall match
    print("=" * 80)
    print("OVERALL COMPARISON")
    print("=" * 80)
    print()
    print(f"PyTorch total infs: {torch.isinf(result_pytorch).sum().item()}")
    print(f"CUDA total infs: {torch.isinf(result_cuda).sum().item()}")
    print(f"Match: {(pytorch_infs == cuda_infs).all().item()}")
    print()


def test_multiple_sizes():
    """Test different vocabulary sizes to find the breaking point."""
    print("=" * 80)
    print("TESTING MULTIPLE VOCABULARY SIZES")
    print("=" * 80)
    print()

    sizes = [100, 500, 1000, 5000, 10000, 50000, 100000, 156896]
    temperature = 1.5
    top_nsigma = 2.0

    for vocab in sizes:
        torch.manual_seed(42)
        logits = torch.randn(1, 1, vocab, device='cuda')

        # PyTorch
        scaled = logits / temperature
        maximum = scaled.max(dim=-1, keepdim=True)[0]
        std = scaled.std(dim=-1, keepdim=True)
        threshold = maximum - top_nsigma * std
        ref = torch.where(scaled >= threshold, scaled, torch.tensor(float('-inf'), device='cuda'))

        # CUDA
        cuda = temperature_topnsigma_cuda.temperature_topnsigma(
            logits.clone(), temperature, top_nsigma
        )

        # Compare
        ref_infs = torch.isinf(ref).sum().item()
        cuda_infs = torch.isinf(cuda).sum().item()
        match = (torch.isinf(ref) == torch.isinf(cuda)).all().item()

        status = "✅" if match else "❌"
        print(f"{status} vocab={vocab:>6d}: PyTorch={ref_infs:>4d} infs, CUDA={cuda_infs:>4d} infs, match={match}")

    print()


def main():
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  DETAILED CUDA vs PYTORCH COMPARISON".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    detailed_compare()
    test_multiple_sizes()

    print()
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("Look at the output above to see:")
    print("  1. At what vocabulary size does it start failing?")
    print("  2. Are too many or too few values being set to -inf?")
    print("  3. What are the specific values that differ?")
    print()


if __name__ == "__main__":
    main()
