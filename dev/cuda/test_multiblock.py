#!/usr/bin/env python3
"""
Test different numbers of positions to find when it breaks.
"""

import torch
import temperature_topnsigma_cuda

def test_different_batch_sizes():
    """Test with increasing batch*seq sizes."""
    print("=" * 80)
    print("TESTING DIFFERENT NUMBER OF POSITIONS")
    print("=" * 80)
    print()

    vocab = 156896
    temperature = 1.0
    top_nsigma = 2.0

    configs = [
        (1, 1),    # 1 position
        (1, 2),    # 2 positions
        (1, 4),    # 4 positions
        (1, 8),    # 8 positions
        (1, 16),   # 16 positions
        (1, 32),   # 32 positions
        (2, 32),   # 64 positions
        (4, 32),   # 128 positions
        (8, 32),   # 256 positions (original failing case)
    ]

    for batch, seq in configs:
        num_positions = batch * seq

        torch.manual_seed(42)
        logits = torch.randn(batch, seq, vocab, device='cuda')

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
        ref_total = torch.isinf(ref).sum().item()
        cuda_total = torch.isinf(cuda).sum().item()
        match = (torch.isinf(ref) == torch.isinf(cuda)).all().item()

        # Per-position check
        ref_per_pos = torch.isinf(ref).view(num_positions, vocab).sum(dim=1)
        cuda_per_pos = torch.isinf(cuda).view(num_positions, vocab).sum(dim=1)
        per_pos_match = (ref_per_pos == cuda_per_pos).all().item()

        status = "✅" if match else "❌"
        print(f"{status} ({batch:>2d}, {seq:>2d}) = {num_positions:>3d} positions: "
              f"total match={match}, per-position match={per_pos_match}")

        if not match:
            # Show which positions differ
            diff_positions = torch.where(ref_per_pos != cuda_per_pos)[0]
            if len(diff_positions) > 0:
                print(f"   Positions with mismatches: {diff_positions.tolist()[:5]}...")
                # Show first mismatch
                pos = diff_positions[0].item()
                print(f"   Position {pos}: PyTorch={ref_per_pos[pos].item()}, "
                      f"CUDA={cuda_per_pos[pos].item()}")

    print()


def test_specific_failing_case():
    """Look at the (8, 32) case in detail."""
    print("=" * 80)
    print("DETAILED LOOK AT (8, 32) CASE")
    print("=" * 80)
    print()

    batch, seq, vocab = 8, 32, 156896
    temperature = 1.0
    top_nsigma = 2.0

    torch.manual_seed(42)
    logits = torch.randn(batch, seq, vocab, device='cuda')

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

    # Compare per position
    ref_per_pos = torch.isinf(ref).view(256, vocab).sum(dim=1)
    cuda_per_pos = torch.isinf(cuda).view(256, vocab).sum(dim=1)

    mismatches = torch.where(ref_per_pos != cuda_per_pos)[0]

    print(f"Total positions: 256")
    print(f"Positions with correct count: {(ref_per_pos == cuda_per_pos).sum().item()}")
    print(f"Positions with wrong count: {len(mismatches)}")
    print()

    if len(mismatches) > 0:
        print(f"First 10 mismatching positions:")
        for i, pos in enumerate(mismatches[:10]):
            pos = pos.item()
            print(f"  Position {pos:>3d}: PyTorch={ref_per_pos[pos].item():>6d}, "
                  f"CUDA={cuda_per_pos[pos].item():>6d}, "
                  f"diff={int(ref_per_pos[pos].item() - cuda_per_pos[pos].item()):>6d}")
    else:
        print("✅ All positions match!")

    print()


def main():
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  MULTI-BLOCK BUG INVESTIGATION".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    test_different_batch_sizes()
    test_specific_failing_case()

    print("=" * 80)
    print("HYPOTHESIS")
    print("=" * 80)
    print()
    print("If it works for small number of positions but fails for many:")
    print("  → Blocks are interfering with each other")
    print("  → Possible race condition or memory corruption")
    print("  → Each block should be completely independent!")
    print()


if __name__ == "__main__":
    main()
