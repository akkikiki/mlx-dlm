#!/usr/bin/env python3
"""
Direct comparison: torch.compile vs Custom CUDA kernel

Benchmarks and analyzes the performance difference.
"""

import torch
import time
import numpy as np

try:
    import temperature_topnsigma_cuda
    CUDA_AVAILABLE = True
except ImportError:
    print("❌ Custom CUDA kernel not available")
    print("   Build with: python setup.py build_ext --inplace")
    CUDA_AVAILABLE = False

# ============================================================================
# Reference Implementation (Eager PyTorch)
# ============================================================================

def temperature_topnsigma_eager(logits, temperature, top_nsigma):
    """Eager PyTorch - multiple kernel launches."""
    logits = logits / temperature
    maximum = logits.max(dim=-1, keepdim=True)[0]
    std = logits.std(dim=-1, keepdim=True)
    threshold = maximum - top_nsigma * std
    logits = torch.where(logits >= threshold, logits, torch.tensor(float('-inf'), device=logits.device))
    return logits


# ============================================================================
# torch.compile Versions
# ============================================================================

def temperature_topnsigma_base(logits, temperature, top_nsigma):
    """Base function for torch.compile."""
    logits = logits / temperature
    maximum = logits.max(dim=-1, keepdim=True)[0]
    std = logits.std(dim=-1, keepdim=True)
    threshold = maximum - top_nsigma * std
    logits = torch.where(logits >= threshold, logits, torch.tensor(float('-inf'), device=logits.device))
    return logits


# Compile with different modes
if torch.__version__ >= "2.0.0":
    COMPILE_AVAILABLE = True

    # Default mode
    temperature_topnsigma_compiled = torch.compile(temperature_topnsigma_base)

    # Max autotune mode (tries multiple implementations)
    temperature_topnsigma_max_autotune = torch.compile(
        temperature_topnsigma_base,
        mode="max-autotune"
    )

    # Reduce overhead mode
    temperature_topnsigma_reduce_overhead = torch.compile(
        temperature_topnsigma_base,
        mode="reduce-overhead"
    )
else:
    COMPILE_AVAILABLE = False
    print(f"⚠️  torch.compile not available (PyTorch {torch.__version__} < 2.0)")


# ============================================================================
# Debugging: Check Why Inf Positions Don't Match
# ============================================================================

def debug_inf_mismatch():
    """Debug why CUDA kernel has different inf positions than PyTorch."""
    print("=" * 80)
    print("DEBUGGING: Inf Position Mismatch")
    print("=" * 80)
    print()

    # Small test case
    batch, seq, vocab = 2, 4, 100
    temperature = 1.5
    top_nsigma = 2.0

    torch.manual_seed(42)
    logits = torch.randn(batch, seq, vocab, device='cuda')

    # PyTorch reference
    result_pytorch = temperature_topnsigma_eager(logits.clone(), temperature, top_nsigma)

    # CUDA kernel
    if CUDA_AVAILABLE:
        result_cuda = temperature_topnsigma_cuda.temperature_topnsigma_optimized(
            logits.clone(), temperature, top_nsigma
        )

        # Compare
        pytorch_inf = torch.isinf(result_pytorch)
        cuda_inf = torch.isinf(result_cuda)

        print(f"PyTorch inf count: {pytorch_inf.sum().item()}")
        print(f"CUDA inf count: {cuda_inf.sum().item()}")
        print(f"Inf positions match: {(pytorch_inf == cuda_inf).all().item()}")
        print()

        # Find mismatches
        mismatch = pytorch_inf != cuda_inf
        if mismatch.any():
            print("Mismatches found!")
            print(f"Number of mismatches: {mismatch.sum().item()}")

            # Show first mismatch
            indices = torch.where(mismatch)
            if len(indices[0]) > 0:
                b, s, v = indices[0][0].item(), indices[1][0].item(), indices[2][0].item()
                print(f"\nFirst mismatch at [{b}, {s}, {v}]:")
                print(f"  PyTorch: {result_pytorch[b, s, v].item()}")
                print(f"  CUDA: {result_cuda[b, s, v].item()}")

                # Show intermediate values
                logits_scaled = logits[b, s, :] / temperature
                max_val = logits_scaled.max().item()
                std_val = logits_scaled.std().item()
                threshold = max_val - top_nsigma * std_val
                original_val = logits_scaled[v].item()

                print(f"\n  Max: {max_val:.6f}")
                print(f"  Std: {std_val:.6f}")
                print(f"  Threshold: {threshold:.6f}")
                print(f"  Original value: {original_val:.6f}")
                print(f"  Above threshold: {original_val >= threshold}")
        else:
            print("✅ No mismatches!")
        print()


# ============================================================================
# Benchmark Function
# ============================================================================

def benchmark_implementation(func, logits, temperature, top_nsigma, name, warmup=10, iterations=100):
    """Benchmark a single implementation."""
    # Warmup
    for _ in range(warmup):
        _ = func(logits.clone(), temperature, top_nsigma)
        torch.cuda.synchronize()

    # Benchmark with CUDA events for precision
    times = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(iterations):
        start_event.record()
        result = func(logits.clone(), temperature, top_nsigma)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    times = np.array(times)
    return {
        'name': name,
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'median': np.median(times),
        'p95': np.percentile(times, 95),
        'p99': np.percentile(times, 99),
    }


# ============================================================================
# Main Comparison
# ============================================================================

def main_comparison():
    """Run comprehensive comparison."""
    print("=" * 80)
    print("TORCH.COMPILE VS CUSTOM CUDA: HEAD TO HEAD")
    print("=" * 80)
    print()

    # Configuration
    batch_size = 8
    seq_len = 32
    vocab_size = 156896
    temperature = 1.0
    top_nsigma = 2.0

    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Vocabulary size: {vocab_size:,}")
    print(f"  Total positions: {batch_size * seq_len}")
    print(f"  Temperature: {temperature}")
    print(f"  Top-nsigma: {top_nsigma}")
    print()

    # Create input
    torch.manual_seed(42)
    logits = torch.randn(batch_size, seq_len, vocab_size, device='cuda')

    # Benchmark all implementations
    results = []

    print("Benchmarking (100 iterations each)...")
    print()

    # Eager PyTorch
    result = benchmark_implementation(
        temperature_topnsigma_eager, logits, temperature, top_nsigma,
        name="Eager PyTorch"
    )
    results.append(result)
    print(f"✓ {result['name']:35s}: {result['mean']:6.2f}ms ± {result['std']:4.2f}ms")

    # torch.compile versions
    if COMPILE_AVAILABLE:
        result = benchmark_implementation(
            temperature_topnsigma_compiled, logits, temperature, top_nsigma,
            name="torch.compile (default)"
        )
        results.append(result)
        print(f"✓ {result['name']:35s}: {result['mean']:6.2f}ms ± {result['std']:4.2f}ms")

        result = benchmark_implementation(
            temperature_topnsigma_max_autotune, logits, temperature, top_nsigma,
            name="torch.compile (max-autotune)"
        )
        results.append(result)
        print(f"✓ {result['name']:35s}: {result['mean']:6.2f}ms ± {result['std']:4.2f}ms")

        result = benchmark_implementation(
            temperature_topnsigma_reduce_overhead, logits, temperature, top_nsigma,
            name="torch.compile (reduce-overhead)"
        )
        results.append(result)
        print(f"✓ {result['name']:35s}: {result['mean']:6.2f}ms ± {result['std']:4.2f}ms")

    # Custom CUDA
    if CUDA_AVAILABLE:
        result = benchmark_implementation(
            lambda l, t, n: temperature_topnsigma_cuda.temperature_topnsigma(l, t, n),
            logits, temperature, top_nsigma,
            name="Custom CUDA (basic)"
        )
        results.append(result)
        print(f"✓ {result['name']:35s}: {result['mean']:6.2f}ms ± {result['std']:4.2f}ms")

        result = benchmark_implementation(
            lambda l, t, n: temperature_topnsigma_cuda.temperature_topnsigma_optimized(l, t, n),
            logits, temperature, top_nsigma,
            name="Custom CUDA (optimized)"
        )
        results.append(result)
        print(f"✓ {result['name']:35s}: {result['mean']:6.2f}ms ± {result['std']:4.2f}ms")

    # Summary table
    print()
    print("=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    print()
    print(f"{'Implementation':<35s} {'Mean':>8s} {'Std':>7s} {'Min':>8s} {'Median':>8s} {'P95':>8s} {'P99':>8s}")
    print("-" * 80)

    for r in results:
        print(f"{r['name']:<35s} {r['mean']:7.2f}ms {r['std']:6.2f}ms {r['min']:7.2f}ms {r['median']:7.2f}ms {r['p95']:7.2f}ms {r['p99']:7.2f}ms")

    # Speedup analysis
    print()
    print("=" * 80)
    print("SPEEDUP ANALYSIS")
    print("=" * 80)
    print()

    baseline = results[0]['mean']  # Eager PyTorch

    sorted_results = sorted(results, key=lambda x: x['mean'])
    fastest = sorted_results[0]

    print(f"Baseline (Eager PyTorch): {baseline:.2f}ms")
    print(f"Fastest ({fastest['name']}): {fastest['mean']:.2f}ms")
    print()

    print(f"{'Implementation':<35s} {'Time':>10s} {'Speedup':>12s}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: x['mean']):
        speedup = baseline / r['mean']
        speedup_str = f"{speedup:.2f}x"
        print(f"{r['name']:<35s} {r['mean']:8.2f}ms {speedup_str:>12s}")

    # Key comparisons
    print()
    print("=" * 80)
    print("KEY COMPARISONS")
    print("=" * 80)
    print()

    # Find torch.compile max-autotune
    compile_max = next((r for r in results if 'max-autotune' in r['name']), None)
    cuda_opt = next((r for r in results if 'CUDA (optimized)' in r['name']), None)

    if compile_max and cuda_opt:
        gap = cuda_opt['mean'] / compile_max['mean']
        print(f"torch.compile (max-autotune) vs Custom CUDA (optimized):")
        print(f"  torch.compile: {compile_max['mean']:.2f}ms")
        print(f"  Custom CUDA:   {cuda_opt['mean']:.2f}ms")
        print(f"  Gap: {gap:.2f}x")
        print()

        if gap < 1.1:
            print("  📊 Verdict: torch.compile is VERY CLOSE to custom CUDA!")
            print("     For most users, torch.compile is the better choice (zero effort).")
        elif gap < 1.3:
            print("  📊 Verdict: torch.compile is COMPETITIVE with custom CUDA")
            print("     Custom CUDA only worth it for critical hot paths.")
        elif gap < 2.0:
            print("  📊 Verdict: Custom CUDA has a noticeable advantage")
            print("     Worth writing custom kernel if this is a bottleneck.")
        else:
            print("  📊 Verdict: Custom CUDA is SIGNIFICANTLY faster")
            print("     Definitely worth the effort for performance-critical code.")

    # Bandwidth analysis
    print()
    print("=" * 80)
    print("BANDWIDTH ANALYSIS")
    print("=" * 80)
    print()

    total_elements = batch_size * seq_len * vocab_size
    bytes_per_element = 4  # float32

    # We read input once and write output once (minimum)
    # But we also read multiple times for reductions
    # Estimate: 1 write + 4 reads (temp, max, mean, var, filter) = 5 total
    total_bytes = total_elements * bytes_per_element * 5

    print(f"Total elements: {total_elements:,}")
    print(f"Estimated memory traffic: {total_bytes / 1e9:.2f} GB")
    print()

    for r in results:
        time_seconds = r['mean'] / 1000
        bandwidth_gbs = total_bytes / time_seconds / 1e9
        print(f"{r['name']:<35s}: {bandwidth_gbs:6.1f} GB/s")

    # Theoretical peak
    print()
    device_name = torch.cuda.get_device_name()
    print(f"GPU: {device_name}")

    # Known memory bandwidths (GB/s)
    bandwidths = {
        'RTX 4090': 1008,
        'RTX 4080': 717,
        'RTX 4000 Ada': 560,  # Professional Ada card
        'A100': 1935,
        'H100': 3350,
        'RTX 3090': 936,
        'RTX 3080': 760,
    }

    peak_bw = None
    for name, bw in bandwidths.items():
        if name in device_name:
            peak_bw = bw
            break

    if peak_bw:
        print(f"Theoretical peak memory bandwidth: {peak_bw} GB/s")
        print()

        fastest_bw = total_bytes / (fastest['mean'] / 1000) / 1e9
        efficiency = (fastest_bw / peak_bw) * 100
        print(f"Fastest implementation efficiency: {efficiency:.1f}%")
        print(f"  (Achieving {fastest_bw:.1f} / {peak_bw} GB/s)")


def main():
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  JIT vs CUSTOM CUDA: DETAILED COMPARISON".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    # Debug inf mismatch first
    if CUDA_AVAILABLE:
        debug_inf_mismatch()

    # Run main comparison
    if COMPILE_AVAILABLE or CUDA_AVAILABLE:
        main_comparison()
    else:
        print("❌ Neither torch.compile nor custom CUDA available!")

    print()
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()
    print("For 90% of users:")
    print("  ✅ Use @torch.compile - excellent performance, zero effort")
    print()
    print("For performance-critical paths:")
    print("  ⚡ Custom CUDA kernel - maximum performance, requires expertise")
    print()
    print("Your situation (LLaDA2 sampling):")
    if COMPILE_AVAILABLE:
        print("  • If sampling once per block: torch.compile is fine")
        print("  • If sampling in tight loop: custom CUDA worth it")
    print()


if __name__ == "__main__":
    main()
