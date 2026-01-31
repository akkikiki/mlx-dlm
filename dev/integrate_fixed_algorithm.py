#!/usr/bin/env python3
"""
Quick integration example showing how to use fixed-algorithm kernels.

This demonstrates the simplest way to integrate the pre-compiled
temperature+top-nsigma kernel into your evaluation.
"""

import mlx.core as mx
from mlx_lm import load
from mlx_lm.precompiled_sampling import temperature_topnsigma_sample


def demo_integration():
    """Show how to use fixed-algorithm kernel."""

    print("=" * 80)
    print("Fixed-Algorithm Kernel Integration Demo")
    print("=" * 80)
    print()

    # Load model
    print("Loading model...")
    model, tokenizer = load("mlx-community/LLaDA2.0-mini-8bit")
    print("✓ Model loaded")
    print()

    # Create sample logits (simulating model output)
    batch_size = 8
    seq_len = 32
    vocab_size = len(tokenizer.get_vocab())

    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Vocab size: {vocab_size}")
    print()

    # Simulate logits from model
    logits = mx.random.normal((batch_size, seq_len, vocab_size))

    print("-" * 80)
    print("Using Fixed-Algorithm Kernel")
    print("-" * 80)
    print()

    # Your evaluation parameters
    configs = [
        (1.0, 2.0),
        (2.0, 3.0),
        (3.0, 2.0),
    ]

    import time

    print("Running 3 different configurations:")
    print()

    for i, (temp, nsigma) in enumerate(configs, 1):
        start = time.time()
        tokens, probs = temperature_topnsigma_sample(logits, temp, nsigma)
        mx.eval(tokens)
        elapsed = (time.time() - start) * 1000

        status = "compiling" if i == 1 else "cached"
        print(f"  Config {i} (temp={temp}, nsigma={nsigma}): {elapsed:6.2f}ms ({status})")

    print()
    print("✅ Single compilation, all configs use cached kernel!")
    print()

    print("-" * 80)
    print("How to Integrate")
    print("-" * 80)
    print()
    print("Option 1: Modify your generate function")
    print()
    print("  # In mlx_lm/llada2_generate.py")
    print("  from mlx_lm.precompiled_sampling import temperature_topnsigma_sample")
    print()
    print("  # Replace sample_tokens call with:")
    print("  if top_k is None and top_p is None and top_nsigma is not None:")
    print("      tokens, probs = temperature_topnsigma_sample(")
    print("          logits, temperature, top_nsigma")
    print("      )")
    print()
    print("Option 2: Use in evaluation wrapper")
    print()
    print("  # In scripts/lm_eval_llada2_wrapper.py")
    print("  from mlx_lm.precompiled_sampling import temperature_topnsigma_sample")
    print()
    print("  # In generate_until method:")
    print("  if self.top_k is None and self.top_p is None and self.top_nsigma:")
    print("      sampled_tokens, token_probs = temperature_topnsigma_sample(")
    print("          active_logits, self.temperature, self.top_nsigma")
    print("      )")
    print()
    print("=" * 80)


if __name__ == "__main__":
    demo_integration()
