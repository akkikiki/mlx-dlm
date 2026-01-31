#!/usr/bin/env python3
"""
Verify that chat template is being applied by running a tiny evaluation.
"""

import sys
import json
sys.path.insert(0, '/Users/yoshinari/work/mlx-lm/scripts')

from lm_eval_llada2_wrapper import LLaDA2Wrapper
from lm_eval import simple_evaluate

print("="*80)
print("Running 2-sample GSM8K test to verify chat template")
print("="*80)
print("\nThis will show if the wrapper applies chat template formatting.\n")

results = simple_evaluate(
    model="llada2",
    model_args="pretrained=mlx-community/LLaDA2.0-mini-8bit,batch_size=2,use_cache=True,temperature=1.0,top_nsigma=2.0",
    tasks=["gsm8k"],
    num_fewshot=3,
    limit=2,  # Just 2 samples
    log_samples=True,
)

samples = results.get('samples', {}).get('gsm8k', [])

print("\n" + "="*80)
print("RESULTS")
print("="*80)

if samples and len(samples) > 0:
    print(f"\n✅ Generated {len(samples)} samples")

    # Check the first sample
    sample = samples[0]

    print("\nWhat lm-eval SENT (arguments field):")
    print("-"*80)
    if 'arguments' in sample and len(sample['arguments']) > 0:
        original_prompt = sample['arguments'][0][0]
        print(original_prompt[:300] + "...")
        has_chat_template = '<role>' in original_prompt
        print(f"\nContains chat markers? {'YES ✅' if has_chat_template else 'NO ❌'}")

    print("\n" + "="*80)
    print("EXPLANATION")
    print("="*80)
    print("""
The 'arguments' field shows what lm-eval-harness SENDS to the wrapper.
The wrapper then applies chat template INTERNALLY before tokenizing.

To verify chat template is working, look for this message during evaluation:
  [INFO] Chat template being applied to prompts

This confirms the wrapper is transforming plain text → chat template format.
    """)

    # Save samples
    output_file = "results/verify_chat_template_samples.json"
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)
    print(f"Samples saved to: {output_file}\n")

else:
    print("❌ No samples generated")
