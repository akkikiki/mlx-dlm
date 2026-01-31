#!/usr/bin/env python3
"""
Compare results between 4-bit and 8-bit models.
"""

import json
import glob
from collections import defaultdict

print("="*80)
print("4-bit vs 8-bit Model Comparison")
print("="*80)

# Load 4-bit results
results_4bit = defaultdict(dict)
for filename in glob.glob("results/gsm8k_temp*_nsigma*.json"):
    if "8bit" not in filename and "samples" not in filename:
        with open(filename) as f:
            data = json.load(f)
            if 'error' not in data:
                temp = data['config']['temperature']
                nsigma = data['config']['top_nsigma']
                score = data['results'].get('exact_match,strict-match', 0.0)
                results_4bit[(temp, nsigma)] = {
                    'score': score,
                    'time': data['time_per_sample_seconds']
                }

# Load 8-bit results
results_8bit = defaultdict(dict)
for filename in glob.glob("results/gsm8k_8bit_temp*_nsigma*.json"):
    if "samples" not in filename:
        with open(filename) as f:
            data = json.load(f)
            if 'error' not in data:
                temp = data['config']['temperature']
                nsigma = data['config']['top_nsigma']
                score = data['results'].get('exact_match,strict-match', 0.0)
                results_8bit[(temp, nsigma)] = {
                    'score': score,
                    'time': data['time_per_sample_seconds']
                }

# Find common configs
common_configs = set(results_4bit.keys()) & set(results_8bit.keys())

if not common_configs:
    print("\nNo common configurations found between 4-bit and 8-bit results.")
    print("Run both evaluations first.")
    exit(0)

print(f"\nFound {len(common_configs)} common configurations\n")

# Print comparison table
print(f"{'Config':<20} {'4-bit Score':<15} {'8-bit Score':<15} {'Diff':<10} {'4-bit Time':<12} {'8-bit Time':<12}")
print("-" * 100)

comparisons = []
for config in sorted(common_configs):
    temp, nsigma = config
    score_4bit = results_4bit[config]['score']
    score_8bit = results_8bit[config]['score']
    time_4bit = results_4bit[config]['time']
    time_8bit = results_8bit[config]['time']

    diff = score_8bit - score_4bit

    comparisons.append({
        'config': config,
        'score_4bit': score_4bit,
        'score_8bit': score_8bit,
        'diff': diff,
        'time_4bit': time_4bit,
        'time_8bit': time_8bit,
    })

    config_str = f"T={temp}, N={nsigma}"
    print(f"{config_str:<20} {score_4bit:<15.4f} {score_8bit:<15.4f} {diff:+.4f}    "
          f"{time_4bit:<12.2f} {time_8bit:<12.2f}")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

avg_score_4bit = sum(c['score_4bit'] for c in comparisons) / len(comparisons)
avg_score_8bit = sum(c['score_8bit'] for c in comparisons) / len(comparisons)
avg_time_4bit = sum(c['time_4bit'] for c in comparisons) / len(comparisons)
avg_time_8bit = sum(c['time_8bit'] for c in comparisons) / len(comparisons)

print(f"\nAverage Scores:")
print(f"  4-bit: {avg_score_4bit:.4f}")
print(f"  8-bit: {avg_score_8bit:.4f}")
print(f"  Difference: {avg_score_8bit - avg_score_4bit:+.4f}")

print(f"\nAverage Time per Sample:")
print(f"  4-bit: {avg_time_4bit:.2f}s")
print(f"  8-bit: {avg_time_8bit:.2f}s")
print(f"  Speedup: {avg_time_4bit/avg_time_8bit:.2f}x" if avg_time_8bit > 0 else "")

# Best configs for each
best_4bit = max(comparisons, key=lambda x: x['score_4bit'])
best_8bit = max(comparisons, key=lambda x: x['score_8bit'])

print(f"\nBest 4-bit Config:")
print(f"  T={best_4bit['config'][0]}, N={best_4bit['config'][1]}: {best_4bit['score_4bit']:.4f}")

print(f"\nBest 8-bit Config:")
print(f"  T={best_8bit['config'][0]}, N={best_8bit['config'][1]}: {best_8bit['score_8bit']:.4f}")

print("="*80)
