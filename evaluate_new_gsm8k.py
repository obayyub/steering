#!/usr/bin/env python3
"""Evaluate new GSM8K results with 1024 tokens."""

from pathlib import Path
from evaluate_benchmarks import evaluate_generations_file

models = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"]
prompts_path = Path("data/benchmarks/gsm8k_eval.json")

for model in models:
    input_path = Path(f"results/capability_eval/gsm8k_corrigible_neutral_HHH_Qwen_Qwen3_{model}.csv")
    output_path = Path(f"results/capability_eval/evaluated/gsm8k_corrigible_neutral_HHH_Qwen_Qwen3_{model}_evaluated.csv")

    if not input_path.exists():
        print(f"Skipping {model}: File not found")
        continue

    print(f"\nEvaluating {model}...")
    try:
        results_df = evaluate_generations_file(input_path, prompts_path, output_path)

        # Quick stats
        valid = results_df[results_df['correct'].notna()]
        if len(valid) > 0:
            baseline = valid[valid['strength'] == 0.0]
            if len(baseline) > 0:
                baseline_acc = baseline['correct'].mean()
                print(f"  Baseline accuracy: {baseline_acc:.1%}")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\nDone!")
