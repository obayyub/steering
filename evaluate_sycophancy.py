#!/usr/bin/env python3
"""Evaluate all sycophancy experiment results."""

from pathlib import Path
from evaluate_benchmarks import evaluate_generations_file

models = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"]

# Behavioral
print("Evaluating behavioral (sycophancy)...")
prompts_path = Path("data/benchmarks/behavioral_eval.json")
for model in models:
    input_path = Path(f"results/capability_eval/behavioral_sycophancy_Qwen_Qwen3_{model}.csv")
    output_path = Path(f"results/capability_eval/evaluated/behavioral_sycophancy_Qwen_Qwen3_{model}_evaluated.csv")

    if not input_path.exists():
        print(f"  Skipping {model}: File not found")
        continue

    print(f"  Evaluating {model}...")
    try:
        results_df = evaluate_generations_file(input_path, prompts_path, output_path)
    except Exception as e:
        print(f"    ERROR: {e}")

# TriviaQA
print("\nEvaluating TriviaQA...")
prompts_path = Path("data/benchmarks/triviaqa_eval.json")
for model in models:
    input_path = Path(f"results/capability_eval/triviaqa_sycophancy_Qwen_Qwen3_{model}.csv")
    output_path = Path(f"results/capability_eval/evaluated/triviaqa_sycophancy_Qwen_Qwen3_{model}_evaluated.csv")

    if not input_path.exists():
        print(f"  Skipping {model}: File not found")
        continue

    print(f"  Evaluating {model}...")
    try:
        results_df = evaluate_generations_file(input_path, prompts_path, output_path)
    except Exception as e:
        print(f"    ERROR: {e}")

# GSM8K
print("\nEvaluating GSM8K...")
prompts_path = Path("data/benchmarks/gsm8k_eval.json")
for model in models:
    input_path = Path(f"results/capability_eval/gsm8k_sycophancy_Qwen_Qwen3_{model}.csv")
    output_path = Path(f"results/capability_eval/evaluated/gsm8k_sycophancy_Qwen_Qwen3_{model}_evaluated.csv")

    if not input_path.exists():
        print(f"  Skipping {model}: File not found")
        continue

    print(f"  Evaluating {model}...")
    try:
        results_df = evaluate_generations_file(input_path, prompts_path, output_path)
    except Exception as e:
        print(f"    ERROR: {e}")

print("\nDone!")
