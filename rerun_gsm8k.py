#!/usr/bin/env python3
"""
Re-run GSM8K with correct max_new_tokens for all models.
"""

import subprocess
from pathlib import Path

# All Qwen3 models
MODELS = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"]

STRENGTHS = [-0.25, -0.1, 0.0, 0.1, 0.25]

def find_steering_vector(model_size: str) -> Path:
    """Find the steering vector for this model."""
    model_name_clean = f"Qwen_Qwen3_{model_size}"
    vector_dir = Path(f"results/steering_vectors/corrigible_neutral_HHH/{model_name_clean}")

    # Find any layer file (use optimal from previous run)
    vector_files = list(vector_dir.glob("corrigible_neutral_HHH_layer_*.pt"))
    if not vector_files:
        raise FileNotFoundError(f"No steering vector found in {vector_dir}")

    # Use the first one (or could parse layer number to find optimal)
    return vector_files[0]


def main():
    print("Re-running GSM8K with max_new_tokens=1024 for all models")
    print("=" * 70)

    for i, model_size in enumerate(MODELS):
        print(f"\n[{i+1}/{len(MODELS)}] Running {model_size}...")

        model_name = f"Qwen/Qwen3-{model_size}"
        model_clean = f"Qwen_Qwen3_{model_size}"

        try:
            vector_path = find_steering_vector(model_size)
            print(f"  Using steering vector: {vector_path.name}")
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            continue

        output_path = f"results/capability_eval/gsm8k_corrigible_neutral_HHH_{model_clean}.csv"

        cmd = [
            "python", "generate_on_benchmarks.py",
            "--model", model_name,
            "--prompts", "data/benchmarks/gsm8k_eval.json",
            "--steering-vector", str(vector_path),
            "--strengths"] + [str(s) for s in STRENGTHS] + [
            "--temperature", "0.0",
            "--max-new-tokens", "1024",
            "--output", output_path,
        ]

        print(f"  Running: {' '.join(cmd[:5])}...")

        try:
            subprocess.run(cmd, check=True)
            print(f"  ✓ Saved to {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: {e}")
            continue

    print("\n" + "=" * 70)
    print("Done! Re-run evaluation with:")
    print("  python evaluate_all_benchmarks.py")
    print("  python analyze_capability_results.py")


if __name__ == "__main__":
    main()
