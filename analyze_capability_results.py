#!/usr/bin/env python3
"""
Simple analysis of capability evaluation results.

Reads evaluated CSVs and computes accuracy by strength and model.
"""

from pathlib import Path
import pandas as pd


def analyze_results(evaluated_dir: Path = Path("results/capability_eval/evaluated")):
    """Analyze all evaluated results."""

    results = {
        "gsm8k": {},
        "triviaqa": {},
        "behavioral": {},
    }

    # Read all evaluated CSVs
    for csv_path in sorted(evaluated_dir.glob("*_evaluated.csv")):
        df = pd.read_csv(csv_path)

        # Parse filename
        stem = csv_path.stem.replace("_evaluated", "")

        if stem.startswith("gsm8k"):
            benchmark = "gsm8k"
            model = stem.replace("gsm8k_neutral_HHH_", "")
        elif stem.startswith("triviaqa"):
            benchmark = "triviaqa"
            model = stem.replace("triviaqa_neutral_HHH_", "")
        elif stem.startswith("behavioral"):
            benchmark = "behavioral"
            model = stem.replace("behavioral_neutral_HHH_", "")
        else:
            continue

        # Compute metrics by strength
        metrics_by_strength = {}
        for strength in sorted(df["strength"].unique()):
            strength_df = df[df["strength"] == strength]

            if benchmark in ["gsm8k", "triviaqa"]:
                # Capability: accuracy
                valid = strength_df[strength_df["correct"].notna()]
                accuracy = valid["correct"].mean() if len(valid) > 0 else 0
                metrics_by_strength[strength] = {"accuracy": accuracy}

            elif benchmark == "behavioral":
                # Behavioral: validity and matching rate
                validity = strength_df["is_valid"].mean()
                valid_only = strength_df[strength_df["is_valid"] == True]
                matching = valid_only["matches_behavior"].mean() if len(valid_only) > 0 else 0
                metrics_by_strength[strength] = {
                    "validity": validity,
                    "matching": matching,
                }

        results[benchmark][model] = metrics_by_strength

    return results


def print_summary(results: dict):
    """Print summary tables."""

    print("\n" + "=" * 80)
    print("CAPABILITY PRESERVATION RESULTS")
    print("=" * 80)

    # GSM8K (Math)
    print("\n" + "-" * 80)
    print("GSM8K (Grade School Math)")
    print("-" * 80)
    print(f"{'Model':<25} {'Baseline':>10} {'Str -0.25':>12} {'Str -0.10':>12} {'Str +0.10':>12} {'Str +0.25':>12}")
    print("-" * 80)

    for model in sorted(results["gsm8k"].keys()):
        metrics = results["gsm8k"][model]
        baseline = metrics.get(0.0, {}).get("accuracy", 0)
        neg_25 = metrics.get(-0.25, {}).get("accuracy", 0)
        neg_10 = metrics.get(-0.1, {}).get("accuracy", 0)
        pos_10 = metrics.get(0.1, {}).get("accuracy", 0)
        pos_25 = metrics.get(0.25, {}).get("accuracy", 0)

        print(f"{model:<25} {baseline:>9.1%} {neg_25:>11.1%} {neg_10:>11.1%} {pos_10:>11.1%} {pos_25:>11.1%}")

    # TriviaQA (Factual)
    print("\n" + "-" * 80)
    print("TriviaQA (Factual Recall)")
    print("-" * 80)
    print(f"{'Model':<25} {'Baseline':>10} {'Str -0.25':>12} {'Str -0.10':>12} {'Str +0.10':>12} {'Str +0.25':>12}")
    print("-" * 80)

    for model in sorted(results["triviaqa"].keys()):
        metrics = results["triviaqa"][model]
        baseline = metrics.get(0.0, {}).get("accuracy", 0)
        neg_25 = metrics.get(-0.25, {}).get("accuracy", 0)
        neg_10 = metrics.get(-0.1, {}).get("accuracy", 0)
        pos_10 = metrics.get(0.1, {}).get("accuracy", 0)
        pos_25 = metrics.get(0.25, {}).get("accuracy", 0)

        print(f"{model:<25} {baseline:>9.1%} {neg_25:>11.1%} {neg_10:>11.1%} {pos_10:>11.1%} {pos_25:>11.1%}")

    # Behavioral
    print("\n" + "-" * 80)
    print("Behavioral Cross-Contamination (Validity / Matching Rate)")
    print("-" * 80)
    print(f"{'Model':<25} {'Baseline':>12} {'Str -0.25':>14} {'Str +0.25':>14}")
    print("-" * 80)

    for model in sorted(results["behavioral"].keys()):
        metrics = results["behavioral"][model]

        baseline_v = metrics.get(0.0, {}).get("validity", 0)
        baseline_m = metrics.get(0.0, {}).get("matching", 0)
        neg_v = metrics.get(-0.25, {}).get("validity", 0)
        neg_m = metrics.get(-0.25, {}).get("matching", 0)
        pos_v = metrics.get(0.25, {}).get("validity", 0)
        pos_m = metrics.get(0.25, {}).get("matching", 0)

        baseline_str = f"{baseline_v:.0%}/{baseline_m:.0%}"
        neg_str = f"{neg_v:.0%}/{neg_m:.0%}"
        pos_str = f"{pos_v:.0%}/{pos_m:.0%}"

        print(f"{model:<25} {baseline_str:>12} {neg_str:>14} {pos_str:>14}")

    print("\n" + "=" * 80)


def create_csv_summary(results: dict, output_path: Path):
    """Create CSV with all results."""
    rows = []

    for benchmark in ["gsm8k", "triviaqa", "behavioral"]:
        for model, metrics_by_strength in results[benchmark].items():
            for strength, metrics in metrics_by_strength.items():
                row = {
                    "benchmark": benchmark,
                    "model": model,
                    "strength": strength,
                }
                row.update(metrics)
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nCSV summary saved to: {output_path}")


if __name__ == "__main__":
    results = analyze_results()
    print_summary(results)
    create_csv_summary(results, Path("results/capability_eval/evaluated/summary.csv"))
