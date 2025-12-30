#!/usr/bin/env python3
"""
Evaluate all benchmark generation results and aggregate metrics.

Processes all generation CSVs in capability_eval/, runs auto-evaluation,
and creates summary reports across models and benchmarks.
"""

import json
from pathlib import Path

import pandas as pd
from evaluate_benchmarks import evaluate_generations_file, compute_metrics


def find_generation_files(results_dir: Path) -> list[dict]:
    """
    Find all generation CSV files and match with their benchmark prompts.

    Returns list of dicts with:
        - generation_path
        - prompts_path
        - benchmark_name
        - model_name
    """
    files = []

    for csv_path in results_dir.glob("*.csv"):
        # Parse filename: {benchmark}_{concept}_{model}.csv
        stem = csv_path.stem

        # Determine benchmark type
        benchmark_name = None
        prompts_path = None

        if stem.startswith("gsm8k"):
            benchmark_name = "gsm8k"
            prompts_path = Path("data/benchmarks/gsm8k_eval.json")
        elif stem.startswith("triviaqa"):
            benchmark_name = "triviaqa"
            prompts_path = Path("data/benchmarks/triviaqa_eval.json")
        elif stem.startswith("behavioral"):
            benchmark_name = "behavioral"
            prompts_path = Path("data/benchmarks/behavioral_eval.json")
        else:
            print(f"Warning: Unknown benchmark type for {csv_path.name}")
            continue

        if not prompts_path.exists():
            print(f"Warning: Prompts file not found: {prompts_path}")
            continue

        # Extract model name (everything after second underscore)
        parts = stem.split("_")
        if len(parts) >= 3:
            # Join everything from the model name onward
            model_name = "_".join(parts[2:])
        else:
            model_name = "unknown"

        files.append({
            "generation_path": csv_path,
            "prompts_path": prompts_path,
            "benchmark_name": benchmark_name,
            "model_name": model_name,
        })

    return sorted(files, key=lambda x: (x["benchmark_name"], x["model_name"]))


def evaluate_all(
    results_dir: Path = Path("results/capability_eval"),
    output_dir: Path = Path("results/capability_eval/evaluated"),
):
    """
    Evaluate all generation files and create summary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all files
    files = find_generation_files(results_dir)

    if not files:
        print(f"No generation files found in {results_dir}")
        return

    print(f"Found {len(files)} generation files to evaluate")
    print("=" * 70)

    all_metrics = []

    for i, file_info in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] {file_info['benchmark_name']} - {file_info['model_name']}")

        # Evaluate
        output_path = output_dir / f"{file_info['benchmark_name']}_{file_info['model_name']}_evaluated.csv"

        try:
            results_df = evaluate_generations_file(
                file_info["generation_path"],
                file_info["prompts_path"],
                output_path,
            )

            # Compute metrics
            metrics = compute_metrics(results_df)
            metrics["benchmark"] = file_info["benchmark_name"]
            metrics["model"] = file_info["model_name"]
            all_metrics.append(metrics)

            print(f"  → Evaluated and saved to {output_path.name}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Save aggregated metrics
    metrics_path = output_dir / "all_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Evaluation complete!")
    print(f"  Individual results: {output_dir}")
    print(f"  Aggregated metrics: {metrics_path}")

    # Create summary report
    create_summary_report(all_metrics, output_dir)


def create_summary_report(all_metrics: list[dict], output_dir: Path):
    """
    Create human-readable summary of results across models and benchmarks.
    """
    print(f"\n{'=' * 70}")
    print("SUMMARY REPORT")
    print("=" * 70)

    # Group by benchmark
    benchmarks = {}
    for metrics in all_metrics:
        benchmark = metrics["benchmark"]
        model = metrics["model"]

        if benchmark not in benchmarks:
            benchmarks[benchmark] = {}
        benchmarks[benchmark][model] = metrics

    # Print summary for each benchmark
    for benchmark_name, models in sorted(benchmarks.items()):
        print(f"\n{benchmark_name.upper()}")
        print("-" * 70)

        if benchmark_name == "gsm8k":
            print_math_summary(models)
        elif benchmark_name == "triviaqa":
            print_factual_summary(models)
        elif benchmark_name == "behavioral":
            print_behavioral_summary(models)

    # Save CSV summary
    summary_rows = []
    for metrics in all_metrics:
        row = {
            "benchmark": metrics["benchmark"],
            "model": metrics["model"],
        }

        # Add relevant metrics based on benchmark
        if "overall_accuracy" in metrics:
            row["accuracy"] = metrics["overall_accuracy"]

        # Extract by-strength metrics if available
        if "by_strength" in metrics:
            for strength, strength_metrics in metrics["by_strength"].items():
                if isinstance(strength, (int, float)):
                    strength_key = f"strength_{strength:+.2f}".replace(".", "_")

                    # Get first category's accuracy/validity
                    for cat_name, cat_metrics in strength_metrics.items():
                        if cat_name not in ["overall_accuracy", "by_strength"]:
                            if "accuracy" in cat_metrics:
                                row[f"{strength_key}_accuracy"] = cat_metrics["accuracy"]
                            if "validity_rate" in cat_metrics:
                                row[f"{strength_key}_validity"] = cat_metrics["validity_rate"]
                            if "matching_rate" in cat_metrics:
                                row[f"{strength_key}_matching"] = cat_metrics["matching_rate"]
                            break

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nCSV summary saved to: {summary_path}")


def print_math_summary(models: dict):
    """Print GSM8K summary."""
    print(f"{'Model':<20} {'Baseline':>10} {'Strength -0.25':>15} {'Strength +0.25':>15}")
    print("-" * 65)

    for model_name in sorted(models.keys()):
        metrics = models[model_name]

        baseline = None
        neg_strength = None
        pos_strength = None

        if "by_strength" in metrics:
            for strength, strength_metrics in metrics["by_strength"].items():
                if abs(float(strength)) < 0.01:
                    baseline = strength_metrics.get("overall_accuracy")
                elif abs(float(strength) + 0.25) < 0.01:
                    neg_strength = strength_metrics.get("overall_accuracy")
                elif abs(float(strength) - 0.25) < 0.01:
                    pos_strength = strength_metrics.get("overall_accuracy")

        baseline_str = f"{baseline:.1%}" if baseline is not None else "N/A"
        neg_str = f"{neg_strength:.1%}" if neg_strength is not None else "N/A"
        pos_str = f"{pos_strength:.1%}" if pos_strength is not None else "N/A"

        print(f"{model_name:<20} {baseline_str:>10} {neg_str:>15} {pos_str:>15}")


def print_factual_summary(models: dict):
    """Print TriviaQA summary."""
    print(f"{'Model':<20} {'Baseline':>10} {'Strength -0.25':>15} {'Strength +0.25':>15}")
    print("-" * 65)

    for model_name in sorted(models.keys()):
        metrics = models[model_name]

        baseline = None
        neg_strength = None
        pos_strength = None

        if "by_strength" in metrics:
            for strength, strength_metrics in metrics["by_strength"].items():
                if abs(float(strength)) < 0.01:
                    baseline = strength_metrics.get("overall_accuracy")
                elif abs(float(strength) + 0.25) < 0.01:
                    neg_strength = strength_metrics.get("overall_accuracy")
                elif abs(float(strength) - 0.25) < 0.01:
                    pos_strength = strength_metrics.get("overall_accuracy")

        baseline_str = f"{baseline:.1%}" if baseline is not None else "N/A"
        neg_str = f"{neg_strength:.1%}" if neg_strength is not None else "N/A"
        pos_str = f"{pos_strength:.1%}" if pos_strength is not None else "N/A"

        print(f"{model_name:<20} {baseline_str:>10} {neg_str:>15} {pos_str:>15}")


def print_behavioral_summary(models: dict):
    """Print behavioral (sycophancy/corrigible) summary."""
    print(f"{'Model':<20} {'Validity':>10} {'Baseline Match':>15} {'Neg Steering':>15} {'Pos Steering':>15}")
    print("-" * 80)

    for model_name in sorted(models.keys()):
        metrics = models[model_name]

        # Get first category metrics (corrigible, sycophancy, etc)
        validity = None
        baseline = None
        neg_strength = None
        pos_strength = None

        if "by_strength" in metrics:
            for strength, strength_metrics in metrics["by_strength"].items():
                # Get first non-meta category
                for cat_name, cat_metrics in strength_metrics.items():
                    if cat_name not in ["overall_accuracy", "by_strength"]:
                        if abs(float(strength)) < 0.01:
                            validity = cat_metrics.get("validity_rate")
                            baseline = cat_metrics.get("matching_rate")
                        elif abs(float(strength) + 0.25) < 0.01:
                            neg_strength = cat_metrics.get("matching_rate")
                        elif abs(float(strength) - 0.25) < 0.01:
                            pos_strength = cat_metrics.get("matching_rate")
                        break

        validity_str = f"{validity:.1%}" if validity is not None else "N/A"
        baseline_str = f"{baseline:.1%}" if baseline is not None else "N/A"
        neg_str = f"{neg_strength:.1%}" if neg_strength is not None else "N/A"
        pos_str = f"{pos_strength:.1%}" if pos_strength is not None else "N/A"

        print(f"{model_name:<20} {validity_str:>10} {baseline_str:>15} {neg_str:>15} {pos_str:>15}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate all benchmark generation results"
    )
    parser.add_argument(
        "--results-dir", type=str, default="results/capability_eval",
        help="Directory containing generation CSVs"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/capability_eval/evaluated",
        help="Directory to save evaluation results"
    )

    args = parser.parse_args()

    evaluate_all(
        results_dir=Path(args.results_dir),
        output_dir=Path(args.output_dir),
    )
