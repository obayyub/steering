#!/usr/bin/env python3
"""
Generic analysis of steering vector results.

Analyzes:
1. Steering effect on target behavior (primary category)
2. Cross-contamination (effect on other categories if present)
3. Capability preservation (TriviaQA, GSM8K, etc.)
4. Model size scaling

Usage:
    python analyze_steering_results.py --concept corrigible_neutral_HHH
    python analyze_steering_results.py --concept sycophancy --output-dir results/sycophancy_analysis
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

sns.set_theme(style="whitegrid")


def parse_filename(filename: str, concept: str) -> Optional[tuple[str, str]]:
    """
    Parse filename to extract benchmark type and model name.

    Returns:
        (benchmark_type, model_name) or None if doesn't match concept
    """
    stem = filename.replace("_evaluated", "")

    # Extract concept from filename
    if f"_{concept}_" not in stem:
        return None

    # Split on concept to get benchmark prefix and model suffix
    prefix, suffix = stem.split(f"_{concept}_", 1)
    return prefix, suffix


def load_results(evaluated_dir: Path, concept: str):
    """Load all evaluated results for a given concept."""
    capability_results = {}  # {benchmark: {model: df}}
    behavioral_results = {}  # {model: df}

    for csv_path in sorted(evaluated_dir.glob("*_evaluated.csv")):
        parsed = parse_filename(csv_path.stem, concept)
        if not parsed:
            continue

        benchmark, model = parsed
        df = pd.read_csv(csv_path)

        if benchmark == "behavioral":
            behavioral_results[model] = df
        else:
            # Capability benchmarks (triviaqa, gsm8k, etc.)
            if benchmark not in capability_results:
                capability_results[benchmark] = {}
            capability_results[benchmark][model] = df

    return capability_results, behavioral_results


def get_model_order(models: list[str]) -> list[str]:
    """Sort models by size if they follow naming convention."""
    # Try to extract size from model name (e.g., Qwen_Qwen3_1.7B -> 1.7)
    def extract_size(model: str) -> float:
        parts = model.split("_")
        for part in parts:
            if "B" in part:
                try:
                    return float(part.replace("B", ""))
                except ValueError:
                    pass
        return float('inf')  # Unknown size goes last

    return sorted(models, key=extract_size)


def get_model_label(model: str) -> str:
    """Extract clean label from model name."""
    # Try to extract just the size (e.g., Qwen_Qwen3_1.7B -> 1.7B)
    parts = model.split("_")
    for part in parts:
        if "B" in part:
            return part
    return model


def analyze_category_steering(
    behavioral_results: dict,
    category: str,
    concept: str,
    output_dir: Path
) -> pd.DataFrame:
    """Analyze steering effect on a specific behavioral category."""
    print("\n" + "="*80)
    print(f"STEERING EFFECT ON {category.upper()}")
    print("="*80)

    metrics = []

    for model, df in behavioral_results.items():
        # Filter to this category
        cat_df = df[df['category'] == category]

        if len(cat_df) == 0:
            continue

        for strength in sorted(cat_df['strength'].unique()):
            strength_df = cat_df[cat_df['strength'] == strength]

            validity = strength_df['is_valid'].mean()
            valid_only = strength_df[strength_df['is_valid'] == True]
            matching = valid_only['matches_behavior'].mean() if len(valid_only) > 0 else 0

            metrics.append({
                'model': model,
                'strength': strength,
                'validity': validity,
                'matching': matching,
            })

    if not metrics:
        print(f"No data found for category '{category}'")
        return pd.DataFrame()

    metrics_df = pd.DataFrame(metrics)

    # Auto-detect available strengths
    strengths = sorted(metrics_df['strength'].unique())
    baseline_strength = 0.0 if 0.0 in strengths else strengths[len(strengths)//2]

    # Determine extreme strengths for delta calculation
    min_strength = strengths[0]
    max_strength = strengths[-1]

    # Print summary
    model_order = get_model_order(list(behavioral_results.keys()))

    print(f"\n{'Model':<20} {'Baseline':>12} {f'Str {min_strength}':>12} {f'Str {max_strength}':>12} {'Delta':>10}")
    print("-"*70)

    for model in model_order:
        if model in behavioral_results:
            model_data = metrics_df[metrics_df['model'] == model]

            baseline_data = model_data[model_data['strength'] == baseline_strength]['matching']
            min_data = model_data[model_data['strength'] == min_strength]['matching']
            max_data = model_data[model_data['strength'] == max_strength]['matching']

            if len(baseline_data) == 0 or len(min_data) == 0 or len(max_data) == 0:
                continue

            baseline = baseline_data.values[0]
            min_val = min_data.values[0]
            max_val = max_data.values[0]
            delta = max_val - min_val

            model_label = get_model_label(model)
            print(f"{model_label:<20} {baseline:>11.1%} {min_val:>11.1%} {max_val:>11.1%} {delta:>9.1%}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    for model in model_order:
        if model in behavioral_results:
            model_data = metrics_df[metrics_df['model'] == model].sort_values('strength')
            if len(model_data) > 0:
                label = get_model_label(model)
                ax.plot(model_data['strength'], model_data['matching'] * 100,
                        marker='o', label=label, linewidth=2, markersize=8)

    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel(f'{concept} Steering Strength', fontsize=12)
    ax.set_ylabel(f'{category.capitalize()} Matching Rate (%)', fontsize=12)
    ax.set_title(f'Steering Effect: {concept} → {category}', fontsize=14, fontweight='bold')
    ax.legend(title='Model Size', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f"{category}_steering_effect.png"
    plt.savefig(output_path, dpi=150)
    print(f"\n📊 Plot saved: {output_path}")
    plt.close()

    return metrics_df


def analyze_capability_preservation(
    capability_results: dict,
    concept: str,
    output_dir: Path
) -> dict[str, pd.DataFrame]:
    """Analyze capability preservation across all capability benchmarks."""
    print("\n" + "="*80)
    print("CAPABILITY PRESERVATION")
    print("="*80)

    all_metrics = {}

    for benchmark, model_results in capability_results.items():
        print(f"\n{benchmark.upper()}:")
        print("-" * 70)

        metrics = []

        for model, df in model_results.items():
            for strength in sorted(df['strength'].unique()):
                strength_df = df[df['strength'] == strength]
                valid = strength_df[strength_df['correct'].notna()]
                accuracy = valid['correct'].mean() if len(valid) > 0 else 0

                metrics.append({
                    'model': model,
                    'strength': strength,
                    'accuracy': accuracy,
                })

        if not metrics:
            continue

        metrics_df = pd.DataFrame(metrics)
        all_metrics[benchmark] = metrics_df

        # Auto-detect strengths
        strengths = sorted(metrics_df['strength'].unique())
        baseline_strength = 0.0 if 0.0 in strengths else strengths[len(strengths)//2]
        min_strength = strengths[0]
        max_strength = strengths[-1]

        # Print summary
        model_order = get_model_order(list(model_results.keys()))

        print(f"\n{'Model':<20} {'Baseline':>12} {f'Str {min_strength}':>12} {f'Str {max_strength}':>12} {'Max Δ':>10}")
        print("-"*70)

        for model in model_order:
            model_data = metrics_df[metrics_df['model'] == model]

            baseline_data = model_data[model_data['strength'] == baseline_strength]['accuracy']
            if len(baseline_data) == 0:
                continue

            baseline = baseline_data.values[0]

            min_data = model_data[model_data['strength'] == min_strength]['accuracy']
            max_data = model_data[model_data['strength'] == max_strength]['accuracy']

            min_val = min_data.values[0] if len(min_data) > 0 else baseline
            max_val = max_data.values[0] if len(max_data) > 0 else baseline

            max_delta = max(abs(min_val - baseline), abs(max_val - baseline))

            model_label = get_model_label(model)
            print(f"{model_label:<20} {baseline:>11.1%} {min_val:>11.1%} {max_val:>11.1%} {max_delta:>9.1%}")

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))

        for model in model_order:
            model_data = metrics_df[metrics_df['model'] == model].sort_values('strength')
            if len(model_data) > 0:
                label = get_model_label(model)
                ax.plot(model_data['strength'], model_data['accuracy'] * 100,
                        marker='o', label=label, linewidth=2, markersize=8)

        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Steering Strength', fontsize=12)
        ax.set_ylabel(f'{benchmark.upper()} Accuracy (%)', fontsize=12)
        ax.set_title(f'Capability Preservation: {benchmark.upper()} Under {concept} Steering', fontsize=14, fontweight='bold')
        ax.legend(title='Model Size', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = output_dir / f"{benchmark}_capability_preservation.png"
        plt.savefig(output_path, dpi=150)
        print(f"📊 Plot saved: {output_path}")
        plt.close()

    return all_metrics


def print_summary(
    primary_category_df: pd.DataFrame,
    other_category_dfs: dict[str, pd.DataFrame],
    capability_dfs: dict[str, pd.DataFrame],
    concept: str,
    primary_category: str
):
    """Print final summary assessment."""
    print("\n" + "="*80)
    print("OVERALL ASSESSMENT")
    print("="*80)

    # Steering effect (primary category)
    if len(primary_category_df) > 0:
        avg_deltas = []
        for model in primary_category_df['model'].unique():
            model_data = primary_category_df[primary_category_df['model'] == model]
            strengths = sorted(model_data['strength'].unique())
            if len(strengths) >= 2:
                min_str = strengths[0]
                max_str = strengths[-1]

                min_val = model_data[model_data['strength'] == min_str]['matching'].values
                max_val = model_data[model_data['strength'] == max_str]['matching'].values

                if len(min_val) > 0 and len(max_val) > 0:
                    avg_deltas.append(max_val[0] - min_val[0])

        if avg_deltas:
            avg_steering_effect = sum(avg_deltas) / len(avg_deltas)

            print(f"\n1. STEERING EFFECT ({primary_category}):")
            print(f"   Average delta (max - min): {avg_steering_effect:.1%}")
            if avg_steering_effect > 0.10:
                print("   ✅ STRONG: Steering has clear effect on intended behavior (>10%)")
            elif avg_steering_effect > 0.05:
                print("   ⚠️  MODERATE: Steering has moderate effect (5-10%)")
            else:
                print("   ❌ WEAK: Steering has minimal effect (<5%)")

    # Cross-contamination (other categories)
    if other_category_dfs:
        print(f"\n2. CROSS-CONTAMINATION:")

        for other_cat, other_df in other_category_dfs.items():
            avg_contamination_list = []

            for model in other_df['model'].unique():
                model_data = other_df[other_df['model'] == model]
                strengths = sorted(model_data['strength'].unique())
                baseline_strength = 0.0 if 0.0 in strengths else strengths[len(strengths)//2]

                baseline_data = model_data[model_data['strength'] == baseline_strength]['matching']
                if len(baseline_data) == 0:
                    continue

                baseline = baseline_data.values[0]

                max_shift = 0
                for strength in strengths:
                    strength_data = model_data[model_data['strength'] == strength]['matching']
                    if len(strength_data) > 0:
                        max_shift = max(max_shift, abs(strength_data.values[0] - baseline))

                avg_contamination_list.append(max_shift)

            if avg_contamination_list:
                avg_contamination = sum(avg_contamination_list) / len(avg_contamination_list)

                print(f"   {other_cat}:")
                print(f"     Average max shift: {avg_contamination:.1%}")
                if avg_contamination < 0.05:
                    print(f"     ✅ MINIMAL: Cross-contamination is negligible (<5%)")
                elif avg_contamination < 0.10:
                    print(f"     ⚠️  MODERATE: Some cross-contamination detected (5-10%)")
                else:
                    print(f"     ❌ SIGNIFICANT: High cross-contamination (>10%)")

    # Capability preservation
    if capability_dfs:
        print(f"\n3. CAPABILITY PRESERVATION:")

        for benchmark, cap_df in capability_dfs.items():
            avg_impact_list = []

            for model in cap_df['model'].unique():
                model_data = cap_df[cap_df['model'] == model]
                strengths = sorted(model_data['strength'].unique())
                baseline_strength = 0.0 if 0.0 in strengths else strengths[len(strengths)//2]

                baseline_data = model_data[model_data['strength'] == baseline_strength]['accuracy']
                if len(baseline_data) == 0:
                    continue

                baseline = baseline_data.values[0]
                max_delta = (model_data['accuracy'] - baseline).abs().max()
                avg_impact_list.append(max_delta)

            if avg_impact_list:
                avg_cap_impact = sum(avg_impact_list) / len(avg_impact_list)

                print(f"   {benchmark}:")
                print(f"     Average max accuracy change: {avg_cap_impact:.1%}")
                if avg_cap_impact < 0.03:
                    print(f"     ✅ PRESERVED: Minimal impact on factual knowledge (<3%)")
                elif avg_cap_impact < 0.05:
                    print(f"     ⚠️  MINOR IMPACT: Small degradation (3-5%)")
                else:
                    print(f"     ❌ DEGRADED: Significant capability loss (>5%)")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze steering vector results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python analyze_steering_results.py --concept corrigible_neutral_HHH
    python analyze_steering_results.py --concept sycophancy --output-dir results/sycophancy_analysis
    python analyze_steering_results.py --concept formality --primary-category formal
        """
    )

    parser.add_argument("--concept", type=str, required=True,
        help="Concept name (e.g., corrigible_neutral_HHH, sycophancy)")
    parser.add_argument("--evaluated-dir", type=Path, default=Path("results/capability_eval/evaluated"),
        help="Directory containing evaluated CSVs")
    parser.add_argument("--output-dir", type=Path, default=None,
        help="Output directory for plots and CSVs (default: results/capability_eval/analysis)")
    parser.add_argument("--primary-category", type=str, default=None,
        help="Primary behavioral category to analyze (auto-detected if not specified)")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = Path("results/capability_eval/analysis")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading evaluation results for concept: {args.concept}")
    capability_results, behavioral_results = load_results(args.evaluated_dir, args.concept)

    print(f"\nFound {len(capability_results)} capability benchmarks")
    print(f"Found {len(behavioral_results)} models with behavioral data")

    # Auto-detect categories from behavioral data
    all_categories = set()
    for df in behavioral_results.values():
        if 'category' in df.columns:
            all_categories.update(df['category'].unique())

    # Determine primary category
    if args.primary_category:
        primary_category = args.primary_category
    elif all_categories:
        # Use first category alphabetically as primary
        primary_category = sorted(all_categories)[0]
    else:
        primary_category = None

    # Analyze behavioral categories
    category_dfs = {}
    primary_category_df = pd.DataFrame()

    if behavioral_results and all_categories:
        for category in sorted(all_categories):
            cat_df = analyze_category_steering(
                behavioral_results, category, args.concept, args.output_dir
            )
            category_dfs[category] = cat_df
            cat_df.to_csv(args.output_dir / f"{category}_steering_effect.csv", index=False)

            if category == primary_category:
                primary_category_df = cat_df

    # Separate primary from cross-contamination
    other_category_dfs = {k: v for k, v in category_dfs.items() if k != primary_category}

    # Analyze capability preservation
    capability_dfs = analyze_capability_preservation(
        capability_results, args.concept, args.output_dir
    )

    for benchmark, df in capability_dfs.items():
        df.to_csv(args.output_dir / f"{benchmark}_capability_preservation.csv", index=False)

    # Summary
    print_summary(
        primary_category_df,
        other_category_dfs,
        capability_dfs,
        args.concept,
        primary_category or "target behavior"
    )

    print(f"\n📁 All results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
