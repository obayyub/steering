#!/usr/bin/env python3
"""
Run full steering experiment across multiple models.

Workflow for each model:
1. Find optimal layer (quick eval across all layers past first third)
2. Run full experiment with optimal layer
3. Aggregate results

Usage:
    # All Qwen3 models
    python run_full_experiment.py --concept sycophancy

    # Specific models
    python run_full_experiment.py --concept sycophancy --models 0.6B 1.7B 4B

    # Skip layer search (use middle third)
    python run_full_experiment.py --concept sycophancy --skip-layer-search
"""

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

from find_optimal_layer import find_optimal_layer, QWEN3_MODELS, clear_memory, get_gpu_memory
from run_eval_experiment import run_experiment


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


def run_full_experiment(
    concept: str,
    models: list[str],
    data_dir: Path,
    output_base: Path,
    skip_layer_search: bool = False,
    layer_search_prompts: int = 20,
    full_eval_prompts: int = 100,
    num_repeats: int = 3,
    dtype: str = "bfloat16",
):
    """
    Run complete experiment across multiple models.

    For each model:
    1. Find optimal layer (unless skip_layer_search=True)
    2. Run full evaluation with optimal/default layer
    """
    # Resolve data paths
    train_data = data_dir / f"{concept}_combined_contrastive_pairs.json"
    eval_data = data_dir / f"{concept}_combined_eval_prompts.json"

    if not train_data.exists():
        # Try without _combined suffix
        train_data = data_dir / f"{concept}_contrastive_pairs.json"
        eval_data = data_dir / f"{concept}_eval_prompts.json"

    if not train_data.exists():
        raise FileNotFoundError(f"Training data not found in {data_dir}")

    print("\n" + "="*70)
    print("  FULL STEERING EXPERIMENT")
    print("="*70)
    print(f"  Concept:     {concept}")
    print(f"  Models:      {', '.join(models)}")
    print(f"  Train data:  {train_data}")
    print(f"  Eval data:   {eval_data}")
    print(f"  Output:      {output_base}")
    print(f"  Layer search: {'Skip (use middle third)' if skip_layer_search else f'Yes ({layer_search_prompts} prompts)'}")
    print(f"  Full eval:   {full_eval_prompts} prompts x {num_repeats} repeats")
    print(f"  GPU:         {get_gpu_memory()}")
    print("="*70)

    experiment_start = time.time()
    all_results = []
    model_times = {}

    for i, model_size in enumerate(models):
        model_start = time.time()
        model_name = QWEN3_MODELS.get(model_size, model_size)
        model_name_clean = model_name.replace("/", "_").replace("-", "_")

        print(f"\n{'='*70}")
        print(f"  MODEL {i+1}/{len(models)}: {model_size}")
        print(f"{'='*70}")

        optimal_layer = None

        # Step 1: Find optimal layer
        if not skip_layer_search:
            print(f"\n[1/2] Finding optimal layer...")
            try:
                layer_result = find_optimal_layer(
                    model_name=model_name,
                    concept=concept,
                    train_data_path=train_data,
                    eval_data_path=eval_data,
                    output_base=output_base,
                    max_eval_prompts=layer_search_prompts,
                    dtype=dtype,
                )
                optimal_layer = layer_result.get("optimal_layer")
                print(f"  -> Optimal layer: {optimal_layer}")
            except Exception as e:
                print(f"  -> Layer search failed: {e}")
                import traceback
                traceback.print_exc()

            clear_memory()

        # Step 2: Run full experiment
        print(f"\n[2/2] Running full evaluation...")
        try:
            result = run_experiment(
                model_name=model_name,
                concept=concept,
                train_data_path=train_data,
                eval_data_path=eval_data,
                output_base=output_base,
                layer=optimal_layer,
                num_repeats=num_repeats,
                max_eval_prompts=full_eval_prompts,
                dtype=dtype,
            )
            result["optimal_layer"] = optimal_layer
            result["model_size"] = model_size
            all_results.append(result)

            print(f"\n  Results for {model_size}:")
            print(f"    Layer:    {result['layer_start']}")
            print(f"    Baseline: {result['baseline_rate']:.1%}")
            print(f"    Max:      {result['max_rate']:.1%}")
            print(f"    Delta:    {result['delta']:.1%}")

        except Exception as e:
            print(f"  -> Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "model": model_name,
                "model_size": model_size,
                "error": str(e),
            })

        clear_memory()

        # Timing
        model_time = time.time() - model_start
        model_times[model_size] = model_time
        remaining = len(models) - (i + 1)

        if remaining > 0:
            avg_time = sum(model_times.values()) / len(model_times)
            eta = avg_time * remaining
            print(f"\n  Time: {format_duration(model_time)} | ETA: {format_duration(eta)}")

    # Final summary
    total_time = time.time() - experiment_start

    print("\n" + "="*70)
    print("  EXPERIMENT COMPLETE")
    print("="*70)
    print(f"\n  Total time: {format_duration(total_time)}")
    print(f"\n  Results by model:")
    print(f"  {'Model':<8} {'Layer':>6} {'Baseline':>10} {'Max':>8} {'Delta':>8}")
    print(f"  {'-'*44}")

    for r in all_results:
        if "error" in r:
            print(f"  {r['model_size']:<8} ERROR: {r['error'][:30]}")
        else:
            layer = r.get('layer_start', '?')
            baseline = r.get('baseline_rate', 0)
            max_rate = r.get('max_rate', 0)
            delta = r.get('delta', 0)
            print(f"  {r['model_size']:<8} {layer:>6} {baseline:>10.1%} {max_rate:>8.1%} {delta:>8.1%}")

    # Save combined results
    summary_dir = output_base / "summaries" / concept
    summary_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = summary_dir / f"experiment_summary_{timestamp}.json"

    summary = {
        "timestamp": datetime.now().isoformat(),
        "concept": concept,
        "models": models,
        "train_data": str(train_data),
        "eval_data": str(eval_data),
        "skip_layer_search": skip_layer_search,
        "total_time_seconds": total_time,
        "model_times": model_times,
        "results": all_results,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary saved: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run full steering experiment across models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run sycophancy experiment on all models
    python run_full_experiment.py --concept sycophancy

    # Run on specific models only
    python run_full_experiment.py --concept sycophancy --models 0.6B 1.7B 4B

    # Skip layer search (faster, uses middle third)
    python run_full_experiment.py --concept sycophancy --skip-layer-search

    # Quick test run
    python run_full_experiment.py --concept sycophancy --models 0.6B \\
        --layer-search-prompts 10 --full-eval-prompts 20 --num-repeats 1
        """
    )

    parser.add_argument("--concept", type=str, required=True,
        help="Concept name (e.g., sycophancy). Looks for data in data/<concept>/")
    parser.add_argument("--models", nargs="+", default=list(QWEN3_MODELS.keys()),
        choices=list(QWEN3_MODELS.keys()),
        help="Models to run (default: all)")
    parser.add_argument("--data-dir", type=str, default=None,
        help="Data directory (default: data/<concept>/)")
    parser.add_argument("--output-base", type=str, default="results")

    parser.add_argument("--skip-layer-search", action="store_true",
        help="Skip layer search, use middle third of layers")
    parser.add_argument("--layer-search-prompts", type=int, default=20,
        help="Number of prompts for layer search (default: 20)")
    parser.add_argument("--full-eval-prompts", type=int, default=100,
        help="Number of prompts for full evaluation (default: 100)")
    parser.add_argument("--num-repeats", type=int, default=3,
        help="Generations per prompt (default: 3)")

    parser.add_argument("--dtype", type=str, default="bfloat16",
        choices=["float32", "float16", "bfloat16"])

    args = parser.parse_args()

    # Resolve data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path("data") / args.concept

    run_full_experiment(
        concept=args.concept,
        models=args.models,
        data_dir=data_dir,
        output_base=Path(args.output_base),
        skip_layer_search=args.skip_layer_search,
        layer_search_prompts=args.layer_search_prompts,
        full_eval_prompts=args.full_eval_prompts,
        num_repeats=args.num_repeats,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
