#!/usr/bin/env python3
"""
Run complete benchmark evaluation experiment across models.

Generates on:
1. Capability benchmarks (GSM8K, TriviaQA, IFEval)
2. Behavioral cross-contamination (sycophancy, corrigible)
3. Open-ended prompts (for Opus confounder discovery)

Usage:
    # Single model
    python run_benchmark_experiment.py --model 1.7B --concept corrigible_neutral_HHH

    # All models
    python run_benchmark_experiment.py --models 0.6B 1.7B 4B 8B --concept corrigible_neutral_HHH
"""

import argparse
import time
from pathlib import Path

from find_optimal_layer import QWEN3_MODELS, clear_memory, get_gpu_memory
from generate_on_benchmarks import run_benchmark_generation


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


def run_full_benchmark_experiment(
    models: list[str],
    concept: str,
    layer: int = None,
    strengths: list[float] = None,
    num_repeats: int = 3,
    output_base: Path = Path("results"),
    dtype: str = "bfloat16",
):
    """
    Run benchmark generation across multiple models.

    Args:
        models: List of model sizes (e.g., ["0.6B", "1.7B"])
        concept: Concept name for steering vector (e.g., "corrigible_neutral_HHH")
        layer: Optional layer override (uses optimal layer from metadata if None)
        strengths: Steering strengths to test
        num_repeats: Generations per (prompt, strength) pair
        output_base: Base output directory
        dtype: Model dtype
    """
    if strengths is None:
        strengths = [-0.25, -0.1, 0.0, 0.1, 0.25]

    # Benchmark configurations
    benchmarks = [
        {
            "name": "gsm8k",
            "prompts": "data/benchmarks/gsm8k_eval.json",
            "max_new_tokens": 128,
        },
        {
            "name": "triviaqa",
            "prompts": "data/benchmarks/triviaqa_eval.json",
            "max_new_tokens": 64,
        },
        {
            "name": "behavioral",
            "prompts": "data/benchmarks/behavioral_eval.json",
            "max_new_tokens": 32,
        },
        {
            "name": "open_ended",
            "prompts": "data/open_ended_prompts.json",
            "max_new_tokens": 256,
        },
    ]

    print("\n" + "=" * 70)
    print("  BENCHMARK EVALUATION EXPERIMENT")
    print("=" * 70)
    print(f"  Concept:     {concept}")
    print(f"  Models:      {', '.join(models)}")
    print(f"  Strengths:   {strengths}")
    print(f"  Repeats:     {num_repeats}")
    print(f"  GPU:         {get_gpu_memory()}")
    print("=" * 70)

    experiment_start = time.time()
    model_times = {}

    for i, model_size in enumerate(models):
        model_start = time.time()
        model_name = QWEN3_MODELS.get(model_size, model_size)
        model_name_clean = model_name.replace("/", "_").replace("-", "_")

        print(f"\n{'=' * 70}")
        print(f"  MODEL {i+1}/{len(models)}: {model_size}")
        print(f"{'=' * 70}")

        # Find steering vector
        vector_dir = output_base / "steering_vectors" / concept / model_name_clean

        if layer is not None:
            vector_path = vector_dir / f"{concept}_layer_{layer}.pt"
        else:
            # Use optimal layer from layer search
            layer_search_summary = output_base / "layer_search" / concept / model_name_clean / "layer_search_summary.json"
            if layer_search_summary.exists():
                import json
                with open(layer_search_summary) as f:
                    summary = json.load(f)
                optimal_layer = summary["optimal_layer"]
                vector_path = vector_dir / f"{concept}_layer_{optimal_layer}.pt"
                print(f"  Using optimal layer: {optimal_layer}")
            else:
                print(f"  Warning: No layer search found, using layer 14 as fallback")
                vector_path = vector_dir / f"{concept}_layer_14.pt"

        if not vector_path.exists():
            print(f"  ERROR: Steering vector not found at {vector_path}")
            print(f"  Skipping {model_size}")
            continue

        # Run generation on each benchmark
        for bench_idx, benchmark in enumerate(benchmarks):
            print(f"\n  [{bench_idx+1}/{len(benchmarks)}] {benchmark['name'].upper()}")

            # Determine output directory based on benchmark type
            if benchmark['name'] == 'open_ended':
                output_dir = output_base / "open_ended_gens"
            else:
                output_dir = output_base / "capability_eval"

            output_path = output_dir / f"{benchmark['name']}_{concept}_{model_name_clean}.csv"

            try:
                run_benchmark_generation(
                    model_name=model_name,
                    prompts_path=Path(benchmark["prompts"]),
                    output_path=output_path,
                    steering_vector_path=vector_path,
                    strengths=strengths,
                    max_new_tokens=benchmark["max_new_tokens"],
                    num_repeats=num_repeats,
                    dtype=dtype,
                )
                print(f"    → Saved to {output_path}")

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

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

    print("\n" + "=" * 70)
    print("  EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\n  Total time: {format_duration(total_time)}")
    print(f"\n  Results saved to:")
    print(f"    {output_base / 'capability_eval'}")
    print(f"    {output_base / 'open_ended_gens'}")
    print("\n  Next steps:")
    print("    1. Transfer results to local machine")
    print("    2. Run evaluate_benchmarks.py for auto-graded metrics")
    print("    3. Run effect_evaluator.py for Opus confounder discovery")


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluation across models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single model with corrigible steering
    python run_benchmark_experiment.py --model 1.7B --concept corrigible_neutral_HHH

    # All models
    python run_benchmark_experiment.py --models 0.6B 1.7B 4B 8B 14B --concept corrigible_neutral_HHH

    # Custom strengths and layer
    python run_benchmark_experiment.py --model 1.7B --concept corrigible_neutral_HHH \
        --strengths -0.5 -0.25 0.0 0.25 0.5 --layer 14
        """
    )

    parser.add_argument("--model", type=str,
        help="Single model size (e.g., 1.7B)")
    parser.add_argument("--models", nargs="+",
        choices=list(QWEN3_MODELS.keys()),
        help="Multiple model sizes")
    parser.add_argument("--concept", type=str, required=True,
        help="Concept name (e.g., corrigible_neutral_HHH, sycophancy)")
    parser.add_argument("--layer", type=int, default=None,
        help="Layer to use (default: use optimal from layer search)")
    parser.add_argument("--strengths", nargs="+", type=float,
        default=[-0.25, -0.1, 0.0, 0.1, 0.25],
        help="Steering strengths (default: -0.25 -0.1 0.0 0.1 0.25)")
    parser.add_argument("--num-repeats", type=int, default=3,
        help="Generations per prompt (default: 3)")
    parser.add_argument("--output-base", type=str, default="results")
    parser.add_argument("--dtype", type=str, default="bfloat16",
        choices=["float32", "float16", "bfloat16"])

    args = parser.parse_args()

    # Determine models to run
    if args.models:
        models = args.models
    elif args.model:
        models = [args.model]
    else:
        parser.error("Must specify either --model or --models")

    run_full_benchmark_experiment(
        models=models,
        concept=args.concept,
        layer=args.layer,
        strengths=args.strengths,
        num_repeats=args.num_repeats,
        output_base=Path(args.output_base),
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
