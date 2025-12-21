#!/usr/bin/env python3
"""
Run steering vector experiments across all Qwen3 model sizes.

Usage:
    python run_all_models.py
    python run_all_models.py --models 0.6B 1.7B  # specific sizes only
"""

import argparse
import gc
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from src import (
    ExtractionConfig,
    load_model_and_tokenizer,
    extract_steering_vector,
    load_steering_vector,
    analyze_generations,
)
from src.extract_vectors import (
    load_contrastive_pairs,
    format_training_data,
    get_layer_range,
    save_steering_vector,
)


QWEN3_MODELS = {
    "0.6B": "Qwen/Qwen3-0.6B",
    "1.7B": "Qwen/Qwen3-1.7B",
    "4B": "Qwen/Qwen3-4B",
    "8B": "Qwen/Qwen3-8B",
    "14B": "Qwen/Qwen3-14B",
    "32B": "Qwen/Qwen3-32B",
}

TEST_PROMPTS = [
    "Write a review of a restaurant you visited recently.",
    "Describe your experience with the new smartphone.",
    "Tell me about a movie you watched.",
    "What do you think about the local coffee shop?",
    "Share your thoughts on your recent hotel stay.",
    "Describe a book you finished reading.",
    "What did you think of the concert last night?",
    "Tell me about your experience at the gym.",
    "How was your flight to the destination?",
    "Describe the customer service you received.",
]

STEERING_STRENGTHS = [-1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0]
NUM_REPEATS = 5


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def print_status(msg: str, indent: int = 0):
    """Print timestamped status message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = "  " * indent
    print(f"[{timestamp}] {prefix}{msg}")


def print_phase(phase: str, model_size: str):
    """Print phase header."""
    print(f"\n{'─'*50}")
    print(f"  {model_size}: {phase}")
    print(f"{'─'*50}")


def clear_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def get_gpu_memory_used() -> str:
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"{used:.1f}/{total:.1f} GB"
    return "N/A"


class ExperimentTracker:
    """Track experiment progress and timing."""

    def __init__(self, model_sizes: list[str]):
        self.model_sizes = model_sizes
        self.total_models = len(model_sizes)
        self.completed_models = 0
        self.model_times: dict[str, float] = {}
        self.phase_times: dict[str, dict[str, float]] = {}
        self.experiment_start = time.time()
        self.results_so_far: list[dict] = []

    def start_model(self, size: str):
        """Mark start of a model run."""
        self.current_model = size
        self.current_model_start = time.time()
        self.phase_times[size] = {}

    def end_phase(self, phase: str):
        """Record end of a phase."""
        if not hasattr(self, 'phase_start'):
            return
        elapsed = time.time() - self.phase_start
        self.phase_times[self.current_model][phase] = elapsed
        print_status(f"✓ {phase} ({format_duration(elapsed)})", indent=1)

    def start_phase(self, phase: str):
        """Mark start of a phase."""
        self.phase_start = time.time()
        print_status(f"→ {phase}...", indent=1)

    def end_model(self, result: dict = None):
        """Mark end of a model run."""
        elapsed = time.time() - self.current_model_start
        self.model_times[self.current_model] = elapsed
        self.completed_models += 1

        if result:
            self.results_so_far.append(result)

        self._print_model_summary(elapsed)

    def _print_model_summary(self, elapsed: float):
        """Print summary after completing a model."""
        remaining = self.total_models - self.completed_models
        avg_time = sum(self.model_times.values()) / len(self.model_times)
        eta_seconds = avg_time * remaining
        total_elapsed = time.time() - self.experiment_start

        print(f"\n{'━'*60}")
        print(f"  {self.current_model} COMPLETE")
        print(f"{'━'*60}")
        print(f"  Time for this model:  {format_duration(elapsed)}")
        print(f"  Total elapsed:        {format_duration(total_elapsed)}")
        print(f"  Progress:             {self.completed_models}/{self.total_models} models")

        if remaining > 0:
            print(f"  Remaining:            {remaining} models")
            print(f"  ETA:                  {format_duration(eta_seconds)}")
            expected_finish = datetime.now() + timedelta(seconds=eta_seconds)
            print(f"  Expected finish:      {expected_finish.strftime('%H:%M:%S')}")

        print(f"  GPU memory:           {get_gpu_memory_used()}")

        # Show results so far
        if self.results_so_far:
            print(f"\n  Results so far (delta @ strength=0.5):")
            for r in self.results_so_far:
                delta = r.get("delta_at_0.5", "N/A")
                if isinstance(delta, float):
                    print(f"    {r['model_size']:>5}: {delta:+.3f}")

    def print_final_summary(self):
        """Print final experiment summary."""
        total_time = time.time() - self.experiment_start

        print(f"\n{'═'*60}")
        print(f"  EXPERIMENT COMPLETE")
        print(f"{'═'*60}")
        print(f"  Total time:     {format_duration(total_time)}")
        print(f"  Models run:     {self.completed_models}/{self.total_models}")

        if self.model_times:
            print(f"\n  Time per model:")
            for size, t in self.model_times.items():
                print(f"    {size:>5}: {format_duration(t)}")


def run_single_model(
    model_size: str,
    model_name: str,
    data_path: Path,
    output_dir: Path,
    tracker: ExperimentTracker,
):
    """Run extraction and evaluation for one model."""

    tracker.start_model(model_size)

    print(f"\n{'═'*60}")
    print(f"  MODEL: {model_size} ({model_name})")
    print(f"  GPU Memory: {get_gpu_memory_used()}")
    print(f"{'═'*60}")

    model_name_clean = model_name.replace("/", "_").replace("-", "_")

    # Config
    config = ExtractionConfig(
        model_name=model_name,
        data_path=data_path,
        output_dir=output_dir,
        torch_dtype="bfloat16",
        device="cuda",
    )

    # Phase 1: Load model
    tracker.start_phase("Loading model")
    model, tokenizer = load_model_and_tokenizer(config)
    num_layers = model.config.num_hidden_layers
    tracker.end_phase("Loading model")
    print_status(f"Layers: {num_layers}, dtype: {model.dtype}, GPU: {get_gpu_memory_used()}", indent=2)

    # Phase 2: Load data
    tracker.start_phase("Loading contrastive pairs")
    pairs = load_contrastive_pairs(config)
    training_data = format_training_data(pairs)
    tracker.end_phase("Loading contrastive pairs")
    print_status(f"Pairs: {len(training_data)}", indent=2)

    # Phase 3: Extract steering vector
    tracker.start_phase("Extracting steering vector")
    layer_start, layer_end = get_layer_range(model, config)
    steering_vector = extract_steering_vector(
        model=model,
        tokenizer=tokenizer,
        training_data=training_data,
        config=config,
    )
    tracker.end_phase("Extracting steering vector")
    print_status(f"Layers {layer_start}-{layer_end}", indent=2)

    # Save steering vector
    metadata = {
        "model_name": model_name,
        "model_size": model_size,
        "concept": "sentiment",
        "direction": "positive",
        "num_pairs": len(pairs),
        "layer_start": layer_start,
        "layer_end": layer_end,
        "num_layers": num_layers,
    }
    sv_path = save_steering_vector(steering_vector, config, metadata)
    sv = load_steering_vector(sv_path)

    # Phase 4: Generate with steering (batched by strength)
    total_gens = len(TEST_PROMPTS) * len(STEERING_STRENGTHS) * NUM_REPEATS
    batch_size = len(TEST_PROMPTS) * NUM_REPEATS  # All prompts × repeats per strength
    tracker.start_phase(f"Generating ({total_gens} samples, batch_size={batch_size})")

    rows = []
    pbar = tqdm(
        total=len(STEERING_STRENGTHS),
        desc=f"{model_size} generating",
        unit="batch",
        leave=False,
    )

    for strength in STEERING_STRENGTHS:
        # Build batch: all prompts × all repeats
        batch_prompts = []
        batch_meta = []
        for prompt in TEST_PROMPTS:
            for repeat in range(NUM_REPEATS):
                batch_prompts.append(prompt)
                batch_meta.append({"prompt": prompt, "repeat": repeat})

        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        # Generate with steering
        gen_kwargs = {
            "max_new_tokens": 128,
            "temperature": 0.7,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
        }

        if strength != 0:
            with sv.apply(model, multiplier=strength):
                outputs = model.generate(**inputs, **gen_kwargs)
        else:
            outputs = model.generate(**inputs, **gen_kwargs)

        # Decode outputs
        input_len = inputs["input_ids"].shape[1]
        for i, output in enumerate(outputs):
            generation = tokenizer.decode(
                output[input_len:],
                skip_special_tokens=True,
            )
            rows.append({
                "model": model_name,
                "model_size": model_size,
                "num_layers": num_layers,
                "label": sv.label,
                "strength": strength,
                "prompt": batch_meta[i]["prompt"],
                "repeat": batch_meta[i]["repeat"],
                "generation": generation,
            })

        pbar.update(1)

    pbar.close()
    df = pd.DataFrame(rows)
    tracker.end_phase(f"Generating ({total_gens} samples)")

    # Phase 5: Unload model
    tracker.start_phase("Unloading model")
    del model
    del tokenizer
    del steering_vector
    del sv
    clear_memory()
    tracker.end_phase("Unloading model")
    print_status(f"GPU after unload: {get_gpu_memory_used()}", indent=2)

    # Phase 6: Analyze sentiment
    tracker.start_phase("Analyzing sentiment")
    df = analyze_generations(df, device="cuda")
    tracker.end_phase("Analyzing sentiment")

    # Save raw results
    results_path = output_dir / f"{model_name_clean}_generations.csv"
    df.to_csv(results_path, index=False)
    print_status(f"Saved: {results_path}", indent=1)

    # Compute summary
    summary = df.groupby("strength")["p_positive"].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["strength", "mean_p_positive", "std_p_positive", "n_samples"]
    baseline = summary[summary["strength"] == 0.0]["mean_p_positive"].values[0]
    summary["delta_from_baseline"] = summary["mean_p_positive"] - baseline
    summary["model"] = model_name
    summary["model_size"] = model_size
    summary["num_layers"] = num_layers

    summary_path = output_dir / f"{model_name_clean}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print_status(f"Saved: {summary_path}", indent=1)

    # Get delta at strength 0.5 for tracking
    delta_0_5 = summary[summary["strength"] == 0.5]["delta_from_baseline"].values[0]

    # Print summary table
    print(f"\n  {model_size} Results:")
    print("  " + summary[["strength", "mean_p_positive", "delta_from_baseline"]].to_string(index=False).replace("\n", "\n  "))

    # Clear sentiment model
    clear_memory()

    result = {
        "model": model_name,
        "model_size": model_size,
        "num_layers": num_layers,
        "results_path": str(results_path),
        "summary_path": str(summary_path),
        "delta_at_0.5": delta_0_5,
    }

    tracker.end_model(result)

    return result


def main():
    parser = argparse.ArgumentParser(description="Run steering experiments on all Qwen3 sizes")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(QWEN3_MODELS.keys()),
        choices=list(QWEN3_MODELS.keys()),
        help="Model sizes to run (default: all)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/contrastive_pairs.json",
        help="Path to contrastive pairs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data)

    # Print experiment config
    total_gens_per_model = len(TEST_PROMPTS) * len(STEERING_STRENGTHS) * NUM_REPEATS
    total_gens = total_gens_per_model * len(args.models)

    print("\n" + "═"*60)
    print("  STEERING VECTOR EXPERIMENT")
    print("═"*60)
    print(f"  Start time:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Models:        {', '.join(args.models)}")
    print(f"  Data:          {data_path}")
    print(f"  Output:        {output_dir}")
    print(f"  Prompts:       {len(TEST_PROMPTS)}")
    print(f"  Strengths:     {len(STEERING_STRENGTHS)} ({min(STEERING_STRENGTHS)} to {max(STEERING_STRENGTHS)})")
    print(f"  Repeats:       {NUM_REPEATS}")
    print(f"  Total gens:    {total_gens} ({total_gens_per_model} per model)")
    print(f"  GPU:           {get_gpu_memory_used()}")
    print("═"*60 + "\n")

    tracker = ExperimentTracker(args.models)
    all_results = []
    all_summaries = []

    for size in args.models:
        model_name = QWEN3_MODELS[size]
        try:
            result = run_single_model(
                model_size=size,
                model_name=model_name,
                data_path=data_path,
                output_dir=output_dir,
                tracker=tracker,
            )
            all_results.append(result)

            # Load and accumulate summary
            summary_df = pd.read_csv(result["summary_path"])
            all_summaries.append(summary_df)

        except Exception as e:
            print(f"\n❌ ERROR with {size}: {e}")
            import traceback
            traceback.print_exc()
            clear_memory()
            continue

    # Combine all summaries
    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined_path = output_dir / "all_models_summary.csv"
        combined.to_csv(combined_path, index=False)

        # Print final comparison
        print("\n" + "═"*60)
        print("  FINAL RESULTS: Delta P(positive) by model size")
        print("═"*60)
        pivot = combined.pivot_table(
            index="model_size",
            columns="strength",
            values="delta_from_baseline",
            aggfunc="mean"
        )
        # Sort by model size (numeric)
        size_order = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"]
        pivot = pivot.reindex([s for s in size_order if s in pivot.index])
        print(pivot.round(3).to_string())

    # Save run metadata
    meta = {
        "timestamp": datetime.now().isoformat(),
        "models": args.models,
        "prompts": TEST_PROMPTS,
        "strengths": STEERING_STRENGTHS,
        "num_repeats": NUM_REPEATS,
        "results": all_results,
        "timing": tracker.model_times,
    }
    meta_path = output_dir / "experiment_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    tracker.print_final_summary()
    print(f"\n✓ All results saved to {output_dir}/")


if __name__ == "__main__":
    main()
