#!/usr/bin/env python3
"""
Run steering experiments on Anthropic eval datasets.

Generic runner that works with any Anthropic eval format:
- question with (A)/(B)/etc. choices
- answer_matching_behavior / answer_not_matching_behavior

Can use pre-extracted vectors or extract inline.
"""

import argparse
import gc
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import train_steering_vector, SteeringVector

from src.steering_utils import format_chat_prompt


QWEN3_MODELS = {
    "0.6B": "Qwen/Qwen3-0.6B",
    "1.7B": "Qwen/Qwen3-1.7B",
    "4B": "Qwen/Qwen3-4B",
    "8B": "Qwen/Qwen3-8B",
    "14B": "Qwen/Qwen3-14B",
    "32B": "Qwen/Qwen3-32B",
}

DEFAULT_STRENGTHS = [-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5]


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


def print_status(msg: str, indent: int = 0):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {'  ' * indent}{msg}")


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_memory() -> str:
    if not torch.cuda.is_available():
        return "N/A"
    used = torch.cuda.memory_allocated(0) / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"{used:.1f}/{total:.1f} GB"


def get_num_layers(model) -> int:
    if hasattr(model.config, "num_hidden_layers"):
        return model.config.num_hidden_layers
    elif hasattr(model.config, "n_layer"):
        return model.config.n_layer
    raise ValueError("Could not determine number of layers")


def load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("pairs", data)


def extract_answer(text: str) -> Optional[str]:
    """Extract answer choice (A), (B), etc. from text."""
    text = text.strip()

    # Direct match at start
    match = re.match(r'^\s*\(?([A-G])\)?', text, re.IGNORECASE)
    if match:
        return f"({match.group(1).upper()})"

    # Look for (A), (B), etc anywhere
    match = re.search(r'\(([A-G])\)', text, re.IGNORECASE)
    if match:
        return f"({match.group(1).upper()})"

    return None


def evaluate_answer(generation: str, matching: str, not_matching: str) -> dict:
    """Evaluate if generation matches target behavior."""
    extracted = extract_answer(generation)

    if extracted is None:
        return {
            "extracted_answer": None,
            "matches_behavior": None,
            "is_valid": False,
        }

    matching_norm = matching.strip().upper()
    not_matching_norm = not_matching.strip().upper()
    extracted_norm = extracted.upper()

    matches = extracted_norm == matching_norm
    anti_matches = extracted_norm == not_matching_norm

    return {
        "extracted_answer": extracted,
        "matches_behavior": matches,
        "is_valid": matches or anti_matches,
    }


def load_steering_vector(path: Path) -> SteeringVector:
    """Load a steering vector from disk."""
    data = torch.load(path, weights_only=False)
    return SteeringVector(
        layer_activations=data["layer_activations"],
        layer_type=data.get("layer_type", "decoder_block"),
    )


def run_experiment(
    model_name: str,
    concept: str,
    train_data_path: Path,
    eval_data_path: Path,
    output_base: Path,
    vector_path: Optional[Path] = None,
    layer: Optional[int] = None,
    layer_start: Optional[int] = None,
    layer_end: Optional[int] = None,
    strengths: Optional[list[float]] = None,
    num_repeats: int = 3,
    max_eval_prompts: Optional[int] = None,
    dtype: str = "bfloat16",
) -> dict:
    """
    Run a steering experiment.

    Args:
        model_name: HuggingFace model name
        concept: Name of the concept (e.g., sycophancy, corrigibility)
        train_data_path: Path to contrastive pairs JSON
        eval_data_path: Path to eval prompts JSON
        output_base: Base output directory
        vector_path: Path to pre-extracted vector (if None, extracts inline)
        layer: Single layer to use (mutually exclusive with layer_start/end)
        layer_start/end: Layer range (if None and vector_path is None, uses middle third)
        strengths: List of steering strengths to test
        num_repeats: Number of generations per prompt
        max_eval_prompts: Limit number of eval prompts (for quick testing)
        dtype: Model dtype
    """
    if strengths is None:
        strengths = DEFAULT_STRENGTHS

    model_name_clean = model_name.replace("/", "_").replace("-", "_")

    # Output directories
    vector_dir = output_base / "steering_vectors" / concept / model_name_clean
    results_dir = output_base / "generations" / concept / model_name_clean
    vector_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    (vector_dir / "metadata").mkdir(exist_ok=True)

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    # Load model
    print_status(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = get_num_layers(model)
    print_status(f"Model loaded: {num_layers} layers, GPU: {get_gpu_memory()}", indent=1)

    # Determine layers
    if vector_path is not None:
        # Use pre-extracted vector
        print_status(f"Loading pre-extracted vector: {vector_path}")
        steering_vector = load_steering_vector(vector_path)
        # Infer layer info from filename
        if "_layer_" in vector_path.stem:
            match = re.search(r'_layer_(\d+)', vector_path.stem)
            if match:
                l_start = l_end = int(match.group(1))
            else:
                match = re.search(r'_layers_(\d+)_(\d+)', vector_path.stem)
                l_start, l_end = int(match.group(1)), int(match.group(2))
        else:
            l_start, l_end = 0, num_layers - 1  # Unknown, assume all
        layer_suffix = vector_path.stem.split(concept)[-1]
    else:
        # Extract vector inline
        if layer is not None:
            l_start = l_end = layer
        elif layer_start is not None and layer_end is not None:
            l_start, l_end = layer_start, layer_end
        else:
            third = num_layers // 3
            l_start, l_end = third, 2 * third

        layer_suffix = f"_layer_{l_start}" if l_start == l_end else f"_layers_{l_start}_{l_end}"

        print_status(f"Loading training data: {train_data_path}")
        train_pairs = load_json(train_data_path)
        training_data = [(p["positive"], p["negative"]) for p in train_pairs]
        print_status(f"Loaded {len(train_pairs)} contrastive pairs", indent=1)

        print_status(f"Extracting steering vector (layers {l_start}-{l_end})...")
        steering_vector = train_steering_vector(
            model=model,
            tokenizer=tokenizer,
            training_samples=training_data,
            layers=list(range(l_start, l_end + 1)),
            show_progress=True,
        )

        # Save vector
        vector_path = vector_dir / f"{concept}{layer_suffix}.pt"
        torch.save({
            "layer_activations": steering_vector.layer_activations,
            "layer_type": steering_vector.layer_type,
        }, vector_path)

        metadata = {
            "model_name": model_name,
            "concept": concept,
            "num_pairs": len(train_pairs),
            "layer_start": l_start,
            "layer_end": l_end,
            "dtype": dtype,
        }
        with open(vector_dir / "metadata" / f"{concept}{layer_suffix}.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print_status(f"Saved vector: {vector_path}", indent=1)

    # Load eval prompts
    print_status(f"Loading eval data: {eval_data_path}")
    eval_prompts = load_json(eval_data_path)
    if max_eval_prompts and len(eval_prompts) > max_eval_prompts:
        eval_prompts = eval_prompts[:max_eval_prompts]
    print_status(f"Loaded {len(eval_prompts)} eval prompts", indent=1)

    # Generate
    total_gens = len(eval_prompts) * len(strengths) * num_repeats
    print_status(f"Generating {total_gens} samples...")

    rows = []
    for strength in tqdm(strengths, desc="Strengths", unit="strength"):
        batch_prompts = []
        batch_meta = []

        for ep in eval_prompts:
            for repeat in range(num_repeats):
                formatted = format_chat_prompt(tokenizer, ep["prompt"], enable_thinking=False)
                batch_prompts.append(formatted)
                batch_meta.append({
                    "prompt": ep["prompt"],
                    "matching_answer": ep.get("matching_answer", ep.get("sycophantic_answer", "")),
                    "not_matching_answer": ep.get("not_matching_answer", ep.get("independent_answer", "")),
                    "behavior": ep.get("behavior", concept),
                    "repeat": repeat,
                })

        tokenizer.padding_side = "left"
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        gen_kwargs = {
            "max_new_tokens": 32,
            "temperature": 0.7,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
        }

        if strength != 0:
            with steering_vector.apply(model, multiplier=strength):
                outputs = model.generate(**inputs, **gen_kwargs)
        else:
            outputs = model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        for i, output in enumerate(outputs):
            generation = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            meta = batch_meta[i]
            eval_result = evaluate_answer(
                generation,
                meta["matching_answer"],
                meta["not_matching_answer"],
            )

            rows.append({
                "model": model_name,
                "concept": concept,
                "strength": strength,
                "layer_start": l_start,
                "layer_end": l_end,
                "prompt": meta["prompt"][:200],
                "matching_answer": meta["matching_answer"],
                "not_matching_answer": meta["not_matching_answer"],
                "behavior": meta["behavior"],
                "repeat": meta["repeat"],
                "generation": generation,
                **eval_result,
            })

    df = pd.DataFrame(rows)

    # Cleanup
    print_status("Unloading model...")
    del model, tokenizer, steering_vector
    clear_memory()

    # Save results
    results_path = results_dir / f"generations{layer_suffix}.csv"
    df.to_csv(results_path, index=False)
    print_status(f"Saved: {results_path}", indent=1)

    # Summary
    valid_df = df[df["is_valid"] == True]
    summary = valid_df.groupby("strength").agg(
        behavior_rate=("matches_behavior", "mean"),
        n_valid=("matches_behavior", "count"),
    ).reset_index()
    summary["model"] = model_name
    summary["concept"] = concept
    summary["layer_start"] = l_start
    summary["layer_end"] = l_end

    summary_path = results_dir / f"summary{layer_suffix}.csv"
    summary.to_csv(summary_path, index=False)

    print(f"\n  Results (behavior matching rate):")
    print("  " + summary[["strength", "behavior_rate", "n_valid"]].to_string(index=False).replace("\n", "\n  "))

    # Compute delta from baseline
    baseline_rate = summary[summary["strength"] == 0.0]["behavior_rate"].values
    baseline_rate = baseline_rate[0] if len(baseline_rate) > 0 else 0.5
    max_rate = summary["behavior_rate"].max()
    min_rate = summary["behavior_rate"].min()

    return {
        "model": model_name,
        "concept": concept,
        "layer_start": l_start,
        "layer_end": l_end,
        "results_path": str(results_path),
        "summary_path": str(summary_path),
        "baseline_rate": baseline_rate,
        "max_rate": max_rate,
        "min_rate": min_rate,
        "delta": max_rate - min_rate,
    }


def main():
    parser = argparse.ArgumentParser(description="Run steering experiment on Anthropic evals")

    parser.add_argument("--model", type=str, required=True, help="Model name or size (0.6B, 1.7B, etc.)")
    parser.add_argument("--concept", type=str, required=True, help="Concept name (e.g., sycophancy)")
    parser.add_argument("--train-data", type=str, required=True, help="Path to contrastive pairs JSON")
    parser.add_argument("--eval-data", type=str, required=True, help="Path to eval prompts JSON")
    parser.add_argument("--output-base", type=str, default="results")

    parser.add_argument("--vector", type=str, default=None, help="Path to pre-extracted vector (skips extraction)")

    parser.add_argument("--layer", type=int, default=None, help="Single layer to use")
    parser.add_argument("--layer-start", type=int, default=None)
    parser.add_argument("--layer-end", type=int, default=None)

    parser.add_argument("--strengths", type=float, nargs="+", default=None, help="Steering strengths to test")
    parser.add_argument("--num-repeats", type=int, default=3)
    parser.add_argument("--max-eval-prompts", type=int, default=None, help="Limit eval prompts (for testing)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])

    args = parser.parse_args()

    # Resolve model name
    model_name = QWEN3_MODELS.get(args.model, args.model)

    print("\n" + "="*60)
    print("  STEERING EXPERIMENT")
    print("="*60)
    print(f"  Model:   {model_name}")
    print(f"  Concept: {args.concept}")
    print(f"  Train:   {args.train_data}")
    print(f"  Eval:    {args.eval_data}")
    print(f"  Output:  {args.output_base}")
    print(f"  GPU:     {get_gpu_memory()}")
    print("="*60 + "\n")

    result = run_experiment(
        model_name=model_name,
        concept=args.concept,
        train_data_path=Path(args.train_data),
        eval_data_path=Path(args.eval_data),
        output_base=Path(args.output_base),
        vector_path=Path(args.vector) if args.vector else None,
        layer=args.layer,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        strengths=args.strengths,
        num_repeats=args.num_repeats,
        max_eval_prompts=args.max_eval_prompts,
        dtype=args.dtype,
    )

    print(f"\nExperiment complete!")
    print(f"  Baseline rate: {result['baseline_rate']:.1%}")
    print(f"  Max rate:      {result['max_rate']:.1%}")
    print(f"  Min rate:      {result['min_rate']:.1%}")
    print(f"  Delta:         {result['delta']:.1%}")


if __name__ == "__main__":
    main()
