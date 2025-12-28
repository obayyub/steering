#!/usr/bin/env python3
"""
Find the optimal layer for steering by evaluating each layer individually.

Extracts vectors for layers past the first third, runs quick evaluation,
and reports which layer produces the best steering effect with least degradation.
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

# Quick eval settings - fewer samples for speed
QUICK_STRENGTHS = [-0.25, 0.0, 0.25]
QUICK_REPEATS = 1
QUICK_MAX_PROMPTS = 20


def print_status(msg: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")


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
    text = text.strip()
    match = re.match(r'^\s*\(?([A-G])\)?', text, re.IGNORECASE)
    if match:
        return f"({match.group(1).upper()})"
    match = re.search(r'\(([A-G])\)', text, re.IGNORECASE)
    if match:
        return f"({match.group(1).upper()})"
    return None


def evaluate_answer(generation: str, matching: str, not_matching: str) -> dict:
    extracted = extract_answer(generation)
    if extracted is None:
        return {"extracted_answer": None, "matches_behavior": None, "is_valid": False}

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


def quick_eval_layer(
    model,
    tokenizer,
    steering_vector: SteeringVector,
    eval_prompts: list[dict],
    layer: int,
    strengths: list[float] = QUICK_STRENGTHS,
    num_repeats: int = QUICK_REPEATS,
) -> dict:
    """Run quick evaluation on a single layer's steering vector."""
    rows = []

    for strength in strengths:
        batch_prompts = []
        batch_meta = []

        for ep in eval_prompts:
            for _ in range(num_repeats):
                formatted = format_chat_prompt(tokenizer, ep["prompt"], enable_thinking=False)
                batch_prompts.append(formatted)
                batch_meta.append({
                    "matching_answer": ep.get("matching_answer", ep.get("sycophantic_answer", "")),
                    "not_matching_answer": ep.get("not_matching_answer", ep.get("independent_answer", "")),
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
            eval_result = evaluate_answer(
                generation,
                batch_meta[i]["matching_answer"],
                batch_meta[i]["not_matching_answer"],
            )
            rows.append({
                "strength": strength,
                "layer": layer,
                **eval_result,
            })

    df = pd.DataFrame(rows)
    valid_df = df[df["is_valid"] == True]

    if len(valid_df) == 0:
        return {
            "layer": layer,
            "baseline_rate": None,
            "positive_rate": None,
            "negative_rate": None,
            "delta": 0,
            "valid_rate": 0,
        }

    by_strength = valid_df.groupby("strength")["matches_behavior"].mean()

    baseline = by_strength.get(0.0, 0.5)
    positive = by_strength.get(0.25, baseline)
    negative = by_strength.get(-0.25, baseline)

    return {
        "layer": layer,
        "baseline_rate": baseline,
        "positive_rate": positive,
        "negative_rate": negative,
        "delta": positive - negative,
        "valid_rate": len(valid_df) / len(df),
    }


def find_optimal_layer(
    model_name: str,
    concept: str,
    train_data_path: Path,
    eval_data_path: Path,
    output_base: Path,
    start_layer_frac: float = 0.33,
    dtype: str = "bfloat16",
    max_eval_prompts: int = QUICK_MAX_PROMPTS,
) -> dict:
    """
    Find optimal layer by evaluating each layer individually.

    Args:
        model_name: HuggingFace model name
        concept: Concept name
        train_data_path: Path to contrastive pairs
        eval_data_path: Path to eval prompts
        output_base: Output directory
        start_layer_frac: Start from this fraction of layers (skip early layers)
        dtype: Model dtype
        max_eval_prompts: Number of prompts for quick eval
    """
    model_name_clean = model_name.replace("/", "_").replace("-", "_")

    # Output directories
    vector_dir = output_base / "steering_vectors" / concept / model_name_clean
    results_dir = output_base / "layer_search" / concept / model_name_clean
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
    start_layer = int(num_layers * start_layer_frac)
    layers_to_test = list(range(start_layer, num_layers))

    print_status(f"Model: {num_layers} layers, testing {len(layers_to_test)} layers ({start_layer}-{num_layers-1})")

    # Load data
    print_status(f"Loading data...")
    train_pairs = load_json(train_data_path)
    eval_prompts = load_json(eval_data_path)[:max_eval_prompts]
    training_data = [(p["positive"], p["negative"]) for p in train_pairs]
    print_status(f"Train pairs: {len(train_pairs)}, Eval prompts: {len(eval_prompts)}")

    # Evaluate each layer
    results = []
    for layer in tqdm(layers_to_test, desc="Evaluating layers"):
        # Extract vector for this layer
        steering_vector = train_steering_vector(
            model=model,
            tokenizer=tokenizer,
            training_samples=training_data,
            layers=[layer],
            read_token_index=-2,  # Extract from A/B token, not the closing )
            show_progress=False,
        )

        # Save vector
        vector_path = vector_dir / f"{concept}_layer_{layer}.pt"
        torch.save({
            "layer_activations": steering_vector.layer_activations,
            "layer_type": steering_vector.layer_type,
        }, vector_path)

        metadata = {
            "model_name": model_name,
            "concept": concept,
            "num_pairs": len(train_pairs),
            "layer": layer,
        }
        with open(vector_dir / "metadata" / f"{concept}_layer_{layer}.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Quick eval
        layer_result = quick_eval_layer(
            model=model,
            tokenizer=tokenizer,
            steering_vector=steering_vector,
            eval_prompts=eval_prompts,
            layer=layer,
        )
        results.append(layer_result)

        # Progress update
        if layer_result["delta"] is not None:
            tqdm.write(f"  Layer {layer}: delta={layer_result['delta']:.3f}, valid={layer_result['valid_rate']:.1%}")

    # Cleanup
    del model, tokenizer
    clear_memory()

    # Save results
    df = pd.DataFrame(results)
    results_path = results_dir / "layer_search_results.csv"
    df.to_csv(results_path, index=False)

    # Find best layer
    valid_results = df[df["delta"].notna() & (df["valid_rate"] > 0.5)]
    if len(valid_results) == 0:
        print("\nNo valid results found!")
        return {"optimal_layer": None, "results": results}

    best_idx = valid_results["delta"].abs().idxmax()
    best_layer = int(valid_results.loc[best_idx, "layer"])
    best_delta = valid_results.loc[best_idx, "delta"]

    # Report
    print("\n" + "="*60)
    print("  LAYER SEARCH RESULTS")
    print("="*60)
    print(f"\n  Best layer: {best_layer} (delta: {best_delta:.3f})")
    print(f"\n  Top 5 layers by steering effect:")

    top5 = valid_results.nlargest(5, "delta")[["layer", "delta", "baseline_rate", "positive_rate", "negative_rate"]]
    print("  " + top5.to_string(index=False).replace("\n", "\n  "))

    print(f"\n  Results saved to: {results_path}")
    print(f"  Vectors saved to: {vector_dir}")

    # Save summary
    summary = {
        "model": model_name,
        "concept": concept,
        "num_layers": num_layers,
        "layers_tested": layers_to_test,
        "optimal_layer": best_layer,
        "optimal_delta": best_delta,
        "top_5_layers": top5["layer"].tolist(),
    }
    with open(results_dir / "layer_search_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Find optimal steering layer")

    parser.add_argument("--model", type=str, required=True, help="Model name or size")
    parser.add_argument("--concept", type=str, required=True, help="Concept name")
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--eval-data", type=str, required=True)
    parser.add_argument("--output-base", type=str, default="results")
    parser.add_argument("--start-layer-frac", type=float, default=0.33, help="Start from this fraction of layers")
    parser.add_argument("--max-eval-prompts", type=int, default=QUICK_MAX_PROMPTS)
    parser.add_argument("--dtype", type=str, default="bfloat16")

    args = parser.parse_args()

    model_name = QWEN3_MODELS.get(args.model, args.model)

    print("\n" + "="*60)
    print("  OPTIMAL LAYER SEARCH")
    print("="*60)
    print(f"  Model:   {model_name}")
    print(f"  Concept: {args.concept}")
    print(f"  GPU:     {get_gpu_memory()}")
    print("="*60 + "\n")

    result = find_optimal_layer(
        model_name=model_name,
        concept=args.concept,
        train_data_path=Path(args.train_data),
        eval_data_path=Path(args.eval_data),
        output_base=Path(args.output_base),
        start_layer_frac=args.start_layer_frac,
        max_eval_prompts=args.max_eval_prompts,
        dtype=args.dtype,
    )

    if result.get("optimal_layer"):
        print(f"\nOptimal layer: {result['optimal_layer']}")
        print(f"\nTo run full experiment with optimal layer:")
        print(f"  python run_eval_experiment.py \\")
        print(f"    --model {args.model} \\")
        print(f"    --concept {args.concept} \\")
        print(f"    --train-data {args.train_data} \\")
        print(f"    --eval-data {args.eval_data} \\")
        print(f"    --layer {result['optimal_layer']}")


if __name__ == "__main__":
    main()
