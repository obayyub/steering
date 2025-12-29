#!/usr/bin/env python3
"""
Generate completions on benchmarks with steering vectors applied.

Works with any JSON prompt file containing:
- prompt: the text to complete
- (optional) category, metric, answer fields for evaluation

Supports:
- Capability benchmarks (GSM8K, TriviaQA, IFEval)
- Behavioral pairs (sycophancy, corrigible, sentiment)
"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import SteeringVector

from src.steering_utils import format_chat_prompt


def load_steering_vector(vector_path: Path, model) -> SteeringVector:
    """Load a steering vector from disk."""
    data = torch.load(vector_path, map_location=model.device)
    return SteeringVector(
        layer_activations=data["layer_activations"],
        layer_type=data.get("layer_type", "decoder_block"),
    )


def generate_with_steering(
    model,
    tokenizer,
    prompts: list[str],
    steering_vector: Optional[SteeringVector] = None,
    strength: float = 0.0,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    batch_size: int = 8,
) -> list[str]:
    """
    Generate completions with optional steering.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompts: List of prompts to complete
        steering_vector: Optional steering vector
        strength: Steering strength multiplier
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        batch_size: Batch size for generation

    Returns:
        List of generated completions
    """
    generations = []

    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Generating (strength={strength})"):
        batch_prompts = prompts[i:i + batch_size]

        tokenizer.padding_side = "left"
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
        }

        # Apply steering if requested
        if steering_vector is not None and strength != 0:
            with steering_vector.apply(model, multiplier=strength):
                outputs = model.generate(**inputs, **gen_kwargs)
        else:
            outputs = model.generate(**inputs, **gen_kwargs)

        # Decode generations
        input_len = inputs["input_ids"].shape[1]
        for output in outputs:
            generation = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            generations.append(generation)

    return generations


def run_benchmark_generation(
    model_name: str,
    prompts_path: Path,
    output_path: Path,
    steering_vector_path: Optional[Path] = None,
    strengths: list[float] = [0.0],
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    batch_size: int = 8,
    num_repeats: int = 1,
    dtype: str = "bfloat16",
):
    """
    Run generation on benchmark prompts with steering.

    Args:
        model_name: HuggingFace model name
        prompts_path: Path to prompts JSON
        output_path: Path to save generations CSV
        steering_vector_path: Optional path to steering vector
        strengths: List of steering strengths to test
        max_new_tokens: Max tokens per generation
        temperature: Sampling temperature
        batch_size: Batch size
        num_repeats: Generations per (prompt, strength) pair
        dtype: Model dtype
    """
    # Load model
    print(f"Loading model: {model_name}")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load steering vector if provided
    steering_vector = None
    if steering_vector_path:
        print(f"Loading steering vector: {steering_vector_path}")
        steering_vector = load_steering_vector(steering_vector_path, model)

    # Load prompts
    print(f"Loading prompts: {prompts_path}")
    with open(prompts_path) as f:
        prompt_data = json.load(f)

    # Format prompts with chat template
    formatted_prompts = []
    for item in prompt_data:
        formatted = format_chat_prompt(tokenizer, item["prompt"], enable_thinking=False)
        formatted_prompts.append(formatted)

    print(f"Generating on {len(formatted_prompts)} prompts")
    print(f"  Strengths: {strengths}")
    print(f"  Repeats: {num_repeats}")

    # Generate for each strength
    rows = []
    for strength in strengths:
        for repeat_idx in range(num_repeats):
            generations = generate_with_steering(
                model=model,
                tokenizer=tokenizer,
                prompts=formatted_prompts,
                steering_vector=steering_vector,
                strength=strength,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                batch_size=batch_size,
            )

            # Record results
            for i, generation in enumerate(generations):
                rows.append({
                    "prompt": prompt_data[i]["prompt"],
                    "formatted_prompt": formatted_prompts[i],
                    "generation": generation,
                    "strength": strength,
                    "repeat": repeat_idx,
                    "category": prompt_data[i].get("category", "unknown"),
                })

    # Save results
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nGenerated {len(df)} completions")
    print(f"Saved to: {output_path}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate completions on benchmarks with steering"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompts", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--steering-vector", type=str, default=None)
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.0])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16")

    args = parser.parse_args()

    run_benchmark_generation(
        model_name=args.model,
        prompts_path=Path(args.prompts),
        output_path=Path(args.output),
        steering_vector_path=Path(args.steering_vector) if args.steering_vector else None,
        strengths=args.strengths,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
        num_repeats=args.num_repeats,
        dtype=args.dtype,
    )
