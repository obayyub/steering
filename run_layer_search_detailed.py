#!/usr/bin/env python3
"""
Run detailed layer search with per-sample data saved for each layer.
Runs both 14B and 32B.
"""

import gc
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import train_steering_vector

from src.steering_utils import format_chat_prompt


MODELS = [
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
]
CONCEPT = "self_awareness_text_model"
STRENGTHS = [-1.0, 0.0, 1.0]
START_LAYER_FRAC = 0.33

DATA_DIR = Path("data/self_awareness")
OUTPUT_BASE = Path("results/results_100prompt/layer_search_detailed")


def extract_answer(text: str) -> str | None:
    import re
    text = text.strip()
    match = re.match(r'^\s*\(?([A-G])\)?', text, re.IGNORECASE)
    if match:
        return f"({match.group(1).upper()})"
    match = re.search(r'\(([A-G])\)', text, re.IGNORECASE)
    if match:
        return f"({match.group(1).upper()})"
    return None


def format_training_data_with_chat(pairs: list[dict], tokenizer) -> list[tuple[str, str]]:
    formatted = []
    for pair in pairs:
        prompt = pair.get("prompt", "")
        if not prompt:
            prompt = pair["positive"].rsplit("(", 1)[0].rstrip()

        messages = [{"role": "user", "content": prompt}]
        base = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        pos_answer = pair.get("matching_answer", "(A)")
        neg_answer = pair.get("not_matching_answer", "(B)")

        if "positive" in pair and pair["positive"].endswith(")"):
            pos_answer = "(" + pair["positive"].rsplit("(", 1)[-1]
        if "negative" in pair and pair["negative"].endswith(")"):
            neg_answer = "(" + pair["negative"].rsplit("(", 1)[-1]

        formatted.append((base + pos_answer, base + neg_answer))
    return formatted


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_model(model_name: str, train_pairs: list, eval_prompts: list):
    """Run layer search for a single model."""
    model_short = model_name.split("/")[-1]
    output_dir = OUTPUT_BASE / model_short
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  MODEL: {model_name}")
    print(f"{'='*70}")

    # Load model
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_data = format_training_data_with_chat(train_pairs, tokenizer)

    # Determine layers to test
    num_layers = model.config.num_hidden_layers
    start_layer = int(num_layers * START_LAYER_FRAC)
    layers_to_test = list(range(start_layer, num_layers))

    print(f"Model has {num_layers} layers, testing {len(layers_to_test)} layers ({start_layer}-{num_layers-1})")

    all_rows = []
    layer_summaries = []

    for layer in tqdm(layers_to_test, desc="Layers"):
        # Extract steering vector for this layer
        steering_vector = train_steering_vector(
            model=model,
            tokenizer=tokenizer,
            training_samples=training_data,
            layers=[layer],
            read_token_index=-2,
            show_progress=False,
        )

        layer_rows = []

        for strength in STRENGTHS:
            # Batch all prompts together
            batch_prompts = []
            batch_meta = []

            for i, ep in enumerate(eval_prompts):
                prompt = ep["prompt"]
                formatted = format_chat_prompt(tokenizer, prompt, enable_thinking=False)
                batch_prompts.append(formatted)
                batch_meta.append({
                    "prompt_idx": i,
                    "prompt": prompt[:200],
                    "matching_answer": ep.get("matching_answer", ""),
                    "not_matching_answer": ep.get("not_matching_answer", ""),
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
                "return_dict_in_generate": True,
                "output_scores": True,
            }

            if strength != 0:
                with steering_vector.apply(model, multiplier=strength):
                    gen_output = model.generate(**inputs, **gen_kwargs)
            else:
                gen_output = model.generate(**inputs, **gen_kwargs)

            # Get token IDs for answer tokens
            token_ids = {
                "A": tokenizer.encode("A", add_special_tokens=False)[0],
                "B": tokenizer.encode("B", add_special_tokens=False)[0],
                "Yes": tokenizer.encode("Yes", add_special_tokens=False)[0],
                "No": tokenizer.encode("No", add_special_tokens=False)[0],
            }

            # First token scores (scores is tuple of tensors, one per generated token)
            first_token_scores = gen_output.scores[0]  # shape: [batch_size, vocab_size]

            input_len = inputs["input_ids"].shape[1]

            for i, output in enumerate(gen_output.sequences):
                generation = tokenizer.decode(output[input_len:], skip_special_tokens=True)
                meta = batch_meta[i]

                # Extract logits for answer tokens
                logits_A = first_token_scores[i, token_ids["A"]].item()
                logits_B = first_token_scores[i, token_ids["B"]].item()
                logits_Yes = first_token_scores[i, token_ids["Yes"]].item()
                logits_No = first_token_scores[i, token_ids["No"]].item()

                extracted = extract_answer(generation)
                matching = meta["matching_answer"]
                not_matching = meta["not_matching_answer"]
                matches = extracted == matching.strip().upper() if extracted else None
                is_valid = extracted in [matching.strip().upper(), not_matching.strip().upper()] if extracted else False

                row = {
                    "layer": layer,
                    "strength": strength,
                    "prompt_idx": meta["prompt_idx"],
                    "prompt": meta["prompt"],
                    "generation": generation,
                    "extracted_answer": extracted,
                    "matching_answer": matching,
                    "not_matching_answer": not_matching,
                    "matches_behavior": matches,
                    "is_valid": is_valid,
                    "logit_A": logits_A,
                    "logit_B": logits_B,
                    "logit_Yes": logits_Yes,
                    "logit_No": logits_No,
                }
                layer_rows.append(row)
                all_rows.append(row)

        # Compute layer summary
        layer_df = pd.DataFrame(layer_rows)
        valid_df = layer_df[layer_df["is_valid"] == True]

        by_strength = valid_df.groupby("strength")["matches_behavior"].mean()
        baseline = by_strength.get(0.0, 0.5)
        positive = by_strength.get(1.0, baseline)
        negative = by_strength.get(-1.0, baseline)
        delta = positive - negative
        validity = len(valid_df) / len(layer_df) if len(layer_df) > 0 else 0

        layer_summaries.append({
            "layer": layer,
            "baseline_rate": baseline,
            "positive_rate": positive,
            "negative_rate": negative,
            "delta": delta,
            "valid_rate": validity,
        })

        tqdm.write(f"  Layer {layer}: delta={delta:+.3f}, valid={validity:.1%}, pos={positive:.1%}, neg={negative:.1%}")

    # Cleanup model before saving
    del model, tokenizer
    clear_memory()

    # Save all per-sample data
    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(output_dir / "per_sample_all_layers.csv", index=False)
    print(f"\nSaved {len(all_df)} per-sample rows to {output_dir}/per_sample_all_layers.csv")

    # Save layer summary
    summary_df = pd.DataFrame(layer_summaries)
    summary_df.to_csv(output_dir / "layer_summary.csv", index=False)
    print(f"Saved layer summary to {output_dir}/layer_summary.csv")

    # Find best layers
    print("\n" + "="*60)
    print("TOP 5 LAYERS BY POSITIVE DELTA")
    print("="*60)
    top5_pos = summary_df.nlargest(5, "delta")
    print(top5_pos.to_string(index=False))

    print("\n" + "="*60)
    print("TOP 5 LAYERS BY ABSOLUTE DELTA")
    print("="*60)
    summary_df["abs_delta"] = summary_df["delta"].abs()
    top5_abs = summary_df.nlargest(5, "abs_delta")
    print(top5_abs[["layer", "baseline_rate", "positive_rate", "negative_rate", "delta", "valid_rate"]].to_string(index=False))

    # Save summary JSON
    best_pos_layer = int(top5_pos.iloc[0]["layer"])
    best_pos_delta = top5_pos.iloc[0]["delta"]

    summary = {
        "model": model_name,
        "concept": CONCEPT,
        "num_layers": num_layers,
        "layers_tested": layers_to_test,
        "best_positive_layer": best_pos_layer,
        "best_positive_delta": best_pos_delta,
        "top_5_positive_layers": [int(x) for x in top5_pos["layer"].tolist()],
        "top_5_abs_layers": [int(x) for x in top5_abs["layer"].tolist()],
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBest positive layer: {best_pos_layer} (delta: {best_pos_delta:+.3f})")

    return summary


def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Load data
    train_data_path = DATA_DIR / "self_awareness_text_model_contrastive_pairs.json"
    eval_data_path = DATA_DIR / "self_awareness_text_model_eval_prompts.json"

    with open(train_data_path) as f:
        train_pairs = json.load(f)
    with open(eval_data_path) as f:
        eval_prompts = json.load(f)

    print(f"Loaded {len(train_pairs)} training pairs, {len(eval_prompts)} eval prompts")
    print(f"Models to run: {MODELS}")

    all_summaries = []

    for model_name in MODELS:
        summary = run_model(model_name, train_pairs, eval_prompts)
        all_summaries.append(summary)
        clear_memory()

    # Final summary
    print("\n" + "="*70)
    print("  ALL MODELS COMPLETE")
    print("="*70)

    for s in all_summaries:
        model_short = s["model"].split("/")[-1]
        print(f"{model_short}: best_layer={s['best_positive_layer']}, delta={s['best_positive_delta']:+.3f}")


if __name__ == "__main__":
    main()
