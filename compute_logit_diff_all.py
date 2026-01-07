#!/usr/bin/env python3
"""
Compute logit differences for steering evaluation across multiple models and concepts.

Does a full layer sweep: trains steering vectors at each layer and computes
P("(A" | prompt) vs P("(B" | prompt) for all prompts at each layer.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import train_steering_vector

from src.steering_utils import format_chat_prompt


QWEN3_MODELS = {
    "4B": "Qwen/Qwen3-4B",
    "8B": "Qwen/Qwen3-8B",
    "14B": "Qwen/Qwen3-14B",
    "32B": "Qwen/Qwen3-32B",
}

CONCEPTS = {
    "corrigible": {
        "name": "corrigible_neutral_HHH",
        "data_dir": "data/corrigible",
        "train_file": "corrigible_neutral_HHH_contrastive_pairs.json",
        "eval_file": "corrigible_neutral_HHH_eval_prompts.json",
    },
    "self_awareness": {
        "name": "self_awareness_text_model",
        "data_dir": "data/self_awareness",
        "train_file": "self_awareness_text_model_contrastive_pairs.json",
        "eval_file": "self_awareness_text_model_eval_prompts.json",
    },
    "sycophancy": {
        "name": "sycophancy",
        "data_dir": "data/sycophancy",
        "train_file": "sycophancy_combined_contrastive_pairs.json",
        "eval_file": "sycophancy_combined_eval_prompts.json",
    },
}

STRENGTHS = [-1.0, 0.0, 1.0]
START_LAYER_FRAC = 0.33


def format_training_data_with_chat(
    pairs: list[dict], tokenizer
) -> list[tuple[str, str]]:
    """Format contrastive pairs with chat template for extraction."""
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


def compute_logit_diff_batch(
    model,
    tokenizer,
    prompts: list[str],
    steering_vector,
    strength: float,
    token_a: int,
    token_b: int,
) -> list[dict]:
    """Compute logit diffs for a batch of prompts."""
    formatted = [
        format_chat_prompt(tokenizer, p, enable_thinking=False) for p in prompts
    ]

    tokenizer.padding_side = "left"
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        if steering_vector and strength != 0:
            with steering_vector.apply(model, multiplier=strength):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)

    # Get logits at last position for each sequence
    # Need to handle padding - find the last non-pad token for each
    results = []
    for i in range(len(prompts)):
        # Find last real token position (before padding)
        attention_mask = inputs["attention_mask"][i]
        last_pos = attention_mask.sum() - 1

        logits = outputs.logits[i, last_pos, :]
        log_probs = torch.log_softmax(logits, dim=-1)

        results.append({
            "logprob_A": log_probs[token_a].item(),
            "logprob_B": log_probs[token_b].item(),
        })

    return results


def run_layer_sweep(
    model_keys: list[str],
    concept_keys: list[str],
    output_dir: Path,
    max_prompts: int | None = None,
    batch_size: int = 8,
):
    """Run logit diff layer sweep across models and concepts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_key in model_keys:
        model_name = QWEN3_MODELS[model_key]
        model_name_clean = model_name.replace("/", "_").replace("-", "_")

        print(f"\n{'='*70}")
        print(f"  MODEL: {model_key} ({model_name})")
        print(f"{'='*70}")

        # Load model
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Get token IDs
        token_a = tokenizer.encode("(A", add_special_tokens=False)[0]
        token_b = tokenizer.encode("(B", add_special_tokens=False)[0]
        print(f"Token IDs: (A={token_a}, (B={token_b})")

        # Determine layers to test
        num_layers = model.config.num_hidden_layers
        start_layer = int(num_layers * START_LAYER_FRAC)
        layers_to_test = list(range(start_layer, num_layers))
        print(f"Testing layers {start_layer} to {num_layers - 1}")

        for concept_key in concept_keys:
            concept_info = CONCEPTS[concept_key]
            concept_name = concept_info["name"]

            print(f"\n  Concept: {concept_key} ({concept_name})")

            # Load training and eval data
            train_path = Path(concept_info["data_dir"]) / concept_info["train_file"]
            eval_path = Path(concept_info["data_dir"]) / concept_info["eval_file"]

            if not train_path.exists() or not eval_path.exists():
                print(f"    Data files not found, skipping")
                continue

            with open(train_path, encoding="utf-8") as f:
                train_pairs = json.load(f)
            with open(eval_path, encoding="utf-8") as f:
                eval_prompts = json.load(f)

            if max_prompts:
                eval_prompts = eval_prompts[:max_prompts]

            print(f"    Train pairs: {len(train_pairs)}, Eval prompts: {len(eval_prompts)}")

            # Format training data
            training_data = format_training_data_with_chat(train_pairs, tokenizer)

            all_results = []
            layer_summaries = []

            for layer in tqdm(layers_to_test, desc="    Layers"):
                # Train steering vector for this layer
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
                    # Process in batches
                    prompts = [ep["prompt"] for ep in eval_prompts]

                    for batch_start in range(0, len(prompts), batch_size):
                        batch_end = min(batch_start + batch_size, len(prompts))
                        batch_prompts = prompts[batch_start:batch_end]
                        batch_eval = eval_prompts[batch_start:batch_end]

                        results = compute_logit_diff_batch(
                            model, tokenizer, batch_prompts,
                            steering_vector, strength,
                            token_a, token_b,
                        )

                        for i, (result, ep) in enumerate(zip(results, batch_eval)):
                            prompt_idx = batch_start + i
                            matching = ep.get(
                                "matching_answer",
                                ep.get("sycophantic_answer", "")
                            )

                            logit_diff_ab = result["logprob_A"] - result["logprob_B"]

                            # Compute diff relative to matching answer
                            if "(A)" in matching.upper():
                                logit_diff_matching = logit_diff_ab
                            else:
                                logit_diff_matching = -logit_diff_ab

                            row = {
                                "model": model_key,
                                "concept": concept_key,
                                "layer": layer,
                                "strength": strength,
                                "prompt_idx": prompt_idx,
                                "matching_answer": matching,
                                "logprob_A": result["logprob_A"],
                                "logprob_B": result["logprob_B"],
                                "logit_diff_AB": logit_diff_ab,
                                "logit_diff_matching": logit_diff_matching,
                            }
                            layer_rows.append(row)
                            all_results.append(row)

                # Compute layer summary
                layer_df = pd.DataFrame(layer_rows)
                by_strength = layer_df.groupby("strength")["logit_diff_matching"].mean()
                baseline = by_strength.get(0.0, 0.0)
                positive = by_strength.get(1.0, baseline)
                negative = by_strength.get(-1.0, baseline)
                delta = positive - negative

                layer_summaries.append({
                    "layer": layer,
                    "baseline_mean": baseline,
                    "positive_mean": positive,
                    "negative_mean": negative,
                    "delta": delta,
                })

                tqdm.write(
                    f"      Layer {layer}: delta={delta:+.2f}, "
                    f"pos={positive:+.2f}, neg={negative:+.2f}"
                )

            # Save per-sample results
            concept_dir = output_dir / concept_key / model_name_clean
            concept_dir.mkdir(parents=True, exist_ok=True)

            df = pd.DataFrame(all_results)
            df.to_csv(concept_dir / "per_sample_all_layers.csv", index=False)
            print(f"    Saved {len(df)} rows to {concept_dir}/per_sample_all_layers.csv")

            # Save layer summary
            summary_df = pd.DataFrame(layer_summaries)
            summary_df.to_csv(concept_dir / "layer_summary.csv", index=False)

            # Find best layers
            print("\n    TOP 5 LAYERS BY DELTA:")
            top5 = summary_df.nlargest(5, "delta")
            for _, row in top5.iterrows():
                print(
                    f"      Layer {int(row['layer'])}: "
                    f"delta={row['delta']:+.2f}"
                )

            # Save summary JSON
            best_layer = int(top5.iloc[0]["layer"])
            best_delta = top5.iloc[0]["delta"]

            summary = {
                "model": model_key,
                "concept": concept_key,
                "num_layers": num_layers,
                "layers_tested": layers_to_test,
                "best_layer": best_layer,
                "best_delta": best_delta,
                "top_5_layers": [int(x) for x in top5["layer"].tolist()],
            }
            with open(concept_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

        # Cleanup model
        del model, tokenizer
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Logit diff layer sweep across models and concepts"
    )

    parser.add_argument(
        "--models", nargs="+", default=["4B", "8B", "14B", "32B"],
        choices=list(QWEN3_MODELS.keys()),
        help="Model sizes to evaluate"
    )
    parser.add_argument(
        "--concepts", nargs="+", default=["corrigible", "self_awareness", "sycophancy"],
        choices=list(CONCEPTS.keys()),
        help="Concepts to evaluate"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/logit_diff_sweep",
        help="Output directory"
    )
    parser.add_argument(
        "--max-prompts", type=int, default=None,
        help="Max prompts per concept (for testing)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for logit computation"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  LOGIT DIFF LAYER SWEEP")
    print("=" * 70)
    print(f"  Models:   {args.models}")
    print(f"  Concepts: {args.concepts}")
    print(f"  Output:   {args.output_dir}")
    print("=" * 70)

    run_layer_sweep(
        model_keys=args.models,
        concept_keys=args.concepts,
        output_dir=Path(args.output_dir),
        max_prompts=args.max_prompts,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
