#!/usr/bin/env python3
"""
Compute logit differences for steering evaluation using forward pass (not generation).
"""

import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import SteeringVector

from src.steering_utils import format_chat_prompt


MODELS = {
    "14B": {
        "name": "Qwen/Qwen3-14B",
        "top_layers": [25, 21, 23, 19, 24],
    },
    "32B": {
        "name": "Qwen/Qwen3-32B",
        "top_layers": [49, 46, 43, 56, 44],
    },
}
STRENGTHS = [-1.0, 0.0, 1.0]
DATA_DIR = Path("data/self_awareness")
VECTOR_BASE = Path("results/results_100prompt/layer_search_detailed")
OUTPUT_DIR = Path("results/results_100prompt/layer_search_detailed/logit_analysis")


def compute_answer_logprobs(
    model,
    tokenizer,
    prompt: str,
    steering_vector: SteeringVector | None,
    strength: float,
) -> dict:
    """Compute log-probs for (A) and (B) answers via forward pass."""

    # Format with chat template
    base_formatted = format_chat_prompt(tokenizer, prompt, enable_thinking=False)

    results = {}
    for answer in ["(A)", "(B)"]:
        # Append answer to get full sequence
        full_text = base_formatted + answer

        inputs = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(model.device)

        # Forward pass (with or without steering)
        with torch.no_grad():
            if steering_vector and strength != 0:
                with steering_vector.apply(model, multiplier=strength):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

        # Get logits at the last real token position (before the final ")")
        # The sequence is: ... (A) or ... (B)
        # We want the logit that predicts ")" given "(A" or "(B"
        logits = outputs.logits[0, -1, :]  # logits at last position

        # Get log-prob of the ")" token
        close_paren_id = tokenizer.encode(")", add_special_tokens=False)[0]
        log_probs = torch.log_softmax(logits, dim=-1)

        results[answer] = {
            "log_prob_close_paren": log_probs[close_paren_id].item(),
            # Also get the log-prob of the sequence (sum of token log-probs)
        }

    return {
        "logprob_A": results["(A)"]["log_prob_close_paren"],
        "logprob_B": results["(B)"]["log_prob_close_paren"],
        "logit_diff_AB": results["(A)"]["log_prob_close_paren"] - results["(B)"]["log_prob_close_paren"],
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load eval prompts
    eval_data_path = DATA_DIR / "self_awareness_text_model_eval_prompts.json"
    with open(eval_data_path) as f:
        eval_prompts = json.load(f)
    print(f"Loaded {len(eval_prompts)} eval prompts")

    all_results = []

    for model_key, model_info in MODELS.items():
        model_name = model_info["name"]
        top_layers = model_info["top_layers"]

        print(f"\n{'='*60}")
        print(f"  {model_key}")
        print(f"{'='*60}")

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

        model_short = model_name.split("/")[-1]

        for layer in top_layers:
            print(f"\n  Layer {layer}:")

            # Load steering vector
            vector_dir = Path(f"results/results_100prompt/steering_vectors/self_awareness_text_model/Qwen_Qwen3_{model_key}")
            vector_path = vector_dir / f"self_awareness_text_model_layer_{layer}.pt"

            if not vector_path.exists():
                print(f"    Vector not found at {vector_path}, skipping layer {layer}")
                continue

            if vector_path.exists():
                sv_data = torch.load(vector_path, weights_only=True)
                steering_vector = SteeringVector(
                    layer_activations=sv_data["layer_activations"],
                    layer_type=sv_data["layer_type"],
                )
            else:
                steering_vector = None

            for strength in STRENGTHS:
                rows = []

                for i, ep in enumerate(tqdm(eval_prompts, desc=f"    Str {strength:+.1f}", leave=False)):
                    prompt = ep["prompt"]
                    matching = ep.get("matching_answer", "")

                    result = compute_answer_logprobs(
                        model, tokenizer, prompt, steering_vector, strength
                    )

                    # Determine if (A) or (B) is the matching answer
                    if matching.strip().upper() == "(A)":
                        logit_diff_matching = result["logit_diff_AB"]
                    else:
                        logit_diff_matching = -result["logit_diff_AB"]

                    rows.append({
                        "model": model_key,
                        "layer": layer,
                        "strength": strength,
                        "prompt_idx": i,
                        "matching_answer": matching,
                        "logprob_A": result["logprob_A"],
                        "logprob_B": result["logprob_B"],
                        "logit_diff_AB": result["logit_diff_AB"],
                        "logit_diff_matching": logit_diff_matching,
                    })

                all_results.extend(rows)

                # Print summary
                df_strength = pd.DataFrame(rows)
                mean_diff = df_strength["logit_diff_matching"].mean()
                print(f"    Strength {strength:+.1f}: mean logit_diff_matching = {mean_diff:.3f}")

        # Cleanup
        del model, tokenizer
        torch.cuda.empty_cache()

    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_DIR / "logit_diff_results.csv", index=False)
    print(f"\nSaved to {OUTPUT_DIR / 'logit_diff_results.csv'}")


if __name__ == "__main__":
    main()
