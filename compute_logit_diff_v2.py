#!/usr/bin/env python3
"""
Compute logit differences for steering evaluation - CORRECTED VERSION.

Computes P("(A" | prompt) vs P("(B" | prompt) via single forward pass,
NOT P(")" | prompt + answer) which was wrong.
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
OUTPUT_DIR = Path("results/results_100prompt/layer_search_detailed")


def main():
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

        # Get token IDs for "(A" and "(B" - these are single tokens in Qwen
        token_A = tokenizer.encode("(A", add_special_tokens=False)[0]
        token_B = tokenizer.encode("(B", add_special_tokens=False)[0]
        print(f"Token IDs: (A={token_A}, (B={token_B})")

        for layer in top_layers:
            print(f"\n  Layer {layer}:")

            # Load steering vector
            vector_dir = Path(f"results/results_100prompt/steering_vectors/self_awareness_text_model/Qwen_Qwen3_{model_key}")
            vector_path = vector_dir / f"self_awareness_text_model_layer_{layer}.pt"

            if not vector_path.exists():
                print(f"    Vector not found at {vector_path}, skipping")
                continue

            sv_data = torch.load(vector_path, weights_only=True)
            steering_vector = SteeringVector(
                layer_activations=sv_data["layer_activations"],
                layer_type=sv_data["layer_type"],
            )

            for strength in STRENGTHS:
                rows = []

                for i, ep in enumerate(tqdm(eval_prompts, desc=f"    Str {strength:+.1f}", leave=False)):
                    prompt = ep["prompt"]
                    matching = ep.get("matching_answer", "")

                    # Format prompt (no answer appended!)
                    formatted = format_chat_prompt(tokenizer, prompt, enable_thinking=False)
                    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

                    # Forward pass
                    with torch.no_grad():
                        if strength != 0:
                            with steering_vector.apply(model, multiplier=strength):
                                outputs = model(**inputs)
                        else:
                            outputs = model(**inputs)

                    # Get logits at LAST position (next token prediction)
                    logits = outputs.logits[0, -1, :]
                    log_probs = torch.log_softmax(logits, dim=-1)

                    logprob_A = log_probs[token_A].item()
                    logprob_B = log_probs[token_B].item()
                    logit_diff_AB = logprob_A - logprob_B

                    # Compute diff relative to matching answer
                    if matching.strip().upper() == "(A)":
                        logit_diff_matching = logit_diff_AB
                    else:
                        logit_diff_matching = -logit_diff_AB

                    rows.append({
                        "model": model_key,
                        "layer": layer,
                        "strength": strength,
                        "prompt_idx": i,
                        "matching_answer": matching,
                        "logprob_A": logprob_A,
                        "logprob_B": logprob_B,
                        "logit_diff_AB": logit_diff_AB,
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
    df.to_csv(OUTPUT_DIR / "logit_diff_results_v2.csv", index=False)
    print(f"\nSaved to {OUTPUT_DIR / 'logit_diff_results_v2.csv'}")


if __name__ == "__main__":
    main()
