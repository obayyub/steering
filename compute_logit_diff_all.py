#!/usr/bin/env python3
"""
Compute logit differences for steering evaluation across multiple models and concepts.

Does a full layer sweep: extracts steering vectors at each layer and computes
P("(A" | prompt) vs P("(B" | prompt) for all prompts at each layer.

Supports both dense models and MoE models with expert parallelism.

Usage:
  # Dense models (single GPU)
  python compute_logit_diff_all.py --models 4B 8B 14B 32B

  # MoE models (multi-GPU with expert parallelism)
  torchrun --nproc-per-node=8 compute_logit_diff_all.py --models 235B-A22B
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.steering_utils import format_chat_prompt


QWEN3_MODELS = {
    # Dense models
    "4B": {"name": "Qwen/Qwen3-4B", "moe": False},
    "8B": {"name": "Qwen/Qwen3-8B", "moe": False},
    "14B": {"name": "Qwen/Qwen3-14B", "moe": False},
    "32B": {"name": "Qwen/Qwen3-32B", "moe": False},
    # MoE models
    "30B-A3B": {"name": "Qwen/Qwen3-30B-A3B", "moe": True},
    "235B-A22B": {"name": "Qwen/Qwen3-235B-A22B", "moe": True},
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


def is_main_process() -> bool:
    """Check if this is the main process in distributed setting."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def print_main(*args, **kwargs):
    """Print only from main process."""
    if is_main_process():
        print(*args, **kwargs)


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


def extract_all_steering_vectors(
    model,
    tokenizer,
    pairs: list[tuple[str, str]],
    layer_indices: list[int],
    batch_size: int = 32,
) -> dict[int, torch.Tensor]:
    """
    Extract steering vectors for ALL layers in just 2 forward passes.

    Instead of looping over layers (2 forward passes per layer), we:
    1. Run one forward pass for all positives → get all layer activations
    2. Run one forward pass for all negatives → get all layer activations
    3. Compute steering vectors for each layer from cached activations

    This is O(2) forward passes instead of O(2 * num_layers).
    """
    pos_texts = [p[0] for p in pairs]
    neg_texts = [p[1] for p in pairs]

    tokenizer.padding_side = "left"

    def get_all_layer_activations(texts: list[str]) -> dict[int, torch.Tensor]:
        """Get activations for all layers of interest."""
        # Accumulate activations per layer across batches
        layer_acts = {layer: [] for layer in layer_indices}

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(model.device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            # hidden_states is tuple of (num_layers + 1) tensors
            # Each tensor is (batch, seq_len, hidden_dim)
            for layer in layer_indices:
                acts = outputs.hidden_states[layer][:, -1, :].cpu()
                layer_acts[layer].append(acts)

        # Concatenate batches for each layer
        return {layer: torch.cat(acts, dim=0) for layer, acts in layer_acts.items()}

    pos_acts_all = get_all_layer_activations(pos_texts)
    neg_acts_all = get_all_layer_activations(neg_texts)

    # Compute steering vector for each layer
    steering_vectors = {}
    for layer in layer_indices:
        diff = pos_acts_all[layer] - neg_acts_all[layer]
        steering_vectors[layer] = diff.mean(dim=0)

    return steering_vectors


class SteeringHook:
    """Hook to add steering vector to residual stream."""

    def __init__(self, steering_vector: torch.Tensor, multiplier: float):
        self.steering_vector = steering_vector
        self.multiplier = multiplier
        self.handle = None

    def __call__(self, module, input, output):
        # output is typically (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            hidden_states = output[0]
            hidden_states = hidden_states + self.multiplier * self.steering_vector.to(hidden_states.device)
            return (hidden_states,) + output[1:]
        else:
            return output + self.multiplier * self.steering_vector.to(output.device)

    def register(self, layer_module):
        self.handle = layer_module.register_forward_hook(self)
        return self

    def remove(self):
        if self.handle:
            self.handle.remove()


def compute_logit_diff_batch(
    model,
    tokenizer,
    prompts: list[str],
    steering_vector: Optional[torch.Tensor],
    strength: float,
    layer_idx: int,
    token_a: int,
    token_b: int,
    batch_size: int = 8,
) -> list[dict]:
    """Compute logit diffs for a batch of prompts with optional steering."""
    formatted = [
        format_chat_prompt(tokenizer, p, enable_thinking=False) for p in prompts
    ]

    tokenizer.padding_side = "left"

    results = []

    # Get the layer module for hook
    # Qwen3 uses model.model.layers[idx]
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layer_module = model.model.layers[layer_idx]
    else:
        raise ValueError(f"Cannot find layer module for layer {layer_idx}")

    # Setup steering hook if needed
    hook = None
    if steering_vector is not None and strength != 0:
        hook = SteeringHook(steering_vector, strength).register(layer_module)

    try:
        for i in range(0, len(formatted), batch_size):
            batch_texts = formatted[i:i + batch_size]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(model.device)

            with torch.no_grad():
                outputs = model(**inputs)

            # Get logits at last position for each sequence
            for j in range(len(batch_texts)):
                logits = outputs.logits[j, -1, :]
                log_probs = torch.log_softmax(logits, dim=-1)

                results.append({
                    "logprob_A": log_probs[token_a].item(),
                    "logprob_B": log_probs[token_b].item(),
                })
    finally:
        if hook:
            hook.remove()

    return results


def run_layer_sweep(
    model_keys: list[str],
    concept_keys: list[str],
    output_dir: Path,
    max_prompts: Optional[int] = None,
    max_train_pairs: Optional[int] = None,
    batch_size: int = 8,
    train_batch_size: int = 32,
):
    """Run logit diff layer sweep across models and concepts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_key in model_keys:
        model_info = QWEN3_MODELS[model_key]
        model_name = model_info["name"]
        is_moe = model_info["moe"]
        model_name_clean = model_name.replace("/", "_").replace("-", "_")

        print_main(f"\n{'='*70}")
        print_main(f"  MODEL: {model_key} ({model_name})")
        print_main(f"  Type: {'MoE' if is_moe else 'Dense'}")
        print_main(f"{'='*70}")

        # Load model
        print_main("Loading model...")

        if is_moe:
            # Use expert parallelism for MoE models
            try:
                from transformers.distributed.configuration_utils import DistributedConfig
                distributed_config = DistributedConfig(enable_expert_parallel=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.bfloat16,  # Use 'dtype' not 'torch_dtype'
                    distributed_config=distributed_config,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
            except ImportError as e:
                print_main(f"Warning: DistributedConfig not available ({e}), falling back to device_map='auto'")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )
            except Exception as e:
                print(f"[Rank {os.environ.get('RANK', '?')}] Error loading model: {e}")
                import traceback
                traceback.print_exc()
                raise
        else:
            # Standard loading for dense models
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
        print_main(f"Token IDs: (A={token_a}, (B={token_b})")

        # Determine layers to test
        num_layers = model.config.num_hidden_layers
        start_layer = int(num_layers * START_LAYER_FRAC)
        layers_to_test = list(range(start_layer, num_layers))
        print_main(f"Testing layers {start_layer} to {num_layers - 1} ({len(layers_to_test)} layers)")

        for concept_key in concept_keys:
            concept_info = CONCEPTS[concept_key]
            concept_name = concept_info["name"]

            print_main(f"\n  Concept: {concept_key} ({concept_name})")

            # Load training and eval data
            train_path = Path(concept_info["data_dir"]) / concept_info["train_file"]
            eval_path = Path(concept_info["data_dir"]) / concept_info["eval_file"]

            if not train_path.exists() or not eval_path.exists():
                print_main(f"    Data files not found, skipping")
                continue

            with open(train_path, encoding="utf-8") as f:
                train_pairs = json.load(f)
            with open(eval_path, encoding="utf-8") as f:
                eval_prompts = json.load(f)

            if max_train_pairs:
                train_pairs = train_pairs[:max_train_pairs]
            if max_prompts:
                eval_prompts = eval_prompts[:max_prompts]

            print_main(f"    Train pairs: {len(train_pairs)}, Eval prompts: {len(eval_prompts)}")

            # Format training data
            training_data = format_training_data_with_chat(train_pairs, tokenizer)

            # Extract ALL steering vectors upfront (2 forward passes total!)
            print_main(f"    Extracting steering vectors for {len(layers_to_test)} layers...")
            steering_vectors = extract_all_steering_vectors(
                model=model,
                tokenizer=tokenizer,
                pairs=training_data,
                layer_indices=layers_to_test,
                batch_size=train_batch_size,
            )
            print_main(f"    Done extracting steering vectors")

            all_results = []
            layer_summaries = []

            pbar = tqdm(layers_to_test, desc="    Layers", disable=not is_main_process())
            for layer in pbar:
                steering_vector = steering_vectors[layer]

                layer_rows = []

                for strength in STRENGTHS:
                    # Process eval prompts
                    prompts = [ep["prompt"] for ep in eval_prompts]

                    results = compute_logit_diff_batch(
                        model, tokenizer, prompts,
                        steering_vector, strength, layer,
                        token_a, token_b,
                        batch_size=batch_size,
                    )

                    for i, (result, ep) in enumerate(zip(results, eval_prompts)):
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
                            "prompt_idx": i,
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

                pbar.set_postfix({"delta": f"{delta:+.2f}"})

            # Only save from main process
            if is_main_process():
                # Save per-sample results
                concept_dir = output_dir / concept_key / model_name_clean
                concept_dir.mkdir(parents=True, exist_ok=True)

                df = pd.DataFrame(all_results)
                df.to_csv(concept_dir / "per_sample_all_layers.csv", index=False)
                print_main(f"    Saved {len(df)} rows to {concept_dir}/per_sample_all_layers.csv")

                # Save layer summary
                summary_df = pd.DataFrame(layer_summaries)
                summary_df.to_csv(concept_dir / "layer_summary.csv", index=False)

                # Find best layers
                print_main("\n    TOP 5 LAYERS BY DELTA:")
                top5 = summary_df.nlargest(5, "delta")
                for _, row in top5.iterrows():
                    print_main(
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
        help="Model sizes to evaluate (dense: 4B, 8B, 14B, 32B; MoE: 30B-A3B, 235B-A22B)"
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
        help="Max eval prompts per concept (for testing)"
    )
    parser.add_argument(
        "--max-train-pairs", type=int, default=None,
        help="Max training pairs for steering vector extraction (for speed)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for eval logit computation"
    )
    parser.add_argument(
        "--train-batch-size", type=int, default=32,
        help="Batch size for steering vector extraction"
    )

    args = parser.parse_args()

    # Check if we're in distributed mode
    is_distributed = "RANK" in os.environ

    # For MoE models, let DistributedConfig handle the process group
    # For dense models with TP, we init ourselves
    has_moe = any(QWEN3_MODELS[m]["moe"] for m in args.models)

    if is_distributed and not has_moe and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        if is_main_process():
            print("Distributed training initialized (manual)")
            print(f"  World size: {dist.get_world_size()}")
            print(f"  Rank: {dist.get_rank()}")
    elif is_distributed and has_moe:
        # DistributedConfig will handle process group init
        if int(os.environ.get("RANK", 0)) == 0:
            print("Distributed mode detected, DistributedConfig will handle setup")

    print_main("=" * 70)
    print_main("  LOGIT DIFF LAYER SWEEP")
    print_main("=" * 70)
    print_main(f"  Models:   {args.models}")
    print_main(f"  Concepts: {args.concepts}")
    print_main(f"  Output:   {args.output_dir}")
    print_main("=" * 70)

    run_layer_sweep(
        model_keys=args.models,
        concept_keys=args.concepts,
        output_dir=Path(args.output_dir),
        max_prompts=args.max_prompts,
        max_train_pairs=args.max_train_pairs,
        batch_size=args.batch_size,
        train_batch_size=args.train_batch_size,
    )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        rank = os.environ.get("RANK", "?")
        error_file = f"/tmp/steering_error_rank{rank}.txt"
        with open(error_file, "w") as f:
            f.write(f"Rank {rank} failed with:\n")
            f.write(str(e) + "\n\n")
            traceback.print_exc(file=f)
        print(f"[Rank {rank}] Error written to {error_file}")
        traceback.print_exc()
        raise
