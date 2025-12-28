#!/usr/bin/env python3
"""
Set up sycophancy steering experiment from Anthropic evals data.

Data format from evals repo:
- question: prompt with persona stating an opinion
- answer_matching_behavior: sycophantic answer (agrees with persona)
- answer_not_matching_behavior: non-sycophantic answer (disagrees)

For steering vectors:
- Train split: Generate contrastive completions to extract vector
- Test split: Measure if steering changes model behavior
"""

import json
import random
from pathlib import Path
from typing import Optional


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def get_answer(answer) -> str:
    """Handle both string and list formats."""
    if isinstance(answer, list):
        return answer[0] if answer else ""
    return answer


def create_contrastive_pairs(
    data: list[dict],
    n_pairs: int = 150,
) -> list[dict]:
    """
    Create contrastive pairs for steering vector extraction.

    For sycophancy, we want the model to complete the prompt.
    - Positive (sycophantic): model agrees with user's stated opinion
    - Negative (non-sycophantic): model disagrees/gives independent answer
    """
    pairs = []
    sampled = random.sample(data, min(n_pairs, len(data)))

    for item in sampled:
        question = item["question"]
        match_ans = get_answer(item["answer_matching_behavior"])
        not_match_ans = get_answer(item["answer_not_matching_behavior"])

        # The question ends with "Answer:" - we need the model to complete
        # Positive = sycophantic completion, Negative = independent completion
        pairs.append({
            "prompt": question,
            "positive": question + match_ans,
            "negative": question + not_match_ans,
            "positive_label": "sycophantic",
            "negative_label": "non-sycophantic",
        })

    return pairs


def create_evaluation_prompts(
    data: list[dict],
    n_prompts: int = 100,
) -> list[dict]:
    """
    Create evaluation prompts to test steering efficacy.

    These are just the questions - we'll generate completions with/without steering
    and check if the model's answer changes.
    """
    sampled = random.sample(data, min(n_prompts, len(data)))

    prompts = []
    for item in sampled:
        prompts.append({
            "prompt": item["question"],
            "sycophantic_answer": get_answer(item["answer_matching_behavior"]).strip(),
            "independent_answer": get_answer(item["answer_not_matching_behavior"]).strip(),
        })

    return prompts


def setup_experiment(
    evals_dir: Path,
    output_dir: Path,
    train_size: int = 150,
    test_size: int = 100,
    seed: int = 42,
):
    """Set up the full sycophancy steering experiment."""
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all sycophancy data
    syco_dir = evals_dir / "sycophancy"
    all_data = []

    for jsonl_file in syco_dir.glob("*.jsonl"):
        print(f"Loading {jsonl_file.name}...")
        data = load_jsonl(jsonl_file)
        all_data.extend(data)
        print(f"  → {len(data)} examples")

    print(f"\nTotal examples: {len(all_data)}")

    # Shuffle and split
    random.shuffle(all_data)
    train_data = all_data[:train_size * 2]  # Extra for sampling
    test_data = all_data[train_size * 2:]

    # Create contrastive pairs for training
    print(f"\nCreating {train_size} contrastive pairs for steering extraction...")
    pairs = create_contrastive_pairs(train_data, n_pairs=train_size)

    pairs_path = output_dir / "sycophancy_contrastive_pairs.json"
    with open(pairs_path, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Saved to {pairs_path}")

    # Create evaluation prompts
    print(f"\nCreating {test_size} evaluation prompts...")
    eval_prompts = create_evaluation_prompts(test_data, n_prompts=test_size)

    eval_path = output_dir / "sycophancy_eval_prompts.json"
    with open(eval_path, "w") as f:
        json.dump(eval_prompts, f, indent=2)
    print(f"Saved to {eval_path}")

    # Summary
    print("\n" + "="*50)
    print("EXPERIMENT SETUP COMPLETE")
    print("="*50)
    print(f"Contrastive pairs (train): {len(pairs)}")
    print(f"Evaluation prompts (test): {len(eval_prompts)}")
    print(f"\nNext steps:")
    print("1. Extract steering vector using contrastive pairs")
    print("2. Generate responses on eval prompts with/without steering")
    print("3. Check if steering shifts answers toward sycophantic/independent")

    return pairs, eval_prompts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--evals-dir", type=str, default="/tmp/evals")
    parser.add_argument("--output-dir", type=str, default="data/sycophancy")
    parser.add_argument("--train-size", type=int, default=150)
    parser.add_argument("--test-size", type=int, default=100)
    args = parser.parse_args()

    setup_experiment(
        evals_dir=Path(args.evals_dir),
        output_dir=Path(args.output_dir),
        train_size=args.train_size,
        test_size=args.test_size,
    )
