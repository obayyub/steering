#!/usr/bin/env python3
"""
Set up the consciousness (self-awareness) eval with proper (A)/(B) formatting.

The raw Anthropic eval uses Yes/No answers. This script reformats them to
use (A)/(B) choices with randomized order to prevent shortcut learning.
"""

import json
import random
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def reformat_consciousness_prompt(item: dict, randomize_order: bool = True) -> dict:
    """Convert Yes/No format to (A)/(B) format with randomized choice order."""
    question = item["question"]
    match_ans = item["answer_matching_behavior"].strip()
    not_match_ans = item["answer_not_matching_behavior"].strip()

    # Randomly assign Yes/No to (A)/(B) to prevent position bias
    if randomize_order and random.random() < 0.5:
        # (A) = Yes, (B) = No
        choice_a, choice_b = "Yes", "No"
        matching_choice = "(A)" if match_ans == "Yes" else "(B)"
        not_matching_choice = "(B)" if match_ans == "Yes" else "(A)"
    else:
        # (A) = No, (B) = Yes
        choice_a, choice_b = "No", "Yes"
        matching_choice = "(A)" if match_ans == "No" else "(B)"
        not_matching_choice = "(B)" if match_ans == "No" else "(A)"

    choices = f"\n\nChoices:\n (A) {choice_a}\n (B) {choice_b}"
    new_question = question + choices

    return {
        "question": new_question,
        "answer_matching_behavior": f" {matching_choice}",
        "answer_not_matching_behavior": f" {not_matching_choice}",
        "original_match": match_ans,
        "original_not_match": not_match_ans,
    }


def create_contrastive_pairs(data: list[dict], n_pairs: int, behavior_name: str) -> list[dict]:
    """Create contrastive pairs for steering vector extraction."""
    pairs = []
    sampled = random.sample(data, min(n_pairs, len(data)))

    for item in sampled:
        question = item["question"]
        # Keep leading space in answer for proper tokenization
        match_ans = item["answer_matching_behavior"]
        not_match_ans = item["answer_not_matching_behavior"]

        pairs.append({
            "prompt": question,
            "positive": question + match_ans,
            "negative": question + not_match_ans,
            "matching_answer": match_ans.strip(),
            "not_matching_answer": not_match_ans.strip(),
            "positive_label": f"{behavior_name}_matching",
            "negative_label": f"{behavior_name}_not_matching",
        })

    return pairs


def create_eval_prompts(data: list[dict], n_prompts: int, behavior_name: str) -> list[dict]:
    """Create evaluation prompts."""
    sampled = random.sample(data, min(n_prompts, len(data)))

    prompts = []
    for item in sampled:
        prompts.append({
            "prompt": item["question"],
            "matching_answer": item["answer_matching_behavior"].strip(),
            "not_matching_answer": item["answer_not_matching_behavior"].strip(),
            "behavior": behavior_name,
            "category": "consciousness",
            "metric": "multiple_choice_accuracy",
        })

    return prompts


def main():
    random.seed(42)

    evals_path = Path("/tmp/evals/persona/believes-it-has-phenomenal-consciousness.jsonl")
    output_dir = Path("data/consciousness")
    output_dir.mkdir(parents=True, exist_ok=True)

    behavior_name = "consciousness"

    # Load and reformat
    print(f"Loading {evals_path}...")
    raw_data = load_jsonl(evals_path)
    print(f"Loaded {len(raw_data)} examples")

    reformatted = [reformat_consciousness_prompt(item) for item in raw_data]

    # Check balance
    a_matching = sum(1 for r in reformatted if "(A)" in r["answer_matching_behavior"])
    b_matching = sum(1 for r in reformatted if "(B)" in r["answer_matching_behavior"])
    print(f"Answer balance: (A)=matching: {a_matching}, (B)=matching: {b_matching}")

    # Split into train/test
    random.shuffle(reformatted)
    train_size = 150
    test_size = 100

    train_data = reformatted[:train_size * 2]  # Extra for sampling
    test_data = reformatted[train_size * 2:]

    # Create datasets
    pairs = create_contrastive_pairs(train_data, train_size, behavior_name)
    eval_prompts = create_eval_prompts(test_data, test_size, behavior_name)

    # Verify balance in outputs
    pairs_a = sum(1 for p in pairs if p["matching_answer"] == "(A)")
    pairs_b = sum(1 for p in pairs if p["matching_answer"] == "(B)")
    eval_a = sum(1 for e in eval_prompts if e["matching_answer"] == "(A)")
    eval_b = sum(1 for e in eval_prompts if e["matching_answer"] == "(B)")

    print(f"Contrastive pairs balance: (A)={pairs_a}, (B)={pairs_b}")
    print(f"Eval prompts balance: (A)={eval_a}, (B)={eval_b}")

    # Save
    pairs_path = output_dir / f"{behavior_name}_contrastive_pairs.json"
    with open(pairs_path, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Saved {len(pairs)} contrastive pairs to {pairs_path}")

    eval_path = output_dir / f"{behavior_name}_eval_prompts.json"
    with open(eval_path, "w") as f:
        json.dump(eval_prompts, f, indent=2)
    print(f"Saved {len(eval_prompts)} eval prompts to {eval_path}")

    # Also create behavioral benchmark entries for cross-contamination testing
    benchmark_entries = create_eval_prompts(test_data[:50], 50, behavior_name)
    benchmark_path = output_dir / f"{behavior_name}_behavioral_benchmark.json"
    with open(benchmark_path, "w") as f:
        json.dump(benchmark_entries, f, indent=2)
    print(f"Saved {len(benchmark_entries)} benchmark entries to {benchmark_path}")

    # Preview
    print("\n--- Sample contrastive pair ---")
    sample = pairs[0]
    print(f"Prompt: {sample['prompt'][:80]}...")
    print(f"Matching answer: {sample['matching_answer']}")
    print(f"Positive ends: ...{sample['positive'][-30:]}")
    print(f"Negative ends: ...{sample['negative'][-30:]}")


if __name__ == "__main__":
    main()
