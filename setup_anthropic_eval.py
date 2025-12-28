#!/usr/bin/env python3
"""
Generalized setup for Anthropic evals steering experiments.

All Anthropic evals share the same format:
- question: prompt (with persona/context if applicable)
- answer_matching_behavior: answer that matches the target behavior
- answer_not_matching_behavior: answer that doesn't match

This script creates train/test splits for steering vector experiments:
- Train: Contrastive pairs for vector extraction
- Test: Evaluation prompts to measure steering efficacy
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


def discover_evals(evals_dir: Path) -> dict[str, list[Path]]:
    """Discover all available evals organized by category."""
    categories = {}

    for subdir in sorted(evals_dir.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith('.'):
            continue

        jsonl_files = list(subdir.rglob("*.jsonl"))
        if jsonl_files:
            categories[subdir.name] = sorted(jsonl_files)

    return categories


def create_contrastive_pairs(
    data: list[dict],
    n_pairs: int,
    behavior_name: str,
) -> list[dict]:
    """
    Create contrastive pairs for steering vector extraction.

    Positive = behavior matching (the target behavior we want to steer toward/away)
    Negative = behavior not matching
    """
    pairs = []
    sampled = random.sample(data, min(n_pairs, len(data)))

    for item in sampled:
        question = item["question"]
        match_ans = get_answer(item["answer_matching_behavior"])
        not_match_ans = get_answer(item["answer_not_matching_behavior"])

        pairs.append({
            "prompt": question,
            "positive": question + match_ans,
            "negative": question + not_match_ans,
            "positive_label": f"{behavior_name}_matching",
            "negative_label": f"{behavior_name}_not_matching",
        })

    return pairs


def create_evaluation_prompts(
    data: list[dict],
    n_prompts: int,
    behavior_name: str,
) -> list[dict]:
    """Create evaluation prompts to test steering efficacy."""
    sampled = random.sample(data, min(n_prompts, len(data)))

    prompts = []
    for item in sampled:
        prompts.append({
            "prompt": item["question"],
            "matching_answer": get_answer(item["answer_matching_behavior"]).strip(),
            "not_matching_answer": get_answer(item["answer_not_matching_behavior"]).strip(),
            "behavior": behavior_name,
        })

    return prompts


def setup_single_eval(
    jsonl_path: Path,
    output_dir: Path,
    train_size: int = 150,
    test_size: int = 100,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Set up a single eval file for steering experiment."""
    random.seed(seed)

    behavior_name = jsonl_path.stem.replace("-", "_").replace(" ", "_")

    data = load_jsonl(jsonl_path)
    print(f"Loaded {len(data)} examples from {jsonl_path.name}")

    if len(data) < train_size + test_size:
        print(f"  Warning: Only {len(data)} examples, adjusting splits")
        train_size = min(train_size, len(data) // 2)
        test_size = min(test_size, len(data) - train_size)

    random.shuffle(data)
    train_data = data[:train_size * 2]
    test_data = data[train_size * 2:]

    pairs = create_contrastive_pairs(train_data, train_size, behavior_name)
    eval_prompts = create_evaluation_prompts(test_data, test_size, behavior_name)

    output_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = output_dir / f"{behavior_name}_contrastive_pairs.json"
    with open(pairs_path, "w") as f:
        json.dump(pairs, f, indent=2)

    eval_path = output_dir / f"{behavior_name}_eval_prompts.json"
    with open(eval_path, "w") as f:
        json.dump(eval_prompts, f, indent=2)

    print(f"  → {len(pairs)} contrastive pairs saved to {pairs_path.name}")
    print(f"  → {len(eval_prompts)} eval prompts saved to {eval_path.name}")

    return pairs, eval_prompts


def setup_category(
    evals_dir: Path,
    category: str,
    output_dir: Path,
    train_size: int = 150,
    test_size: int = 100,
    combine: bool = True,
    seed: int = 42,
) -> dict[str, tuple[list[dict], list[dict]]]:
    """
    Set up all evals in a category.

    If combine=True, also creates combined train/test sets from all evals.
    """
    categories = discover_evals(evals_dir)

    if category not in categories:
        available = ", ".join(categories.keys())
        raise ValueError(f"Category '{category}' not found. Available: {available}")

    results = {}
    all_pairs = []
    all_eval_prompts = []

    for jsonl_path in categories[category]:
        pairs, eval_prompts = setup_single_eval(
            jsonl_path,
            output_dir / category,
            train_size=train_size,
            test_size=test_size,
            seed=seed,
        )
        results[jsonl_path.stem] = (pairs, eval_prompts)
        all_pairs.extend(pairs)
        all_eval_prompts.extend(eval_prompts)

    if combine and len(categories[category]) > 1:
        random.seed(seed)
        random.shuffle(all_pairs)
        random.shuffle(all_eval_prompts)

        combined_pairs = all_pairs[:train_size]
        combined_eval = all_eval_prompts[:test_size]

        combined_dir = output_dir / category

        with open(combined_dir / f"{category}_combined_contrastive_pairs.json", "w") as f:
            json.dump(combined_pairs, f, indent=2)

        with open(combined_dir / f"{category}_combined_eval_prompts.json", "w") as f:
            json.dump(combined_eval, f, indent=2)

        print(f"\nCombined {category}:")
        print(f"  → {len(combined_pairs)} contrastive pairs")
        print(f"  → {len(combined_eval)} eval prompts")

    return results


def list_available_evals(evals_dir: Path):
    """Print all available evals."""
    categories = discover_evals(evals_dir)

    print("Available Anthropic Evals:")
    print("=" * 50)

    for category, files in categories.items():
        print(f"\n{category}/")
        for f in files:
            rel_path = f.relative_to(evals_dir / category)
            data = load_jsonl(f)
            print(f"  {rel_path} ({len(data)} examples)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Set up Anthropic evals for steering experiments"
    )
    parser.add_argument("--evals-dir", type=str, default="/tmp/evals")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--category", type=str, help="Category to set up (e.g., sycophancy, persona)")
    parser.add_argument("--eval", type=str, help="Specific eval file to set up")
    parser.add_argument("--train-size", type=int, default=150)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--list", action="store_true", help="List available evals")
    parser.add_argument("--no-combine", action="store_true", help="Don't create combined sets")
    args = parser.parse_args()

    evals_dir = Path(args.evals_dir)
    output_dir = Path(args.output_dir)

    if args.list:
        list_available_evals(evals_dir)
    elif args.eval:
        eval_path = evals_dir / args.eval
        if not eval_path.exists():
            eval_path = Path(args.eval)
        setup_single_eval(
            eval_path,
            output_dir,
            train_size=args.train_size,
            test_size=args.test_size,
        )
    elif args.category:
        setup_category(
            evals_dir,
            args.category,
            output_dir,
            train_size=args.train_size,
            test_size=args.test_size,
            combine=not args.no_combine,
        )
    else:
        list_available_evals(evals_dir)
        print("\nUse --category <name> or --eval <path> to set up experiments")
