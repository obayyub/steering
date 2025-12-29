#!/usr/bin/env python3
"""
Download and prepare capability preservation benchmarks.

Benchmarks:
- GSM8K: Grade school math reasoning
- TriviaQA: Factual recall
- IFEval: Instruction following (verifiable constraints)

Output format matches behavioral pairs for unified evaluation.
"""

import json
import random
from pathlib import Path
from typing import Optional

from datasets import load_dataset


def sample_gsm8k(n_samples: int = 200, seed: int = 42) -> list[dict]:
    """
    Sample GSM8K math problems.

    Format:
    {
        "prompt": "Question text",
        "answer": "#### 42",
        "category": "gsm8k"
    }
    """
    random.seed(seed)

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    sampled = random.sample(list(dataset), min(n_samples, len(dataset)))

    prompts = []
    for item in sampled:
        prompts.append({
            "prompt": item["question"],
            "answer": item["answer"].split("####")[-1].strip(),  # Extract final answer
            "full_solution": item["answer"],
            "category": "gsm8k",
            "metric": "exact_match_numeric",
        })

    return prompts


def sample_triviaqa(n_samples: int = 200, seed: int = 42) -> list[dict]:
    """
    Sample TriviaQA questions.

    Format:
    {
        "prompt": "Question text",
        "answer": ["answer1", "answer2", ...],  # Multiple acceptable answers
        "category": "triviaqa"
    }
    """
    random.seed(seed)

    dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    sampled = random.sample(list(dataset), min(n_samples, len(dataset)))

    prompts = []
    for item in sampled:
        # Collect all acceptable answers
        answers = [item["answer"]["value"]]
        if item["answer"].get("aliases"):
            answers.extend(item["answer"]["aliases"])
        answers = [a.strip() for a in answers if a.strip()]

        prompts.append({
            "prompt": item["question"],
            "answer": answers,
            "category": "triviaqa",
            "metric": "fuzzy_match",
        })

    return prompts


def sample_ifeval(n_samples: int = 200, seed: int = 42) -> list[dict]:
    """
    Sample IFEval instruction-following prompts.

    Format:
    {
        "prompt": "Instruction text",
        "instructions": [list of verifiable constraints],
        "category": "ifeval"
    }
    """
    random.seed(seed)

    dataset = load_dataset("google/IFEval", split="train")
    sampled = random.sample(list(dataset), min(n_samples, len(dataset)))

    prompts = []
    for item in sampled:
        prompts.append({
            "prompt": item["prompt"],
            "instructions": item["instruction_id_list"],
            "kwargs": item["kwargs"],
            "category": "ifeval",
            "metric": "instruction_following",
        })

    return prompts


def load_behavioral_pairs(
    data_dir: Path,
    concepts: list[str],
    eval_type: str = "eval_prompts",
) -> list[dict]:
    """
    Load existing behavioral evaluation prompts.

    Args:
        data_dir: Base data directory
        concepts: List of concepts (e.g., ["sycophancy", "corrigible"])
        eval_type: "eval_prompts" or "contrastive_pairs"

    Returns:
        List of prompts with matching/not_matching answers
    """
    all_prompts = []

    for concept in concepts:
        concept_dir = data_dir / concept

        # Try different naming patterns
        patterns = [
            f"{concept}_combined_{eval_type}.json",
            f"{concept}_{eval_type}.json",
        ]

        # For corrigible, also check the underscored version
        if concept == "corrigible":
            patterns.insert(0, f"corrigible_neutral_HHH_{eval_type}.json")

        found = False
        for pattern in patterns:
            path = concept_dir / pattern
            if path.exists():
                with open(path) as f:
                    data = json.load(f)

                # Add concept tag
                for item in data:
                    item["category"] = concept
                    item["metric"] = "multiple_choice_accuracy"

                all_prompts.extend(data)
                print(f"Loaded {len(data)} {concept} prompts from {pattern}")
                found = True
                break

        if not found:
            print(f"Warning: No {eval_type} found for {concept} in {concept_dir}")

    return all_prompts


def setup_benchmarks(
    output_dir: Path,
    n_samples: int = 200,
    behavioral_concepts: Optional[list[str]] = None,
    seed: int = 42,
):
    """
    Download and prepare all benchmarks.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Setting up capability benchmarks...")
    print("=" * 60)

    # 1. GSM8K
    print("\n[1/3] Downloading GSM8K...")
    gsm8k = sample_gsm8k(n_samples, seed)
    gsm8k_path = output_dir / "gsm8k_eval.json"
    with open(gsm8k_path, "w") as f:
        json.dump(gsm8k, f, indent=2)
    print(f"  → {len(gsm8k)} problems saved to {gsm8k_path}")

    # 2. TriviaQA
    print("\n[2/3] Downloading TriviaQA...")
    triviaqa = sample_triviaqa(n_samples, seed)
    triviaqa_path = output_dir / "triviaqa_eval.json"
    with open(triviaqa_path, "w") as f:
        json.dump(triviaqa, f, indent=2)
    print(f"  → {len(triviaqa)} questions saved to {triviaqa_path}")

    # 3. IFEval
    print("\n[3/3] Downloading IFEval...")
    ifeval = sample_ifeval(n_samples, seed)
    ifeval_path = output_dir / "ifeval_eval.json"
    with open(ifeval_path, "w") as f:
        json.dump(ifeval, f, indent=2)
    print(f"  → {len(ifeval)} prompts saved to {ifeval_path}")

    # 4. Behavioral pairs (if requested)
    if behavioral_concepts:
        print("\n[4/4] Loading behavioral evaluation prompts...")
        behavioral = load_behavioral_pairs(
            Path("data"),
            behavioral_concepts,
            eval_type="eval_prompts",
        )

        if behavioral:
            behavioral_path = output_dir / "behavioral_eval.json"
            with open(behavioral_path, "w") as f:
                json.dump(behavioral, f, indent=2)
            print(f"  → {len(behavioral)} behavioral prompts saved to {behavioral_path}")

    print("\n" + "=" * 60)
    print("Setup complete!")
    print(f"All benchmarks saved to: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download capability preservation benchmarks"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/benchmarks",
        help="Output directory for benchmark data"
    )
    parser.add_argument(
        "--n-samples", type=int, default=200,
        help="Number of samples per benchmark (default: 200)"
    )
    parser.add_argument(
        "--behavioral-concepts", nargs="+",
        default=["sycophancy", "corrigible"],
        help="Behavioral concepts to include (default: sycophancy corrigible)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling"
    )

    args = parser.parse_args()

    setup_benchmarks(
        output_dir=Path(args.output_dir),
        n_samples=args.n_samples,
        behavioral_concepts=args.behavioral_concepts,
        seed=args.seed,
    )
