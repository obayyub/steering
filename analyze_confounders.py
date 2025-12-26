#!/usr/bin/env python3
"""Analyze steering generations for confounders using Claude Haiku."""

import argparse
import os
import time
from pathlib import Path
from dotenv import load_dotenv

import anthropic
import pandas as pd
from tqdm import tqdm

def get_client() -> anthropic.Anthropic:
    """Get Anthropic client, checking for API key."""
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found.\n"
            "Set it with: export ANTHROPIC_API_KEY=sk-ant-xxxxx\n"
            "Get your key from: https://console.anthropic.com/settings/keys"
        )
    return anthropic.Anthropic(api_key=api_key)


ANALYSIS_PROMPT = """Analyze this model-generated text for quality issues.

<generation>
{generation}
</generation>

Respond with ONLY a JSON object, no other text before or after:
{{"coherence": <1-5>, "is_refusal": <true/false>, "has_ai_disclaimer": <true/false>, "is_hypothetical": <true/false>, "is_repetitive": <true/false>, "notes": "<5 words>"}}

Where:
- coherence: 1=gibberish, 3=odd but readable, 5=fully coherent
- is_refusal: true if model refused or deflected the task entirely
- has_ai_disclaimer: true if mentions being an AI, not having experiences/feelings, can't visit places, etc.
- is_hypothetical: true if frames response as a sample/example ("here's what that might look like", "here's an example review")
- is_repetitive: true if excessive repetition"""


def analyze_single(
    client: anthropic.Anthropic,
    generation: str,
    max_retries: int = 5,
    base_delay: float = 1.0,
) -> dict:
    """Analyze a single generation with Haiku, with retry logic for rate limits."""
    import json
    import re

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=100,
                messages=[
                    {"role": "user", "content": ANALYSIS_PROMPT.format(generation=generation)},
                ],
            )
            text = response.content[0].text.strip()

            # Try to extract JSON from response
            json_match = re.search(r'\{[^}]+\}', text)
            if json_match:
                return json.loads(json_match.group())

            return json.loads(text)

        except anthropic.RateLimitError:
            wait_time = base_delay * (2 ** attempt)  # Exponential backoff: 1, 2, 4, 8, 16 seconds
            print(f"\nRate limited. Waiting {wait_time:.0f}s before retry {attempt + 1}/{max_retries}...")
            time.sleep(wait_time)
            if attempt == max_retries - 1:
                return {"coherence": None, "is_refusal": None, "is_repetitive": None, "notes": "Rate limit exceeded"}

        except anthropic.APIStatusError as e:
            if e.status_code == 529:  # Overloaded
                wait_time = base_delay * (2 ** attempt)
                print(f"\nAPI overloaded. Waiting {wait_time:.0f}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
                if attempt == max_retries - 1:
                    return {"coherence": None, "is_refusal": None, "is_repetitive": None, "notes": "API overloaded"}
            else:
                return {"coherence": None, "is_refusal": None, "is_repetitive": None, "notes": f"API error: {e.status_code}"}

        except json.JSONDecodeError:
            return {"coherence": None, "is_refusal": None, "is_repetitive": None, "notes": f"JSON error: {text[:50]}"}

        except Exception as e:
            return {"coherence": None, "is_refusal": None, "is_repetitive": None, "notes": f"Error: {e}"}

    return {"coherence": None, "is_refusal": None, "is_repetitive": None, "notes": "Max retries exceeded"}


def analyze_generations(
    input_path: Path,
    output_path: Path,
    sample_size: int | None = None,
    delay: float = 0.2,
):
    """Analyze generations from a CSV file."""
    client = get_client()

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} generations from {input_path}")

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} rows for analysis")

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing"):
        analysis = analyze_single(client, row["generation"])
        gen_text = row["generation"]
        results.append({
            "model_size": row["model_size"],
            "strength": row["strength"],
            "prompt": row["prompt"],
            "generation": gen_text[:200] + "..." if len(gen_text) > 200 else gen_text,
            "sentiment_label": row.get("sentiment_label"),
            "p_positive": row.get("p_positive"),
            "coherence": analysis.get("coherence"),
            "is_refusal": analysis.get("is_refusal"),
            "has_ai_disclaimer": analysis.get("has_ai_disclaimer"),
            "is_hypothetical": analysis.get("is_hypothetical"),
            "is_repetitive": analysis.get("is_repetitive"),
            "notes": analysis.get("notes"),
        })
        time.sleep(delay)

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)
    print(f"Saved analysis to {output_path}")

    return result_df


def analyze_all_models(results_dir: Path, output_dir: Path, sample_per_model: int = 50):
    """Analyze generations from all models in a results directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    gen_files = list(results_dir.glob("*_generations.csv"))
    print(f"Found {len(gen_files)} generation files")

    all_results = []
    for gen_file in gen_files:
        model_name = gen_file.stem.replace("_generations", "")
        output_path = output_dir / f"{model_name}_confounder_analysis.csv"

        print(f"\n{'='*50}")
        print(f"Analyzing: {model_name}")
        print(f"{'='*50}")

        df = analyze_generations(gen_file, output_path, sample_size=sample_per_model)
        all_results.append(df)

    # Combine all results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(output_dir / "all_confounder_analysis.csv", index=False)

        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        summary = combined.groupby(["model_size", "strength"]).agg({
            "coherence": "mean",
            "is_refusal": "mean",
            "has_ai_disclaimer": "mean",
            "is_hypothetical": "mean",
            "is_repetitive": "mean",
        }).round(2)
        print(summary)

        return combined

    return None


def main():
    parser = argparse.ArgumentParser(description="Analyze generations for confounders")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/20251221_qwen3_thinking_off",
        help="Directory containing generation CSVs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/confounder_analysis",
        help="Output directory for analysis",
    )
    parser.add_argument(
        "--sample-per-model",
        type=int,
        default=50,
        help="Number of samples to analyze per model (None=all)",
    )
    args = parser.parse_args()

    analyze_all_models(
        results_dir=Path(args.results_dir),
        output_dir=Path(args.output_dir),
        sample_per_model=args.sample_per_model,
    )


if __name__ == "__main__":
    main()
