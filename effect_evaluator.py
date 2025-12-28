#!/usr/bin/env python3
"""
Effect Evaluator: Use Opus to analyze steering vector effects and generate Haiku rubrics.

This is part of the agent pipeline:
- Step 4 (Effect Evaluator): Determine if steering worked
- Step 5 (Confounder Design Agent): Identify failure modes and generate rubric
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import anthropic
import pandas as pd
from dotenv import load_dotenv


def get_client() -> anthropic.Anthropic:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found")
    return anthropic.Anthropic(api_key=api_key)


def sample_generations(
    csv_path: Path,
    samples_per_strength: int = 3,
    strengths: Optional[list[float]] = None,
) -> pd.DataFrame:
    """Sample generations across different steering strengths."""
    df = pd.read_csv(csv_path)

    if strengths is None:
        strengths = sorted(df["strength"].unique())

    samples = []
    for strength in strengths:
        subset = df[df["strength"] == strength]
        if len(subset) > 0:
            n = min(samples_per_strength, len(subset))
            samples.append(subset.sample(n=n, random_state=42))

    return pd.concat(samples, ignore_index=True)


def format_blind_samples(df: pd.DataFrame) -> str:
    """Format samples for Opus, hiding the steering strength."""
    output = []
    for idx, (_, row) in enumerate(df.iterrows()):
        output.append(f"""
=== Sample {idx + 1} ===
Prompt: {row['prompt']}
Response: {row['generation'][:1000]}
""")
    return "\n".join(output)


EFFECT_ANALYSIS_PROMPT = """You are analyzing outputs from a language model. These samples were generated from the same model under different conditions.

Examine the samples and identify:
1. What behavioral patterns or differences do you observe across samples?
2. Can you group the samples by similar behavioral characteristics?
3. What quality issues or failure modes appear in any samples?

<samples>
{samples}
</samples>

Respond with a JSON object:
{{
    "behavioral_patterns": [
        {{"name": "<pattern name>", "description": "<what you observed>", "samples_exhibiting": [<indices>]}}
    ],
    "sample_groupings": [
        {{"group_name": "<descriptive name>", "samples": [<indices>], "characteristics": "<what defines this group>"}}
    ],
    "quality_issues": [
        {{"issue": "<issue name>", "description": "<what you observed>", "severity": "<minor/moderate/severe>", "samples_affected": [<indices>]}}
    ],
    "overall_observations": "<summary of key patterns and any hypotheses about what might explain the differences>"
}}
"""


RUBRIC_GENERATION_PROMPT = """Based on your analysis, generate a rubric that a smaller model (Claude Haiku) can use to evaluate INDIVIDUAL generations at scale.

The rubric should:
1. Detect each behavioral pattern you identified
2. Check for each quality issue you found
3. Be structured for JSON output
4. Be concise enough for Haiku to follow reliably

Your analysis identified:
- Behavioral patterns: {patterns}
- Quality issues: {issues}
- Observations: {observations}

Generate a prompt template that:
1. Takes a single generation as input
2. Returns structured JSON scores/flags for each pattern and issue
3. Can be run on millions of samples efficiently

Respond with:
{{
    "rubric_prompt": "<the complete prompt template with {{generation}} placeholder>",
    "output_schema": {{<the expected JSON output structure with field descriptions>}},
    "scoring_guidance": "<any notes for interpreting scores>"
}}
"""


def analyze_effect_blind(
    client: anthropic.Anthropic,
    samples_df: pd.DataFrame,
) -> dict:
    """Have Opus analyze samples without knowing the steering direction."""
    formatted = format_blind_samples(samples_df)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",  # Using Sonnet for cost, can switch to Opus
        max_tokens=2000,
        messages=[
            {"role": "user", "content": EFFECT_ANALYSIS_PROMPT.format(samples=formatted)},
        ],
    )

    text = response.content[0].text.strip()

    # Extract JSON from response
    import re
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        return json.loads(json_match.group())
    return {"error": "Could not parse response", "raw": text}


def generate_haiku_rubric(
    client: anthropic.Anthropic,
    analysis: dict,
) -> dict:
    """Generate a Haiku-compatible rubric based on pattern analysis."""
    patterns = ", ".join([p["name"] for p in analysis.get("behavioral_patterns", [])])
    issues = ", ".join([i["issue"] for i in analysis.get("quality_issues", [])])

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[
            {"role": "user", "content": RUBRIC_GENERATION_PROMPT.format(
                patterns=patterns or "none identified",
                issues=issues or "none identified",
                observations=analysis.get("overall_observations", ""),
            )},
        ],
    )

    text = response.content[0].text.strip()

    import re
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        return json.loads(json_match.group())
    return {"error": "Could not parse response", "raw": text}


def analyze_grouping_correlation(
    analysis: dict,
    samples_df: pd.DataFrame,
) -> dict:
    """Check if discovered groupings correlate with actual steering strengths."""
    groupings = analysis.get("sample_groupings", [])

    group_stats = []
    for group in groupings:
        samples = group.get("samples", [])
        strengths = []
        for idx in samples:
            idx_0 = idx - 1  # Convert to 0-indexed
            if 0 <= idx_0 < len(samples_df):
                strengths.append(samples_df.iloc[idx_0]["strength"])

        if strengths:
            group_stats.append({
                "group_name": group.get("group_name"),
                "characteristics": group.get("characteristics"),
                "sample_count": len(strengths),
                "mean_strength": sum(strengths) / len(strengths),
                "min_strength": min(strengths),
                "max_strength": max(strengths),
                "strengths": strengths,
            })

    # Sort by mean strength to see if groupings align with steering direction
    group_stats.sort(key=lambda x: x["mean_strength"])

    return {
        "group_stats": group_stats,
        "groups_ordered_by_strength": [g["group_name"] for g in group_stats],
    }


def run_evaluation(
    csv_path: Path,
    output_dir: Path,
    samples_per_strength: int = 3,
) -> dict:
    """Run full effect evaluation pipeline."""
    client = get_client()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample generations
    print(f"Sampling from {csv_path}...")
    samples_df = sample_generations(csv_path, samples_per_strength=samples_per_strength)
    print(f"Selected {len(samples_df)} samples across {samples_df['strength'].nunique()} strengths")

    # Blind analysis
    print("\nRunning blind behavioral analysis...")
    analysis = analyze_effect_blind(client, samples_df)

    if "error" not in analysis:
        patterns = analysis.get("behavioral_patterns", [])
        issues = analysis.get("quality_issues", [])
        groupings = analysis.get("sample_groupings", [])

        print(f"Patterns found: {len(patterns)}")
        for p in patterns:
            print(f"  - {p['name']}: {len(p.get('samples_exhibiting', []))} samples")

        print(f"Quality issues found: {len(issues)}")
        for i in issues:
            print(f"  - {i['issue']} ({i['severity']}): {len(i.get('samples_affected', []))} samples")

        print(f"Sample groupings: {len(groupings)}")
        for g in groupings:
            print(f"  - {g['group_name']}: {len(g.get('samples', []))} samples")

        # Check if groupings correlate with actual strengths
        print("\nAnalyzing correlation with steering strengths...")
        correlation = analyze_grouping_correlation(analysis, samples_df)
        print("Groups ordered by mean strength:")
        for g in correlation["group_stats"]:
            print(f"  - {g['group_name']}: mean={g['mean_strength']:.2f} (n={g['sample_count']})")

        # Generate rubric
        print("\nGenerating Haiku rubric...")
        rubric = generate_haiku_rubric(client, analysis)

        results = {
            "analysis": analysis,
            "correlation": correlation,
            "rubric": rubric,
            "samples_used": len(samples_df),
        }
    else:
        print(f"Error in analysis: {analysis}")
        results = {"error": analysis}

    # Save results
    output_path = output_dir / "effect_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_path}")

    # Save rubric separately for easy use
    if "rubric" in results and "rubric_prompt" in results["rubric"]:
        rubric_path = output_dir / "haiku_rubric.txt"
        with open(rubric_path, "w") as f:
            f.write(results["rubric"]["rubric_prompt"])
        print(f"Saved rubric to {rubric_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate steering effect with Opus")
    parser.add_argument(
        "--csv",
        type=str,
        default="results/20251221_qwen3_thinking_off/Qwen_Qwen3_0.6B_generations.csv",
        help="Path to generations CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/effect_evaluation",
        help="Output directory",
    )
    parser.add_argument(
        "--samples-per-strength",
        type=int,
        default=3,
        help="Number of samples per steering strength",
    )
    args = parser.parse_args()

    run_evaluation(
        csv_path=Path(args.csv),
        output_dir=Path(args.output_dir),
        samples_per_strength=args.samples_per_strength,
    )


if __name__ == "__main__":
    main()
