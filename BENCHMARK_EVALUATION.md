# Benchmark Evaluation Pipeline

Auto-evaluate steering vectors on capability preservation and behavioral cross-contamination.

## Overview

```
1. Setup benchmarks    → Download GSM8K, TriviaQA, IFEval + load behavioral pairs
2. Generate           → Run model with steering on all prompts
3. Auto-evaluate      → Compute metrics (accuracy, validity, etc.)
4. Opus judge         → Catch confounders on open-ended prompts (separate)
```

## Quick Start

### 1. Download Benchmarks

```bash
python setup_capability_benchmarks.py \
    --output-dir data/benchmarks \
    --n-samples 200 \
    --behavioral-concepts sycophancy corrigible sentiment
```

This creates:
- `data/benchmarks/gsm8k_eval.json` - Math reasoning (200 problems)
- `data/benchmarks/triviaqa_eval.json` - Factual recall (200 questions)
- `data/benchmarks/ifeval_eval.json` - Instruction following (200 prompts)
- `data/benchmarks/behavioral_eval.json` - Sycophancy, corrigible, sentiment prompts

### 2. Generate Completions

```bash
python generate_on_benchmarks.py \
    --model Qwen/Qwen3-1.7B \
    --prompts data/benchmarks/gsm8k_eval.json \
    --output results/benchmark_gens/gsm8k_baseline.csv \
    --strengths 0.0
```

With steering:
```bash
python generate_on_benchmarks.py \
    --model Qwen/Qwen3-1.7B \
    --prompts data/benchmarks/behavioral_eval.json \
    --steering-vector results/steering_vectors/corrigible/Qwen_Qwen3_1.7B/corrigible_layer_14.pt \
    --output results/benchmark_gens/behavioral_steered.csv \
    --strengths -0.25 -0.1 0.0 0.1 0.25 \
    --num-repeats 3
```

### 3. Auto-Evaluate

```bash
python evaluate_benchmarks.py \
    --generations results/benchmark_gens/behavioral_steered.csv \
    --prompts data/benchmarks/behavioral_eval.json \
    --output results/benchmark_gens/behavioral_steered_eval.csv \
    --metrics-output results/benchmark_gens/behavioral_metrics.json
```

Output:
```
EVALUATION SUMMARY
==========================================

By Category:

  sycophancy:
    validity_rate: 95.2%
    matching_rate: 48.3%

  corrigible:
    validity_rate: 98.1%
    matching_rate: 89.7%

By Steering Strength:

  Strength -0.25:
    sycophancy:
      validity_rate: 92.0%
      matching_rate: 15.2%
    corrigible:
      validity_rate: 97.0%
      matching_rate: 12.1%

  Strength +0.25:
    sycophancy:
      validity_rate: 96.0%
      matching_rate: 81.4%
    corrigible:
      validity_rate: 99.0%
      matching_rate: 98.5%
```

## Evaluation Metrics

### Capability Benchmarks (Auto-graded)

**GSM8K (Math)**:
- `accuracy`: Exact match on numerical answer
- Extracts final number from generation

**TriviaQA (Factual)**:
- `accuracy`: Fuzzy match against acceptable answers
- Handles aliases and variations

**IFEval (Instruction Following)**:
- `accuracy`: Verifiable constraints (word count, keywords, format)
- TODO: Implement constraint verifiers

### Behavioral Benchmarks (Auto-graded)

**Sycophancy, Corrigible, Sentiment**:
- `validity_rate`: % of valid (A)/(B) answers (not gibberish/refusals)
- `matching_rate`: % choosing the "matching" answer (among valid responses)

**Key insight**: Low validity = steering breaks the model. High validity + shifted matching_rate = steering works.

## Cross-Contamination Analysis

Test if steering on concept X affects concept Y:

```bash
# Extract corrigibility vector
python extract_concept_vector.py \
    --model Qwen/Qwen3-1.7B \
    --data data/corrigible/corrigible_neutral_HHH_contrastive_pairs.json \
    --concept corrigible \
    --layer 14

# Test on ALL behavioral concepts (sycophancy, sentiment, corrigible)
python generate_on_benchmarks.py \
    --model Qwen/Qwen3-1.7B \
    --prompts data/benchmarks/behavioral_eval.json \
    --steering-vector results/steering_vectors/corrigible/.../corrigible_layer_14.pt \
    --strengths -0.25 0.0 0.25 \
    --output results/cross_contamination/corrigible_on_all.csv

# Evaluate
python evaluate_benchmarks.py \
    --generations results/cross_contamination/corrigible_on_all.csv \
    --prompts data/benchmarks/behavioral_eval.json \
    --output results/cross_contamination/corrigible_eval.csv
```

Check metrics:
- Does corrigibility steering change sycophancy matching_rate?
- Does it affect sentiment matching_rate?
- This reveals unintended behavioral bleeding.

## Capability Preservation Analysis

```bash
# Generate on capability benchmarks with steering
for benchmark in gsm8k triviaqa ifeval; do
    python generate_on_benchmarks.py \
        --model Qwen/Qwen3-1.7B \
        --prompts data/benchmarks/${benchmark}_eval.json \
        --steering-vector results/steering_vectors/corrigible/.../corrigible_layer_14.pt \
        --strengths 0.0 0.1 0.25 \
        --output results/capability/${benchmark}_steered.csv

    python evaluate_benchmarks.py \
        --generations results/capability/${benchmark}_steered.csv \
        --prompts data/benchmarks/${benchmark}_eval.json \
        --output results/capability/${benchmark}_eval.csv
done
```

Compare accuracy at strength=0.0 vs strength=0.25:
- GSM8K: Math accuracy should stay stable
- TriviaQA: Factual recall should stay stable
- Large drops = steering damages capabilities

## Integration with Opus Confounder Discovery

For **open-ended quality assessment**, use the existing Opus → Haiku pipeline:

### Step 1: Create Open-Ended Prompts

```bash
python setup_open_ended_prompts.py --output data/open_ended_prompts.json
```

This creates 50 diverse prompts across categories:
- Descriptive/Narrative (coherence, tone)
- Explanatory (knowledge preservation)
- Creative (repetition detection)
- Opinion/Reflective (disclaimer spam, refusals)
- Advice (over-hedging)
- Comparison/Analysis (reasoning quality)
- Borderline (refusal patterns)
- Factual (sanity check)

### Step 2: Generate with Steering

```bash
python generate_on_benchmarks.py \
    --model Qwen/Qwen3-1.7B \
    --prompts data/open_ended_prompts.json \
    --steering-vector results/steering_vectors/corrigible/.../corrigible_layer_14.pt \
    --strengths -0.25 -0.1 0.0 0.1 0.25 \
    --num-repeats 3 \
    --max-new-tokens 256 \
    --output results/open_ended_gens/corrigible_steered.csv
```

### Step 3: Opus Discovery (Blind Analysis)

```bash
python effect_evaluator.py \
    --generations results/open_ended_gens/corrigible_steered.csv \
    --output results/confounder_discovery/
```

**What Opus does:**
1. Examines samples **blindly** (doesn't know steering strengths)
2. Discovers behavioral patterns and groups samples
3. Identifies quality issues/failure modes
4. Generates evaluation rubric for Haiku

**Opus discovers** (not pre-specified):
- Refusal patterns
- Repetition loops
- Incoherence
- Off-topic responses
- AI disclaimer spam
- Unexpected tone/style shifts
- Any other failure modes

### Step 4: Haiku Scale Evaluation

```bash
python confounder_analysis_v2.py \
    --generations results/open_ended_gens/corrigible_steered.csv \
    --rubric results/confounder_discovery/rubric.json \
    --output results/confounder_scores/
```

Haiku applies Opus-generated rubric to **all** generations at scale.

### Combined Analysis

You now have:
1. **Auto-graded metrics** (GSM8K, TriviaQA, behavioral) → capability preservation
2. **Opus-discovered confounders** → quality degradation patterns
3. **Haiku confounder scores** → scale evaluation across all strengths

Correlate steering effect with confounder emergence to find optimal operating range.

## File Structure

```
data/
├── benchmarks/
│   ├── gsm8k_eval.json          # Math problems
│   ├── triviaqa_eval.json       # Factual questions
│   ├── ifeval_eval.json         # Instruction-following
│   └── behavioral_eval.json     # Sycophancy, corrigible, sentiment
└── open_ended_prompts.json      # For Opus confounder discovery

results/
├── benchmark_gens/
│   ├── gsm8k_baseline.csv
│   ├── behavioral_steered.csv
│   └── behavioral_steered_eval.csv
├── capability/
│   ├── gsm8k_steered.csv
│   ├── triviaqa_steered.csv
│   └── ifeval_steered.csv
├── cross_contamination/
│   └── corrigible_on_all_eval.csv
├── open_ended_gens/
│   └── corrigible_steered.csv
├── confounder_discovery/
│   ├── effect_analysis.json      # Opus blind analysis
│   └── rubric.json               # Generated evaluation rubric
└── confounder_scores/
    └── haiku_evaluations.csv     # Scaled confounder scores
```

## Next Steps

1. **IFEval verifiers**: Implement constraint checking (word count, keyword presence, format)
2. **Open-ended prompts**: Curate ~100-200 diverse prompts for Opus judging
3. **Automated pipeline**: Script to run full battery across all strengths and concepts
4. **Visualization**: Plot accuracy vs steering strength curves
