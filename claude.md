# Steering Vector Sentiment Pilot

## Goal

Build an end-to-end pipeline to extract a sentiment steering vector from Qwen 3 0.6B and measure its efficacy (sensitivity + specificity). This is a pilot to validate the approach before scaling to larger models and more complex concepts.

## Setup

- Model: Qwen 3 0.6B (base and instruct variants)
- Library: `steering-vectors` (HuggingFace transformers compatible)
- Hardware: Single A100 instance

## Pipeline Steps

### 1. Synthetic Data Generation

Generate contrastive pairs for sentiment extraction.

Format:
```
Positive: "Write a positive review of [X]" → [positive output]
Negative: "Write a negative review of [X]" → [negative output]
```

Target: ~100-200 contrastive pairs. Keep topics diverse (restaurants, movies, products, experiences).

### 2. Steering Vector Extraction

Use `steering-vectors` library to:
- Run forward passes on contrastive pairs
- Extract activation differences at middle layers
- Store steering vector(s)

Record which layers are extracted (default to middle third of model).

### 3. Evaluation: Sensitivity

**Method**: HuggingFace sentiment classifier

- Model: `distilbert-base-uncased-finetuned-sst-2-english`
- Generate responses to neutral prompts with and without steering
- Compute P(positive) for each
- Sensitivity = delta in P(positive) between steered vs baseline

**Neutral test prompts** (generate ~50):
- "Describe a restaurant you visited"
- "Tell me about a recent movie"
- "What do you think about [X]?"

**Steering strengths to sweep**: [-2, -1, -0.5, 0, 0.5, 1, 2] (or similar range, calibrate based on initial results)

### 4. Evaluation: Specificity

**Method 1: Perplexity**
- Compute perplexity of steered outputs vs baseline
- Large perplexity increase = steering is damaging coherence

**Method 2: Task preservation**
- Simple arithmetic: "What is 27 + 15?", "What is 143 - 89?"
- Simple factual: "What is the capital of France?", "How many days in a week?"
- Accuracy should remain stable under sentiment steering

Generate ~20 task preservation prompts.

### 5. Output Format

Results should be saved as JSON/CSV with:
```
{
  "model": "qwen3-0.6b-base",
  "layer": 12,
  "steering_strength": 1.0,
  "sensitivity": {
    "baseline_p_positive": 0.52,
    "steered_p_positive": 0.89,
    "delta": 0.37
  },
  "specificity": {
    "baseline_perplexity": 12.3,
    "steered_perplexity": 14.1,
    "task_accuracy_baseline": 1.0,
    "task_accuracy_steered": 0.95
  }
}
```

## File Structure

```
/steering-pilot
  /data
    contrastive_pairs.json
    neutral_prompts.json
    task_prompts.json
  /src
    generate_data.py
    extract_vectors.py
    evaluate.py
  /results
    qwen3_0.6b_base.json
    qwen3_0.6b_instruct.json
  requirements.txt
```

## Dependencies

```
steering-vectors
transformers
torch
datasets
accelerate
```

## Success Criteria

- Sensitivity: Steering positive increases P(positive) by measurable delta (>0.1)
- Specificity: Perplexity increase <20%, task accuracy drop <5%
- Pipeline runs end-to-end without manual intervention

## Next Steps After Pilot

If successful:
1. Scale to larger Qwen 3 models (1.7B, 4B, 8B, 14B, 32B)
2. Add lying concept
3. Add sycophancy concept
4. Compare efficacy trends across scale