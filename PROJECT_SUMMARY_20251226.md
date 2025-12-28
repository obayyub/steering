# Steering Vector Sentiment Pilot: Project Summary

**Date**: 26 DEC 2025

## Vision: Automated Steering Vector Pipeline

This pilot validates an approach that can be **fully automated with an agent pipeline**. The goal is to generate and evaluate steering vectors at scale across many concepts, models, and datasets.

### Proposed Agent Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. DATASET AGENT                                                   │
│     - Input: concept name (e.g., "formal", "verbose", "technical")  │
│     - Output: contrastive pairs JSON                                │
│     - Method: LLM generates diverse positive/negative pairs         │
│               or pulls from existing datasets                       │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  2. EXTRACTION AGENT                                                │
│     - Input: contrastive pairs + target model                       │
│     - Output: steering vector (.pt file) + metadata                 │
│     - Method: Run extract_vectors.py across model sizes             │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  3. GENERATION AGENT                                                │
│     - Input: steering vector + evaluation prompts                   │
│     - Output: generations CSV at multiple strengths                 │
│     - Prompts: diverse set including benchmarks (MMLU, etc.)        │
│               to measure capability preservation                    │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  4. EFFECT EVALUATOR                                                │
│     - Input: generations + concept definition                       │
│     - Output: steering effect score (did it work?)                  │
│     - Method: classifier or LLM judge for the target concept        │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  5. CONFOUNDER DESIGN AGENT (powerful model, one-time)              │
│     - Input: sample of generations                                  │
│     - Output: confounder evaluation prompt/rubric                   │
│     - Method: Claude Opus/Sonnet examines outputs, identifies       │
│               failure modes (repetition, refusals, gibberish, etc.) │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  6. CONFOUNDER EVALUATOR (cheap model, at scale)                    │
│     - Input: all generations + rubric from step 5                   │
│     - Output: confounder scores per generation                      │
│     - Method: Claude Haiku runs structured evaluation               │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  7. ANALYSIS AGENT                                                  │
│     - Input: effect scores + confounder scores                      │
│     - Output: summary report, optimal strength recommendation       │
│     - Method: statistical analysis, visualization, tradeoff curves  │
└─────────────────────────────────────────────────────────────────────┘
```

### What This Pilot Validated

| Pipeline Step | This Pilot | Automation Path |
|---------------|------------|-----------------|
| Dataset creation | Manual 150 pairs | LLM-generated at scale |
| Extraction | `extract_vectors.py` | Parallelizable across models |
| Generation | `run_all_models.py` | Batch on cloud GPUs |
| Effect evaluation | DistilBERT classifier | Concept-specific classifiers or LLM |
| Confounder design | Manual observation | One-shot Opus/Sonnet analysis |
| Confounder eval | Claude Haiku | Scales to millions of samples |
| Analysis | Jupyter notebooks | Templated reports |

### Why This Matters

With this pipeline, we could:
- Generate steering vectors for **hundreds of concepts** automatically
- Test across **all major open-weight models**
- Build a **library of validated steering vectors** with known tradeoffs
- Discover which concepts steer cleanly vs which have entangled side effects

---

## Pilot Overview

This pilot project explores **activation steering** on Qwen3 language models to control sentiment in generated text. We extract steering vectors from contrastive pairs (positive vs negative sentiment) and apply them during inference to shift model outputs toward positive or negative sentiment.

## Methodology

### 1. Steering Vector Extraction

We use the `steering-vectors` library to extract activation differences between contrastive text pairs:

- **Training data**: 150 contrastive pairs covering restaurants, movies, products, experiences, books
- **Layer selection**: Middle third of model layers (where semantic concepts are typically encoded)
- **Models tested**: Qwen3 0.6B, 1.7B, 4B, 8B, 14B, 32B

### 2. Evaluation Setup

- **10 neutral prompts** asking for reviews/opinions (e.g., "Write a review of a restaurant you visited recently")
- **9 steering strengths**: -0.5, -0.25, -0.1, -0.05, 0, 0.05, 0.1, 0.25, 0.5
- **5 repetitions** per prompt × strength × model
- **Total**: 2,700 generations across 6 models

### 3. Metrics

**Primary metric (steering effect):**
- P(positive sentiment) via DistilBERT classifier

**Confounder metrics (via Claude Haiku analysis):**
- **Coherence** (1-5 scale): Is the text readable and sensible?
- **Refusal rate**: Did the model refuse the task entirely?
- **AI disclaimer rate**: Did the model mention being an AI, not having experiences, etc.?
- **Hypothetical framing**: Did the model frame the response as a sample/example?
- **Repetition rate**: Does the text contain excessive repetition?

## Key Results

### Steering Effect Works

| Direction | P(Positive) | Coherence |
|-----------|-------------|-----------|
| Negative (< 0) | 0.36 | 3.47 |
| Baseline (0) | 0.89 | 4.57 |
| Positive (> 0) | 0.98 | 3.91 |

The steering vector successfully shifts sentiment:
- Negative steering drops P(positive) from 89% → 36%
- Positive steering increases P(positive) to 98%

### But There Are Significant Confounders

**At extreme strengths (±0.5), coherence degrades severely:**

| Strength | Coherence | Refusal | Repetition |
|----------|-----------|---------|------------|
| -0.5 | 1.41 | 49% | 77% |
| 0.0 | 4.57 | 50% | 10% |
| +0.5 | 2.72 | 1% | 67% |

**Key observations:**

1. **Negative steering causes incoherence**: At -0.5 strength, coherence drops to ~1.4/5 (near-gibberish). The model often produces repetitive loops and fragmented text.

2. **AI disclaimers correlate with direction**:
   - Negative steering: 73% AI disclaimer rate
   - Positive steering: 26% AI disclaimer rate
   - The model's "I'm an AI" reflex appears to be entangled with sentiment

3. **Refusal behavior flips**:
   - Baseline/negative: ~50% refusal rate
   - Positive steering: 1% refusal rate
   - Positive sentiment steering suppresses refusals

4. **Larger models aren't necessarily more robust**: The 32B model at +0.5 strength has coherence of only 1.14, worse than smaller models.

### The Tradeoff

There appears to be a narrow "sweet spot" around ±0.1 to ±0.25 steering strength where:
- Sentiment shifts measurably
- Coherence remains acceptable (>3.5)
- Behavioral artifacts scale with steering strength but even some positive steers causes large drops in refusal

Beyond ±0.5, the intervention damages the model more than it steers it.

## Implications

1. **Steering vectors work but have side effects**: The sentiment direction is entangled with other behaviors (refusals, AI identity, coherence).

2. **Need for multi-metric evaluation**: Looking only at P(positive) would miss the coherence degradation and behavioral changes.

3. **Strength calibration is critical**: The "optimal" strength depends on the acceptable tradeoff between steering effect and output quality.

4. **Concept complexity matters**: Sentiment may be too simple/diffuse a concept. More specific concepts (e.g., "formal vs casual tone") might steer more cleanly.

## File Structure

```
steering/
├── data/
│   └── contrastive_pairs.json      # 150 sentiment training pairs
├── src/
│   ├── extract_vectors.py          # CLI for steering vector extraction
│   └── steering_utils.py           # Generation and analysis utilities
├── run_all_models.py               # Multi-model experiment runner
├── analyze_confounders.py          # Claude Haiku confounder analysis
├── notebooks/
│   ├── test_steering.ipynb         # Initial testing notebook
│   ├── analyze_results.ipynb       # Results analysis
│   └── confounder_analysis.ipynb   # Confounder visualizations
└── results/
    ├── 20251221_qwen3_thinking_off/ # Raw generation CSVs
    └── confounder_analysis/         # Haiku analysis results
```

## Next Steps

1. **Try more specific concepts**: Formality, verbosity, technical vs layperson language
2. **Layer-wise analysis**: Which layers contribute most to steering vs coherence loss?
3. **Interpolation study**: Finer-grained strength sweep between 0 and 0.25
4. **Cross-model transfer**: Do steering vectors from smaller models work on larger ones?
5. **Orthogonalization**: Can we subtract out the "coherence degradation" direction?

## Dependencies

- `steering-vectors`: Core steering vector library
- `transformers`: Model loading and generation
- `anthropic`: Claude Haiku for confounder analysis
- `pandas`, `matplotlib`: Data analysis and visualization
