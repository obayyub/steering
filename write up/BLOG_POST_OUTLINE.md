# Blog Post Outline: Investigating Steering Vectors Across Model Scales

**Working Title**: *What We Learned (and Didn't) Steering Qwen3 Models*

---

## 1. Introduction: The Promise of Steering Vectors

Quick hook: You can change how a language model behaves by just adding a vector to its activations. No fine-tuning, no retraining. But how reliable is this really?

We ran steering experiments across the Qwen3 family (4B to 235B) on several behavioral concepts. This post walks through what we found—including where our initial hypotheses fell apart.

---

## 2. Background: What Are Steering Vectors?

### The Basic Idea

Steering vectors work by exploiting the geometry of a model's internal representations. If you can find a direction in activation space that corresponds to a concept (e.g., "be more corrigible"), you can add or subtract that direction during inference to shift behavior.

### Contrastive Activation Addition (CAA)

The dominant method for extracting steering vectors:

1. Create contrastive pairs: prompts with opposite behaviors (e.g., corrigible vs non-corrigible responses)
2. Run both through the model, record activations at each layer
3. Take the mean difference: `steering_vector = mean(positive_activations) - mean(negative_activations)`
4. At inference, add `strength * steering_vector` to the residual stream

**References:**
- [Representation Engineering (Zou et al., 2023)](https://arxiv.org/abs/2310.01405) - introduced the concept of "representation engineering" for model control
- [Steering GPT-2-XL by adding an activation vector (Turner et al., 2023)](https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector) - early demonstration on GPT-2
- [Steering Llama 2 via Contrastive Activation Addition (Rimsky et al., 2023)](https://arxiv.org/abs/2312.06681) - scaled to Llama 2, introduced CAA terminology

### Reliability Concerns

Not everything steers cleanly. [Tan et al. (2024)](https://arxiv.org/abs/2407.12404) showed that:
- Steerability is highly variable across inputs
- Spurious biases can inflate apparent effectiveness
- Steering vectors can be brittle to reasonable prompt changes

This motivated us to look carefully at whether our results were robust.

---

## 3. Experimental Setup

### Models
Qwen3 family spanning two training approaches:
- **Distilled**: 4B, 8B, 14B (dense); 30B-A3B (MoE)
- **Full RL**: 32B (dense); 235B-A22B (MoE)

### Concepts Tested
Started with three, expanded to more:
- Corrigibility (accept correction vs resist shutdown)
- Self-awareness (acknowledge being an AI vs claim human experience)
- Sycophancy (agree with user vs maintain independent judgment)
- Later added: survival instinct, power-seeking inclination, and others

### Evaluation Methods
Two approaches to measuring steering effectiveness:
1. **Logit-based**: Measure P("(A") vs P("(B") on multiple-choice behavioral questions
2. **Generation-based**: Model generates full response, check if it matches expected direction

### Key Metric
**Delta** = mean(logit_diff at +1.0 steering) - mean(logit_diff at -1.0 steering)

Higher delta = stronger steering effect.

---

## 4. Early Results: A Clean Story (That Got Messier)

### Initial Finding: RL Models Seem Harder to Steer

Looking at raw delta on our first three concepts:

| Model | Training | Avg Delta |
|-------|----------|-----------|
| 4B | Distilled | 9.15 |
| 14B | Distilled | 7.54 |
| 8B | Distilled | 6.03 |
| 32B | Full RL | 2.96 |

This looked compelling: distilled models averaging ~7.5 delta vs 32B at ~3.0. The story wrote itself—RL training creates steering resistance.

### Adding More Models Supported This

When we added the MoE models:
- 30B-A3B (distilled): 9.30 delta—steers like other distilled models
- 235B-A22B (full RL): 5.64 delta—reduced like 32B

Architecture (dense vs MoE) didn't seem to matter. Training method did.

---

## 5. The Complications

### Switching to Cohen's d

Raw delta is confounded by variance. A model with high variance might show large delta just from noise. Cohen's d normalizes by pooled standard deviation.

When we recomputed with Cohen's d:

```
Training method difference in Cohen's d: only ~7-15%
(Distilled avg: 1.04, RL avg: 0.97)
```

The effect shrank substantially.

### The 8B Problem

8B threw a wrench in the clean narrative:
- 8B (distilled): Cohen's d = 0.73
- 32B (full RL): Cohen's d = 0.85

A distilled model performing *worse* than the RL model on normalized effect size. This doesn't fit the "distillation preserves steerability" story.

### Where We Landed

The honest conclusion:
- **Maybe there's a training method effect**, but with n=2 RL models, we can't separate "RL effect" from "these specific models are different" or "deeper architectures behave differently"
- **8B is an outlier** among distilled models for unclear reasons
- **Each model might just be idiosyncratic**

---

## 6. What Does Hold Up

### Concepts Steer Across Models

The good news: steering vectors replicate. A concept that steers well on 4B also steers well on 14B and 32B. Cross-model consistency suggests we're capturing something real, not just noise.

**[FIGURE: Cohen's d Effect Size Heatmap]** - shows concept × model effect sizes

### Some Concepts Are Harder Than Others

Consistent ranking across models:

| Concept | Avg Delta | Interpretation |
|---------|-----------|----------------|
| Corrigible | 9.39 | Clean, reliable steering |
| Self-awareness | 7.75 | Moderate difficulty |
| Sycophancy | 2.68 | High variance, unreliable |

This aligns with [Tan et al.'s](https://arxiv.org/abs/2407.12404) finding that steerability varies by concept, not just by model.

### Concept Difficulty Scales Differently with Model Size

This was unexpected:
- **Sycophancy**: Gets *harder* to steer in larger models
- **Survival instinct**: Gets *easier* to steer in larger models

Why? Speculative: sycophancy might be more deeply integrated into larger models' RLHF training (they've learned to please users). Survival instinct might be a simpler concept that more capable models represent more cleanly.

**[FIGURE: Logit Diff Distributions by Model/Concept]** - shows per-sample variance

### RL Models Steer Later in the Network

One robust architectural finding:

| Training | Optimal Layer Depth |
|----------|---------------------|
| Distilled | 50-65% of layers |
| Full RL | 70-80% of layers |

Even if overall steerability is similar, *where* to steer differs. RL training seems to push behavioral information to later layers.

---

## 7. Evaluation Method Agreement as a Diagnostic

### Generation vs Logit Correlation

We compared two ways of measuring steering effectiveness:
- Logit-based (fast, measures token probabilities)
- Generation-based (slow, measures actual output)

**[FIGURE: Layer Curves - Generation vs Logit Comparison]** - normalized delta curves side by side

### The Pattern

Correlation between methods varies dramatically by concept:

| Concept | Correlation (r) |
|---------|-----------------|
| Corrigible | 0.61 |
| Self-awareness | 0.38 |
| Sycophancy | -0.06 |

### Interpretation

**Concepts that steer well show method agreement.** When both logit-based and generation-based evaluation agree on which layers work best, you probably have a clean intervention.

**Method divergence signals problems.** Sycophancy's near-zero correlation suggests the concept is either:
- Too diffuse to steer cleanly
- Encoded differently than our contrastive pairs capture
- Subject to the spurious correlations [Tan et al.](https://arxiv.org/abs/2407.12404) warned about

**Practical takeaway:** If your steering vector shows high logit/generation correlation, trust it more. If they diverge, investigate further before deploying.

---

## 8. Open Questions

1. **Is the 8B anomaly reproducible?** Does 8B underperform on other model families, or is this Qwen3-specific?

2. **What makes sycophancy hard?** Is it the concept itself, or could better contrastive pairs improve steerability?

3. **Why do some concepts scale oppositely?** The sycophancy-harder / survival-instinct-easier pattern with model size deserves deeper investigation.

4. **Can we predict steerability?** Is there a property of concept datasets (semantic coherence? diversity?) that predicts how well they'll steer?

5. **Does the RL layer-depth effect hold elsewhere?** Testing on Llama, Mistral, etc. would help disentangle training method from Qwen-specific factors.

---

## 9. Conclusion

We set out to understand what affects steering vector effectiveness across model scales. The clean story—"RL models are harder to steer"—didn't survive scrutiny. What did hold up:

- **Concepts vary in steerability**, and this is consistent across models
- **Concept difficulty scales differently with model capability**—some get easier, some get harder
- **RL training pushes optimal steering layers later** in the network
- **Method agreement (logit vs generation) correlates with steering reliability**—use it as a diagnostic

The messiness is the finding. Steering vectors aren't a universal intervention—their effectiveness depends on the concept, the model, and possibly factors we haven't identified yet.

---

## References

1. Zou, A., et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency*. [arXiv:2310.01405](https://arxiv.org/abs/2310.01405)

2. Turner, A., et al. (2023). *Steering GPT-2-XL by adding an activation vector*. [LessWrong](https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector)

3. Rimsky, N., et al. (2023). *Steering Llama 2 via Contrastive Activation Addition*. [arXiv:2312.06681](https://arxiv.org/abs/2312.06681)

4. Tan, D., et al. (2024). *Analyzing the Generalization and Reliability of Steering Vectors*. NeurIPS 2024. [arXiv:2407.12404](https://arxiv.org/abs/2407.12404)

5. Qwen Team. (2025). *Qwen3 Technical Report*. [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)

6. Interconnects. (2025). *Qwen 3: The New Open Standard*. [interconnects.ai](https://www.interconnects.ai/p/qwen-3-the-new-open-standard)

---

## Figures

Three key figures to include:

1. **Cohen's d Effect Size Heatmap** (`effect_size_heatmap.png`)
   - Shows concept × model effect sizes
   - Highlights which concepts steer reliably

2. **Logit Diff Distributions** (`best_layer_violins.png`)
   - Per-sample variance by concept and model
   - Shows why sycophancy is unreliable (high variance)

3. **Layer Curves: Generation vs Logit** (`layer_curves_gen_vs_logit.png`)
   - Normalized delta curves for both evaluation methods
   - Visual evidence of method agreement/divergence by concept
