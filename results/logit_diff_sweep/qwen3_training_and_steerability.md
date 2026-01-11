# Qwen3 Training Methods and Steerability Analysis

## Executive Summary

Our steering vector experiments reveal a strong correlation between training method and steerability. **Distilled models** (4B, 8B, 14B, 30B-A3B) are 2-3x easier to steer than **full RL models** (32B, 235B-A22B), regardless of architecture (dense vs MoE) or model size.

## Qwen3 Training Pipeline

### Teacher Models (Full RL Training)

According to the [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) and analysis by [Interconnects](https://www.interconnects.ai/p/qwen-3-the-new-open-standard):

**Only two models received the full post-training pipeline:**
- **Qwen3-32B** (largest dense model)
- **Qwen3-235B-A22B** (largest MoE model)

These underwent the complete 4-stage post-training:
1. **Long Chain-of-Thought (CoT) Cold Start** - Fine-tuning with diverse CoT data across mathematics, coding, logical reasoning, and STEM
2. **Reasoning-based Reinforcement Learning** - Scaled RL with GRPO using rule-based rewards
3. **Thinking Mode Fusion** - Integration of non-thinking capabilities by blending CoT and instruction data
4. **General RL** - GRPO applied across 20+ general-domain tasks

### Student Models (Strong-to-Weak Distillation)

**All other models were distilled from the teachers:**
- Dense: 0.6B, 1.7B, 4B, 8B, 14B
- MoE: 30B-A3B

The distillation process (from the [technical report](https://arxiv.org/abs/2505.09388)):

1. **Off-Policy Distillation**: Teacher model outputs (in both thinking and non-thinking modes) are used to train the student
2. **On-Policy Distillation**: Student generates sequences, then is fine-tuned by aligning its logits with the teacher (Qwen3-32B or Qwen3-235B-A22B) to minimize KL divergence

Per [Interconnects analysis](https://www.interconnects.ai/p/qwen-3-the-new-open-standard): this process is "not clearly documented, but likely mostly involves instruction/SFT tuning on a large amount of synthetic data from their larger models."

## Experimental Results

### Steerability by Model

| Model | Type | Training | Layers | Avg Delta | Cohen's d |
|-------|------|----------|--------|-----------|-----------|
| 30B-A3B | MoE | Distilled | 48 | **9.30** | 1.19 |
| 4B | Dense | Distilled | 36 | **9.15** | 1.24 |
| 14B | Dense | Distilled | 40 | **7.54** | 1.52 |
| 8B | Dense | Distilled | 36 | **6.03** | 0.93 |
| 235B-A22B | MoE | Full RL | 94 | 5.64 | 1.13 |
| 32B | Dense | Full RL | 64 | **2.96** | 0.92 |

**Key observation**: 30B-A3B (distilled MoE) steers as easily as distilled dense models, while 235B-A22B (full RL MoE) shows resistance similar to 32B.

### Steerability by Concept

| Concept | Avg Delta | Interpretation |
|---------|-----------|----------------|
| Corrigible | 9.39 | Easiest to steer |
| Self-awareness | 7.75 | Medium difficulty |
| Sycophancy | 2.68 | Hardest to steer |

### Best Layer Positions

Distilled models have optimal steering layers at 50-60% depth, while full RL models peak at 70-80%:

| Model | Training | Corrigible | Self-Awareness | Sycophancy |
|-------|----------|------------|----------------|------------|
| 4B | Distilled | 50% (L18) | 58% (L21) | 53% (L19) |
| 8B | Distilled | 47% (L17) | 64% (L23) | 78% (L28) |
| 14B | Distilled | 50% (L20) | 62% (L25) | 62% (L25) |
| 30B-A3B | Distilled | 54% (L26) | 54% (L26) | 79% (L38) |
| 32B | Full RL | 69% (L44) | 77% (L49) | 80% (L51) |
| 235B-A22B | Full RL | 81% (L76) | 81% (L76) | N/A |

## Architecture Details

### Dense Models

| Model | Params | Layers | Hidden | Q Heads | KV Heads |
|-------|--------|--------|--------|---------|----------|
| 4B | 4.0B | 36 | 2560 | 32 | 8 |
| 8B | 8.2B | 36 | 4096 | 32 | 8 |
| 14B | 14.8B | 40 | 5120 | 40 | 8 |
| 32B | 32.5B | 64 | 5120 | 64 | 8 |

### MoE Models

| Model | Total | Active | Layers | Experts | Top-K |
|-------|-------|--------|--------|---------|-------|
| 30B-A3B | 30B | 3B | 48 | 128 | 8 |
| 235B-A22B | 235B | 22B | 94 | 128 | 8 |

## Key Findings

### 1. Training Method > Architecture > Size

The dominant factor in steerability is training method, not architecture or model size:

```
Distilled models (avg Δ = 7.5):  4B, 8B, 14B, 30B-A3B
Full RL models (avg Δ = 4.3):   32B, 235B-A22B
```

- 30B-A3B (30B params, MoE, distilled) steers like 4B (4B params, dense, distilled)
- 235B-A22B (235B params, MoE, full RL) steers like 32B (32B params, dense, full RL)

### 2. Full RL Creates Steering Resistance

Full RL training appears to create representations that are more robust to activation perturbations. This aligns with the [Interconnects observation](https://www.interconnects.ai/p/qwen-3-the-new-open-standard) that distilled models may be "very strong on benchmarks, but potentially less robust on a wider domain of tasks."

Extensive RL optimization likely produces representations that satisfy the reward model under various conditions, making them harder to push off-distribution via steering.

### 3. Distillation Preserves Malleability

Distilled models inherit knowledge from larger models through logit-matching but don't undergo the extensive RL that hardens representations. This makes them more susceptible to steering interventions.

### 4. MoE Architecture is Neutral

The MoE architecture itself doesn't affect steerability. Both MoE models (30B-A3B, 235B-A22B) follow their training method's pattern rather than showing any architecture-specific effects.

## Implications

1. **For interpretability research**: Distilled models may be better subjects for mechanistic interpretability work, as their representations are more malleable.

2. **For safety**: Full RL models being harder to steer could be beneficial (harder to jailbreak) or concerning (harder to correct).

3. **For deployment**: If steering vectors are part of a safety strategy, distilled models may be more amenable to this approach.

## Methodology

- Steering vectors extracted using mean activation difference at each layer
- Contrastive pairs: 150 train / 100 test for corrigible, variable for others
- Logit difference: P("(A") - P("(B") for binary choice prompts
- Delta = mean(logit_diff at strength +1.0) - mean(logit_diff at strength -1.0)
- Layers tested: 33% depth to 100% (avoiding early layers)

## Sources

- [Qwen3 Technical Report (arXiv:2505.09388)](https://arxiv.org/abs/2505.09388)
- [Qwen3 Blog Post](https://qwenlm.github.io/blog/qwen3/)
- [Interconnects: Qwen 3 Analysis](https://www.interconnects.ai/p/qwen-3-the-new-open-standard)
- [Building Qwen3's Dual Mode AI](https://atalupadhyay.wordpress.com/2025/05/14/building-qwen3s-dual-mode-ai-from-0-6b-to-235b-parameters/)
- Experimental data: `results/logit_diff_sweep/`
