# Logit Diff Steering Analysis Findings

## Summary

Analysis of steering vector effectiveness across Qwen3 models (4B, 8B, 14B, 32B) for three concepts: corrigible, self_awareness, and sycophancy.

## Key Finding: 32B is Hardest to Steer

| Model | Avg Delta (steerability) | Layers | Training Method |
|-------|--------------------------|--------|-----------------|
| 4B    | 9.15                     | 36     | Distillation    |
| 14B   | 7.54                     | 40     | Distillation    |
| 8B    | 6.03                     | 36     | Distillation    |
| 32B   | 2.96                     | 64     | Full RL         |

## Architecture Details

| Model | Layers | Hidden | Heads | KV Heads | Head Dim | GQA Ratio |
|-------|--------|--------|-------|----------|----------|-----------|
| 4B    | 36     | 2560   | 32    | 8        | 80       | 4         |
| 8B    | 36     | 4096   | 32    | 8        | 128      | 4         |
| 14B   | 40     | 5120   | 40    | 8        | 128      | 5         |
| 32B   | 64     | 5120   | 64    | 8        | 80       | 8         |

## Training Differences (from Qwen3 Technical Report)

### Full RL Pipeline (Frontier Models)
- **32B** and **235B-A22B** only
- Training stages:
  1. Long Chain-of-Thought (CoT) cold start
  2. Reasoning-based reinforcement learning (RL)
  3. Thinking mode fusion
  4. General RL
- Similar to DeepSeek R1 recipe

### Strong-to-Weak Distillation
- **14B, 8B, 4B, 1.7B, 0.6B** and **30B-A3B**
- Mostly SFT tuning on synthetic data from larger models
- Process "not clearly documented" per external analysis

## Best Layer Positions

32B is unique - its best steering layers are in the final third (69-80%) for all concepts, while other models have best layers in the middle (50-65%).

| Model | Corrigible | Self-Awareness | Sycophancy |
|-------|------------|----------------|------------|
| 4B    | 50% (L18)  | 58% (L21)      | 53% (L19)  |
| 8B    | 47% (L17)  | 64% (L23)      | 78% (L28)  |
| 14B   | 50% (L20)  | 62% (L25)      | 62% (L25)  |
| 32B   | 69% (L44)  | 77% (L49)      | 80% (L51)  |

## Interpretation

1. **32B's resistance to steering** is likely explained by its full RL training, which made it more robust to activation perturbations.

2. **8B being slightly harder to steer than 4B/14B** is NOT explained by training differences (all three used distillation). This could be:
   - Random variation (n=4 is small)
   - Undocumented differences in distillation process
   - Architectural factors (8B has larger hidden dim than 4B)

3. **Architectural correlations** (with only 4 data points):
   - Layers: r = -0.87 (more layers → harder to steer)
   - Heads: r = -0.85
   - Params per layer: r = -0.82

   These correlations are largely driven by 32B being an outlier.

## Methodology

- Steering vectors trained at each layer (starting from 33% depth)
- Logit differences computed: P("(A" | prompt) vs P("(B" | prompt)
- Strengths tested: -1.0, 0.0, +1.0
- Delta = mean(logit_diff at +1.0) - mean(logit_diff at -1.0)

## Sources

- [Qwen3 Blog Post](https://qwenlm.github.io/blog/qwen3/)
- [Qwen3 Technical Report (arXiv)](https://arxiv.org/pdf/2505.09388)
- [Interconnects: Qwen 3 Analysis](https://www.interconnects.ai/p/qwen-3-the-new-open-standard)
