# Critique of "What I Learned (and Didn't) Steering Qwen 3 Models"

**Date:** 2026-01-15

---

## Grammar/Typos to Fix

| Line | Issue |
|------|-------|
| 23 | "varaible" → "variable" |
| 28 | "judgement" → "judgment" (minor, both acceptable) |
| 34 | "generation based" → "generation-based" |
| 38 | "corribility" → "corrigibility", "sycohphancy" → "sycophancy", "though" → "through" |
| 40 | "intial" → "initial", "differnece" → "difference", "ful" → "full" |
| 53 | "become" → "becomes" |
| 61 | "have less" → "has fewer", double space after "conclusions," |
| 73 | "at the 8B" → "as the 8B" |
| 81 | "optical" → "optimal", "atleast" → "at least", "constrastive" → "contrastive" |
| 85 | "the require" → "they require" |
| 89 | "perfomed" → "performed" |
| 93 | "neglible" → "negligible" |
| 97 | "a 8%" → "an 8%", "corribility" → "corrigibility" |

---

## Structural Issues

**1. Missing promised content:** Line 23 says "(table below):" but there's no table showing the Qwen3 model family with their sizes/training methods. You should add this or remove the reference.

**2. Title mismatch:** Your document title is "What I Learned (and Didn't) Steering Qwen 3 Models" but the H3 header says "Investigating Steering Vectors Across Model Scale." Pick one or make the H1 the catchy title and H3 a subtitle.

**3. Abrupt conclusion:** The ending is just two sentences after a horizontal rule. This is your chance to tie things together and leave an impression.

---

## What to Cut/Condense

**The "Steering Vectors" and "CAA" sections** — If your audience is AI safety researchers or ML practitioners, they likely already know what steering vectors are. Consider:
- Combining these into one short "Background" section (3-4 sentences max)
- Or adding a note like "For readers unfamiliar with steering vectors, see [link]" and moving on

This would tighten the post and get to *your* contribution faster.

---

## What to Expand

**1. The layer depth finding is actually interesting!** Section "Full RL Models Steer Later in the Network" is a genuine positive result but it's undersold. You found a consistent pattern across models and datasets. This deserves:
- More emphasis (maybe even promote it in the intro as a finding)
- Speculation about *why* this might be (you hint at it but could develop it)
- Whether this has practical implications for practitioners choosing steering layers

**2. Conclusion needs work.** Currently it reads as "I didn't find much." But you actually found:
- Models are idiosyncratic in ways that aren't predictable
- RL models steer later in the network
- Logit-diff and generation-based eval can disagree substantially
- Dataset construction matters more than expected

Frame these as learnings. Add: What would you do differently? What questions remain? What should practitioners take away?

**3. Methodology decisions:** You made smart choices (normalizing by baseline variance, balancing A/B answers). Being explicit about *why* you made these decisions shows research maturity. A sentence or two explaining the reasoning would strengthen the piece.

---

## Overall Impression

**What works:**
- Honest narrative arc: hypothesis → initial results → complication → revised understanding
- Good statistical thinking (Glass's Delta normalization)
- Willingness to report null results
- Good use of visualizations

**For demonstrating research capability:**
The post does show you can:
- Form and test hypotheses systematically
- Recognize and correct for statistical artifacts
- Revise conclusions based on evidence
- Work across multiple models/datasets

**Main weakness:** The framing is too defeatist. "Models are idiosyncratic" is a finding, not a failure. The layer depth result is genuinely interesting. Reframe the conclusion from "I didn't find what I was looking for" to "Here's what I learned about what makes steering vectors work (and not work)."
