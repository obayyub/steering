# Research Notes & Low-Priority TODOs

## Open Questions

### Chat Format Impact on Steering Efficacy
**Observation**: Sycophancy steering showed improved performance after switching to chat template format during extraction (matching the generation format). Corrigible-neutral-HHH worked better overall.

**Questions**:
- How much did the format fix improve sycophancy specifically?
- Is sycophancy inherently harder to steer than corrigibility?
- Does extraction format mismatch explain the initial poor results?

**Next Steps** (when time permits):
- Re-run original sycophancy experiment with raw text extraction (no chat template) to quantify the format impact
- Compare layer-by-layer: does format mismatch affect different layers differently?
- Test hypothesis: concepts requiring nuanced social reasoning (sycophancy) may be more sensitive to extraction methodology than binary disposition concepts (corrigibility)

**Priority**: Low - interesting for methodology paper, not blocking current work

---

## Future Investigations

### Layer-wise Analysis
- Which layers contribute most to steering vs coherence preservation?
- Can we identify "safe" layers that steer without artifacts?

### Concept Taxonomy
- Build dataset of concepts ranked by steerability
- Identify what makes some concepts steer cleanly vs others
