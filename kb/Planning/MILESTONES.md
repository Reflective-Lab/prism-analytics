---
source: mixed
---
# Milestones

> See `~/dev/reflective/stack/bedrock-platform/EPIC.md` for the coarse-grained outcomes these milestones advance.

---

## Current: v1.2.0 — Fuzzy Inference Release

**Target:** 2026-05 | **Tracks:** Converge 3.8.1

- [x] Ship Mamdani FIS (activated-rule trace, confidence, per-rule strengths)
- [x] Ship Sugeno FIS (order-0 and order-1 consequents, weighted-average output)
- [x] Ship defuzzification module (Centroid, Bisector, MoM, Height, WeightedAverage)
- [x] Ship `FuzzyInferencePack` and `SugenoInferencePack` as Converge pack adapters
- [ ] Restore coverage floor to 80% (currently at 60% — lowered to land fuzzy code)
- [ ] First clean `just release-check` run with fuzzy code included
- [ ] Tag v1.2.0

---

## Shipped: v1.1.0 — Converge 3.8.1 Foundation

**Released:** 2026-05

- [x] Adopt Converge 3.8.1 contract baseline (converge-prefixed crate names)
- [x] Adopt Extension Release Checklist (security-audit, coverage, performance-profile, soak)
- [x] Wire CI workflows: `ci`, `coverage`, `security`, `stability`
- [x] Enable crates.io publishing
- [x] Tag v1.1.0

---

## Shipped: v1.0.0 — Initial Release

**Released:** 2026-05

- [x] Initial release
- [x] Tag v1.0.0
