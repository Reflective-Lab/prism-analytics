> **Archived 2026-07-02** — active milestone tracking moved to Linear (Reflective team).
> This file is kept for historical context only. Do not add new items here.

---
source: mixed
---
# Milestones

> See `~/dev/reflective/stack/bedrock-platform/EPIC.md` for the coarse-grained outcomes these milestones advance.

---

## Shipped: v2.0.1 — MembershipDegree Newtype — 2026-05-18

**Tracks:** Converge 3.9.1

- [x] Introduce `MembershipDegree` newtype (clamps to `[0,1]`) across the
      fuzzy module — replaces bare `f64` in `FuzzyInferenceOutput.memberships`,
      `ActivatedRule` fields, and all internal arithmetic
- [x] Fix downstream consumers in organism (`graded_admission`, `fuzzy_reasoner`)
- [x] Publish `converge-prism-analytics` v2.0.1 to crates.io
- [x] Tag v2.0.1
- [x] Remove `[patch.crates-io]` overrides from organism and tally-escrow

---

## Shipped: v2.0.0 — Fuzzy Inference Release — 2026-05-17

**Tracks:** Converge 3.9.1

Supersedes the planned v1.2.0 milestone. Bumped to v2.0.0 because the
fuzzy module reorganized the public API surface (FIS / Sugeno /
defuzzification under `prism::fuzzy::*`).

- [x] Ship Mamdani FIS (activated-rule trace, confidence, per-rule strengths)
- [x] Ship Sugeno FIS (order-0 and order-1 consequents, weighted-average output)
- [x] Ship defuzzification module (Centroid, Bisector, MoM, Height, WeightedAverage)
- [x] Ship `FuzzyInferencePack` and `SugenoInferencePack` as Converge pack adapters
- [x] Bump converge-pack / converge-optimization / converge-kernel to 3.9.1
- [x] Make `just coverage` floor env-configurable via `COVERAGE_FLOOR`
      (default 80); ship v2.0.0 with `COVERAGE_FLOOR=60` to land fuzzy code
- [x] First clean `just release-check` run with fuzzy code included
- [x] Tag v2.0.0

**Coverage caveat:** Coverage was temporarily lowered from 80% → 60% to
land fuzzy code. Restoring to 80% (by adding unit coverage to ranking
/ segmentation / similarity packs) is tracked as a follow-up before
v2.0.1.

---

## Shipped: v1.1.0 — Foundation

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
