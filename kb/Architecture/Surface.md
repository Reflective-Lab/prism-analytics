---
tags: [architecture, surface]
source: mixed
---
# Surface

`prism` exposes one canonical published crate (`converge-prism-analytics`)
whose Rust library name is `prism`.

## Public surface

- `FeatureAgent` and `InferenceAgent`
- `prism::fuzzy` reusable fuzzy inference capability
- Training pipeline agents for dataset, validation, feature engineering,
  training, evaluation, registry, monitoring, deployment, and sample inference
- Analytics packs for anomaly detection, classification, descriptive stats,
  forecasting, fuzzy inference (Mamdani + Sugeno + Tsukamoto), ranking,
  regression, segmentation, similarity, and trend detection

## Contract dependencies

- `converge-pack` — `Pack`, `ProposedPlan`, `ProblemSpec`
- `converge-optimization` — pack invariants and gate evaluation helpers
- `converge-storage` — optional storage feature

## Forbidden imports

Per [Extension Release Checklist §1](https://github.com/Reflective-Lab/converge/blob/main/kb/Standards/Extension%20Release%20Checklist.md):

- No imports of `converge-core` internals.
- No imports of foundation `runtime`, `provider`, or transport crates.
- No re-exports of foundation types except those promised stable.
