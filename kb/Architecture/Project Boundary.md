---
tags: [architecture, strategy, boundary]
source: human
date: 2026-05-09
---
# Project Boundary — prism-analytics vs. trained-model project

## The decision

prism-analytics is a **pure inference library** — no training, no learned
weights, no gradient descent. A separate future project will own trained models.

## What belongs in prism-analytics

- Hand-authored rules (Mamdani, Sugeno, Tsukamoto FIS)
- Closed-form statistics (z-score, descriptive stats, SES, OLS)
- Inference from pre-fit parameters (logistic classifier, linear regression,
  k-means, cosine similarity, ranking, trend detection)
- Anything where the "model" is fully described by its inputs at call time

The defining property: **deterministic, explainable, no training pipeline**.

## What belongs in the trained-model project

- Random Forests and gradient boosted trees (XGBoost-style ensembles)
- ANFIS — Adaptive Neuro-Fuzzy Inference System (learned Sugeno via Burn)
- Learned embeddings, SVMs with kernels, any model whose parameters come
  from fitting to data

Planned stack: **Burn** as the training framework (native Rust, GPU-capable).

## The fuzzy boundary

Mamdani and Sugeno with **expert-authored rules** → prism-analytics.

ANFIS → trained-model project. ANFIS is Sugeno + backprop on Gaussian MF
parameters, which requires a training pipeline and loss function — exactly
what Burn is for.

This is why ANFIS is listed as a deliberate non-goal in
[[Fuzzy Logic Capability#Slice Mapping]].

## Why the split

- Keeps prism-analytics dependency-light and audit-friendly (no training deps,
  no large binary artifacts).
- Training pipelines have different release cadences, data governance concerns,
  and compute requirements than inference libraries.
- Explainability and determinism are easier to guarantee when there are no
  learned weights in the loop.
