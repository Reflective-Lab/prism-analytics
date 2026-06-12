---
tags: [positioning, pitch, analytics, fuzzy]
source: llm
date: 2026-06-12
---
# Positioning

Why Prism exists, why it plays well with LLMs, and the full algorithm
catalog. Companion pitches live in the Ferrox, Arbiter, and Soter knowledge
bases; this note is the Prism chapter of the same story.

## Elevator Pitch

Prism is the **perception layer of the Converge platform**: closed-form
analytics and inference exposed as typed, explainable Suggestors. It turns
raw data — CSV, TSV, Parquet, Excel — into features, signals, and graded
judgments: *is this anomalous, what does it trend toward, which cluster does
it belong to, how much should this evidence matter?*

Everything in Prism is deliberately **closed-form and inspectable**: z-scores,
regressions, smoothing, k-means, fuzzy rules — algorithms whose every output
can be traced back to arithmetic a reviewer can check. Trained models,
training loops, and model registries live in Crucible; Prism stays the layer
you can audit with a pencil. Each proposal carries typed provenance
(`PRISM_PROVENANCE`) and lands as `Observed` / `Argued` evidence — signals
for the Converge promotion path, never promotion authority.

## Why It Plays Well With LLMs

An LLM reasons fluently about *qualitative* claims but cannot be trusted to
*compute* — it will estimate a mean, hallucinate a percentile, and argue both
sides of a trend. Prism is the calculator bolted to the reasoner:

- The LLM asks product-shaped questions; Prism answers with deterministic,
  reproducible numbers and typed provenance the LLM can cite rather than
  invent.
- **Fuzzy inference is the standout bridge**: LLMs and humans both speak in
  graded language ("somewhat risky", "mostly compliant"). `prism::fuzzy`
  gives that language formal semantics — membership functions, explainable
  rule firings, named defuzzification — so vague-but-meaningful judgments
  become auditable arithmetic instead of vibes.
- `MaterialityDegree` separates "how true is this?" from "how much should
  this matter to the decision?" — exactly the distinction an agent needs
  when weighing evidence for an action.

The LLM narrates and decides what to ask; Prism measures and grades. Neither
hallucinates the other's job.

## What It Solves Better Than Anything Else

Prism's niche is **explainable, dependency-light analytics inside the
governed loop**. Not a notebook, not a model server: a set of small, exact
solvers that formations can call in-process, with every output typed, traced
(`prism.suggestor.execute` spans), and reviewable. Where a trained model
gives you an opinion with a confidence score, a Prism pack gives you an
answer with a derivation. For the large class of product questions that do
not need deep learning — thresholds, trends, rankings, segments, graded
risk — closed-form beats trained on latency, auditability, and cost, and
never drifts.

## Algorithm Catalog

### Analytics packs

| Algorithm | Pack | Tagline |
|---|---|---|
| Z-score anomaly detection | `AnomalyDetectionPack` | How many standard deviations from normal is this point? |
| Logistic classification | `ClassificationPack` | A probability between 0 and 1, with coefficients you can read. |
| Gaussian naive Bayes classification | `NaiveBayesPack` | Fast probabilistic labeling from feature evidence, priors included. |
| Descriptive statistics | `DescriptiveStatsPack` | Mean, median, variance, percentiles — the ground truth of any dataset. |
| Exponential smoothing | `ForecastingPack` | Tomorrow as a weighted memory of yesterday. |
| Linear regression | `RegressionPack` | The line through the noise, with slope you can defend. |
| K-means clustering | `SegmentationPack` | Let the data choose its own groups. |
| Pairwise vector similarity | `SimilarityPack` | How alike are these two things, as a number? |
| Weighted multi-criteria ranking | `RankingPack` | Many incomparable criteria, one defensible ordering. |
| Moving-average trend detection | `TrendDetectionPack` | Is it actually going up, or do you just want it to? |

### Fuzzy inference (`prism::fuzzy` + `FuzzyInferencePack`)

| Algorithm | Tagline |
|---|---|
| Triangular / trapezoidal / shoulder membership functions | Piecewise-linear graded truth — simple enough to certify. |
| Gaussian membership | Smooth graded truth: `μ(x) = exp(−(x−c)²/2σ²)`. |
| Mamdani-style rule inference | Human-readable IF–THEN rules with explainable firings. |
| Sugeno (TSK) inference | Rules that conclude in functions — crisp outputs, fast evaluation. |
| Tsukamoto inference | Monotone consequents: every rule yields a crisp, ranked answer. |
| Defuzzification (centroid and friends) | From a fuzzy verdict back to one actionable number. |
| `MembershipDegree` / `MaterialityDegree` | "How true is it?" versus "how much should it matter?" — kept distinct on purpose. |

### Supporting machinery

| Capability | Tagline |
|---|---|
| Polars feature extraction (`FeatureAgent`) | Columnar speed from CSV, TSV, Parquet, and Excel into typed feature vectors. |
| Burn inference example | The marked path to trained models — which live in Crucible, not here. |

## Boundaries (One-Line Reminders)

- Prism answers: *what does the data say, and how much should it matter?*
  (`Observed` / `Argued`)
- Arbiter answers: *should this concrete request be allowed now?* (`Decided`)
- Ferrox answers: *what is the best feasible plan?* (`Searched`,
  optimization)
- Soter answers: *can any modeled request violate this invariant?*
  (`Searched`, symbolic)
- Fuzzy results are graded inference, not authorization and not proof — they
  inform gates; they never override Cedar. See
  [[Architecture/Fuzzy Logic Capability]] and [[Architecture/Project Boundary]].
