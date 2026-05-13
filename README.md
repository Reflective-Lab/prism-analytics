# prism

[![CI](https://github.com/Reflective-Lab/prism-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/Reflective-Lab/prism-analytics/actions/workflows/ci.yml)
[![Coverage](https://github.com/Reflective-Lab/prism-analytics/actions/workflows/coverage.yml/badge.svg)](https://github.com/Reflective-Lab/prism-analytics/actions/workflows/coverage.yml)
[![Security](https://github.com/Reflective-Lab/prism-analytics/actions/workflows/security.yml/badge.svg)](https://github.com/Reflective-Lab/prism-analytics/actions/workflows/security.yml)
[![Stability](https://github.com/Reflective-Lab/prism-analytics/actions/workflows/stability.yml/badge.svg)](https://github.com/Reflective-Lab/prism-analytics/actions/workflows/stability.yml)
[![Crates.io](https://img.shields.io/crates/v/converge-prism-analytics.svg)](https://crates.io/crates/converge-prism-analytics)
[![docs.rs](https://docs.rs/converge-prism-analytics/badge.svg)](https://docs.rs/converge-prism-analytics)
[![dependency status](https://deps.rs/repo/github/Reflective-Lab/prism-analytics/status.svg)](https://deps.rs/repo/github/Reflective-Lab/prism-analytics)
![MSRV](https://img.shields.io/badge/MSRV-1.94.0-blue)
<img alt="gitleaks badge" src="https://img.shields.io/badge/protected%20by-gitleaks-blue">
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Analytics and ML suggestors for Converge formations.

`prism` is a Converge extension. It keeps feature extraction, analytic packs,
training, inference, model registry, monitoring, and deployment-decision
suggestors outside the Converge foundation while using Converge contracts for
in-loop behavior.

Cargo package: `converge-prism-analytics`. Rust library name remains `prism`.

## Why It Exists

Converge should not become an analytics framework. Prism gives formations a
place to ask data-driven agents for proposals while Converge keeps authority
over promotion.

## What Prism Owns

- Polars-based ingestion and feature extraction.
- Burn-based inference examples.
- Reusable fuzzy inference capability through `prism::fuzzy`.
- Analytic pack solvers and typed inputs/outputs.
- Training pipeline agents: dataset, validation, feature engineering,
  hyperparameter search, model training, evaluation, registry, monitoring,
  deployment decision, and sample inference.
- Typed proposal provenance through `ProvenanceSource` / `PRISM_PROVENANCE`.
- Suggestor-boundary tracing through `prism.suggestor.execute` spans.
- Compile-fail tests that enforce Converge authority boundaries.

## Packs

| Pack | Algorithm family |
|---|---|
| `AnomalyDetectionPack` | Z-score anomaly detection |
| `ClassificationPack` | Logistic classification |
| `DescriptiveStatsPack` | Mean, median, variance, percentiles |
| `ForecastingPack` | Exponential smoothing |
| `FuzzyInferencePack` | Membership functions and explainable fuzzy rules |
| `RankingPack` | Weighted multi-criteria ranking |
| `RegressionPack` | Linear regression |
| `SegmentationPack` | K-means clustering |
| `SimilarityPack` | Pairwise vector similarity |
| `TrendDetectionPack` | Moving-average trend detection |

## Boundary

| Layer | Responsibility |
|---|---|
| Converge | Suggestor contract, proposal promotion, and shared context. |
| Prism | Analytics packs, feature agents, training agents, and ML pipeline behavior. |
| Products | Domain datasets, model rollout policy, credentials, and deployment topology. |

## Repository Layout

```text
crates/prism/
  src/engine.rs     FeatureAgent and feature vectors
  src/fuzzy/        Reusable fuzzy membership and inference capability
  src/ingest.rs     CSV, TSV, Parquet, and optional Excel ingestion
  src/model.rs      Burn inference example
  src/packs/        Analytics packs and solvers
  src/training.rs   Training and monitoring suggestors
  tests/            Integration, property, negative, and compile-fail tests
```

## Usage

```rust
use prism::FeatureAgent;

let agent = FeatureAgent::new(None);
engine.register_suggestor(agent);
```

## Feature Flags

- Default: none.
- `storage`: enables optional `converge-storage` support.
- `excel`: enables Excel ingestion through `calamine`.

## Development

```sh
just check
just check-all
just test
just lint
just doc
```

Converge platform dependencies resolve from crates.io.

## Project Files

- [AGENTS.md](AGENTS.md) - agent entrypoint and boundary rules.
- [CHANGELOG.md](CHANGELOG.md) - release notes.
- [CONTRIBUTING.md](CONTRIBUTING.md) - contribution guide.
- [SECURITY.md](SECURITY.md) - vulnerability reporting and operator notes.
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) - community expectations.

## Status

Extracted from `converge/crates/analytics` on 2026-05-05 as part of the v3.8
foundation extraction.

## License

MIT - see [LICENSE](LICENSE).
