# prism

Analytics and ML suggestors for Converge formations.

`prism` is a Converge extension. It keeps feature extraction, analytic packs,
training, inference, model registry, monitoring, and deployment-decision
suggestors outside the Converge foundation while using Converge contracts for
in-loop behavior.

## Why It Exists

Converge should not become an analytics framework. Prism gives formations a
place to ask data-driven agents for proposals while Converge keeps authority
over promotion.

## What Prism Owns

- Polars-based ingestion and feature extraction.
- Burn-based inference examples.
- Analytic pack solvers and typed inputs/outputs.
- Training pipeline agents: dataset, validation, feature engineering,
  hyperparameter search, model training, evaluation, registry, monitoring,
  deployment decision, and sample inference.
- Compile-fail tests that enforce Converge authority boundaries.

## Packs

| Pack | Algorithm family |
|---|---|
| `AnomalyDetectionPack` | Z-score anomaly detection |
| `ClassificationPack` | Logistic classification |
| `DescriptiveStatsPack` | Mean, median, variance, percentiles |
| `ForecastingPack` | Exponential smoothing |
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

While Converge platform crates are unreleased, this workspace patches local
Converge crates at `../../work/converge/crates/...`.

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
