# Changelog

All notable changes to prism will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Cargo package renamed from `prism` to `converge-prism-analytics`; Rust
  library name remains `prism`.

## [1.0.0] - 2026-05-05

### Added

Initial release. Extracted from `converge/crates/analytics` as a Converge extension per [ADR-008](https://github.com/Reflective-Lab/converge/blob/main/kb/Architecture/ADRs/ADR-008-extension-crate-boundaries.md).

- Feature extraction agent (Polars)
- Inference agent (Burn)
- Training pipeline: dataset, validation, feature engineering, hyperparameter search, evaluation, registry, monitoring, deployment
- Analytics packs: anomaly detection, classification, descriptive stats, forecasting, ranking, regression, segmentation, similarity, trend detection
- Optional storage feature (`converge-storage`) and Excel ingestion (`calamine`)

### Changed

- Crate renamed from `converge-analytics` to `prism`
