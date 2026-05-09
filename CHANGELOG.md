# Changelog

All notable changes to prism will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `prism::fuzzy` reusable fuzzy logic capability plus `FuzzyInferencePack` for
  explainable fuzzy-rule inference over linguistic variables, membership
  functions, activated-rule traces, and graded outputs.
- `MembershipFunction::Gaussian { center, sigma }` — `μ(x) = exp(-((x-c)²)/(2σ²))`
  with `σ > 0` validation. Fifth membership-function shape alongside
  triangular, trapezoidal, and the two shoulders.
- `prism::fuzzy::defuzz` module with `DefuzzMethod` (Centroid, Bisector,
  MeanOfMaxima, Height) and `defuzzify_mamdani(output, variables, output_var,
  domain, method)`. Defuzzification is a separate opt-in pass on Mamdani
  output so the activated-rule trace is preserved on the original output.
  `weighted_average(rules)` exposed for callers with crisp `(strength, value)`
  pairs.
- `prism::fuzzy::sugeno` Sugeno (Takagi–Sugeno) FIS — `SugenoFunction`
  (Constant, Linear) consequents, `SugenoRule`, `SugenoInferenceInput`,
  `SugenoInferenceOutput` with per-rule firing strengths and consequent
  values, `SugenoInferenceEngine`. Output is a single weighted-average crisp
  value when at least one rule fires.
- `SugenoInferencePack` — Converge pack adapter for Sugeno inference, with
  invariants: `valid-output` (critical, finite output when rules fire),
  `valid-memberships` (critical, inputs in [0,1]), `rule-activation`
  (advisory, at least one rule fires).

## [1.1.0] - 2026-05-07

### Changed

- Cargo package renamed from `prism` to `converge-prism-analytics`; Rust
  library name remains `prism`.
- Workspace `[lints.clippy]` extends the existing pedantic-allow list with
  `default_trait_access`, `struct_field_names`, `unreadable_literal`,
  `manual_let_else`, `items_after_statements`, `return_self_not_must_use`,
  `ignore_without_reason`, and `float_cmp`. These remain `pedantic` warnings
  upstream — we simply do not gate the release on stylistic noise.
- `Justfile`'s `security-audit` recipe now passes the same
  `--ignore RUSTSEC-*` flags `cargo-deny` already ignores in `deny.toml`,
  so the local and CI gates agree.

### Fixed

- `cargo clippy --fix` cleanups across `engine`, `ingest`, `model`,
  `training`, and the analytics packs: `Option::map_or`, `Default` use,
  collapsed `if let` chains, and `String::new()` over `"".into()`.
- `tests/compile_fail/*.stderr` snapshots refreshed for current rustc
  diagnostics (now lists both `CorrectionTarget::Fact` and
  `OverrideTarget::Fact` in the help text).
- `deny.toml` extended with `MPL-2.0` / `NCSA` allow entries so the
  license gate matches the foundation's allowed list.

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
