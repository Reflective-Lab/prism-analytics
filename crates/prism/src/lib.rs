// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! # prism
//!
//! Closed-form analytics and inference Suggestors for the Converge Engine.
//!
//! Prism owns hand-authored, deterministic inference: feature extraction,
//! inference packs, and fuzzy inference (Mamdani / Sugeno / Tsukamoto).
//! Training-pipeline concerns — dataset loading, train/val split,
//! hyperparameter search, model fitting, registry, and deployment —
//! live in `crucible-models`. The boundary is: prism never fits, crucible
//! never owns expert rules.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use prism::FeatureAgent;
//!
//! engine.register_suggestor(FeatureAgent::new(config));
//! ```
//!
//! ## Available Suggestors
//!
//! - [`FeatureAgent`] — Polars-based feature extraction
//! - [`InferenceAgent`] — Burn-based inference over feature vectors
//! - [`FuzzyInferencePack`] — fuzzy membership and rule inference

pub mod batch;
pub mod engine;
pub mod fuzzy;
pub mod model;
pub mod packs;
pub mod provenance;
pub mod suggestor;

pub use engine::FeatureAgent;
pub use model::InferenceAgent;
pub use packs::{
    AnomalyDetectionPack, ClassificationPack, DescriptiveStatsPack, ForecastingPack,
    FuzzyInferencePack, NaiveBayesPack, RankingPack, RegressionPack, SegmentationPack,
    SimilarityPack, TrendDetectionPack,
};
pub use provenance::{PRISM_PROVENANCE, ProvenanceSource, UnknownProvenanceSource};
