// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! # prism
//!
//! ML and analytics capabilities as Suggestors for the Converge Engine.
//!
//! Access through [`FeatureAgent`] and the training pipeline agents.
//! Register them in a formation for data-driven convergence.
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
//! - Training pipeline suggestors — dataset, validation, feature engineering,
//!   training, evaluation, registry, monitoring, deployment

pub mod batch;
pub mod engine;
pub mod fuzzy;
pub mod ingest;
pub mod model;
pub mod packs;
#[cfg(feature = "storage")]
pub mod storage;
pub mod suggestor;
pub mod training;

pub use engine::FeatureAgent;
pub use model::InferenceAgent;
pub use packs::{
    AnomalyDetectionPack, ClassificationPack, DescriptiveStatsPack, ForecastingPack,
    FuzzyInferencePack, NaiveBayesPack, RankingPack, RegressionPack, SegmentationPack,
    SimilarityPack, TrendDetectionPack,
};
pub use training::{
    DataValidationAgent, DatasetAgent, DeploymentAgent, FeatureEngineeringAgent,
    HyperparameterSearchAgent, ModelEvaluationAgent, ModelRegistryAgent, ModelTrainingAgent,
    MonitoringAgent, SampleInferenceAgent,
};
