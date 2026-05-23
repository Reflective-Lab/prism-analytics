// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! Fuzzy inference engine: Mamdani, Sugeno, Tsukamoto.
//!
//! Extracted from `prism-analytics` so it can be reused across Converge
//! extensions (manifold-adapters, ferrox-solvers, helms) without dragging
//! prism's heavy ML dependencies (Polars, Burn) into the dependency tree.
//!
//! # Modules
//!
//! - [`types`] — membership functions, linguistic variables, rules.
//! - [`engine`] — Mamdani-style inference engine (max-aggregation).
//! - [`defuzz`] — defuzzification (centroid, bisector, MoM, height,
//!   weighted-average).
//! - [`sugeno`] — Sugeno (zero/first-order) inference with constant or
//!   linear consequents.
//! - [`tsukamoto`] — Tsukamoto inference (monotonic consequents).
//!
//! Re-exports the full public surface from each submodule at the crate
//! root for ergonomic single-import use:
//!
//! ```rust,ignore
//! use converge_fuzzy_inference::{
//!     FuzzyInferenceEngine, FuzzyInferenceInput, LinguisticVariable,
//!     MembershipFunction, FuzzyRule,
//! };
//! ```

mod defuzz;
mod engine;
mod sugeno;
mod tsukamoto;
mod types;

pub use defuzz::*;
pub use engine::*;
pub use sugeno::*;
pub use tsukamoto::*;
pub use types::*;
