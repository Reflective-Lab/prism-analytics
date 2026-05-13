use super::types::*;
use converge_pack::gate::GateResult as Result;
use converge_pack::gate::{ProblemSpec, ReplayEnvelope, SolverReport};
use std::f64::consts::TAU;

pub struct GaussianNaiveBayes;

impl GaussianNaiveBayes {
    pub fn solve(
        &self,
        input: &NaiveBayesInput,
        spec: &ProblemSpec,
    ) -> Result<(NaiveBayesOutput, SolverReport)> {
        // log P(C|x) ∝ log P(C) + Σ log N(x_i; μ_{C,i}, σ_{C,i})
        // log N(x; μ, σ) = -0.5·ln(2πσ²) - (x-μ)²/(2σ²)
        let log_scores: Vec<(String, f64)> = input
            .classes
            .iter()
            .map(|class| {
                let log_prior = class.prior.ln();
                let log_likelihood: f64 = class
                    .feature_params
                    .iter()
                    .zip(&input.features)
                    .map(|(params, x)| {
                        let variance = params.std_dev * params.std_dev;
                        -0.5 * (TAU * variance).ln() - (x - params.mean).powi(2) / (2.0 * variance)
                    })
                    .sum();
                (class.name.clone(), log_prior + log_likelihood)
            })
            .collect();

        // Softmax: subtract max for numerical stability
        let max_score = log_scores
            .iter()
            .map(|(_, s)| *s)
            .fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = log_scores
            .iter()
            .map(|(_, s)| (s - max_score).exp())
            .collect();
        let sum_exp: f64 = exp_scores.iter().sum();

        let mut probabilities: Vec<ClassProbability> = log_scores
            .iter()
            .zip(exp_scores.iter())
            .map(|((name, log_score), exp_s)| ClassProbability {
                class: name.clone(),
                probability: exp_s / sum_exp,
                log_score: *log_score,
            })
            .collect();

        probabilities.sort_by(|a, b| {
            b.probability
                .partial_cmp(&a.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let predicted = probabilities[0].class.clone();
        let confidence = probabilities[0].probability;

        let output = NaiveBayesOutput {
            predicted,
            confidence,
            probabilities,
        };

        let replay = ReplayEnvelope::minimal(spec.seed());
        let report = SolverReport::optimal("gaussian-naive-bayes-v1", confidence, replay);

        Ok((output, report))
    }
}
