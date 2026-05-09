use serde::{Deserialize, Serialize};

use super::types::{FuzzyInferenceOutput, LinguisticVariable};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum DefuzzMethod {
    Centroid,
    Bisector,
    MeanOfMaxima,
    Height,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Domain {
    pub min: f64,
    pub max: f64,
    pub steps: usize,
}

impl Domain {
    pub fn new(min: f64, max: f64, steps: usize) -> Self {
        Self { min, max, steps }
    }

    fn is_valid(&self) -> bool {
        self.min.is_finite() && self.max.is_finite() && self.min < self.max && self.steps > 0
    }
}

pub fn defuzzify_mamdani(
    output: &FuzzyInferenceOutput,
    variables: &[LinguisticVariable],
    output_variable: &str,
    domain: Domain,
    method: DefuzzMethod,
) -> Option<f64> {
    if !domain.is_valid() {
        return None;
    }

    let variable = variables.iter().find(|v| v.name == output_variable)?;
    let prefix = format!("{output_variable}.");

    let consequents: Vec<(&super::types::FuzzySet, f64)> = output
        .memberships
        .iter()
        .filter_map(|(key, strength)| {
            key.strip_prefix(&prefix).and_then(|set_name| {
                variable
                    .sets
                    .iter()
                    .find(|s| s.name == set_name)
                    .map(|s| (s, *strength))
            })
        })
        .filter(|(_, strength)| *strength > 0.0)
        .collect();

    if consequents.is_empty() {
        return None;
    }

    let dx = (domain.max - domain.min) / (domain.steps as f64);
    let samples: Vec<(f64, f64)> = (0..=domain.steps)
        .map(|i| {
            let x = domain.min + (i as f64) * dx;
            let mu = consequents
                .iter()
                .map(|(set, strength)| set.function.evaluate(x).min(*strength))
                .fold(0.0_f64, f64::max);
            (x, mu)
        })
        .collect();

    match method {
        DefuzzMethod::Centroid => {
            let num: f64 = samples.iter().map(|(x, mu)| x * mu).sum();
            let den: f64 = samples.iter().map(|(_, mu)| *mu).sum();
            if den == 0.0 { None } else { Some(num / den) }
        }
        DefuzzMethod::Bisector => {
            let total: f64 = samples.iter().map(|(_, mu)| *mu).sum();
            if total == 0.0 {
                return None;
            }
            let half = total / 2.0;
            let mut acc = 0.0;
            for (x, mu) in &samples {
                acc += mu;
                if acc >= half {
                    return Some(*x);
                }
            }
            samples.last().map(|(x, _)| *x)
        }
        DefuzzMethod::MeanOfMaxima => {
            let max_mu = samples.iter().map(|(_, mu)| *mu).fold(0.0_f64, f64::max);
            if max_mu == 0.0 {
                return None;
            }
            let xs: Vec<f64> = samples
                .iter()
                .filter(|(_, mu)| (mu - max_mu).abs() < 1e-9)
                .map(|(x, _)| *x)
                .collect();
            if xs.is_empty() {
                None
            } else {
                Some(xs.iter().sum::<f64>() / (xs.len() as f64))
            }
        }
        DefuzzMethod::Height => samples
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(x, _)| *x),
    }
}

pub fn weighted_average(rules: &[(f64, f64)]) -> Option<f64> {
    let den: f64 = rules.iter().map(|(strength, _)| *strength).sum();
    if den == 0.0 || !den.is_finite() {
        return None;
    }
    let num: f64 = rules.iter().map(|(strength, value)| strength * value).sum();
    if !num.is_finite() {
        return None;
    }
    Some(num / den)
}
