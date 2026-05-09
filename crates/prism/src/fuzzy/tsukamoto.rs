use std::collections::BTreeMap;

use converge_pack::gate::GateResult as Result;
use converge_pack::gate::{ProblemSpec, ReplayEnvelope, SolverReport};
use serde::{Deserialize, Serialize};

use super::types::{
    FuzzyExpression, LinguisticVariable, evaluate_expression, evaluate_input_memberships,
    validate_expression, validate_variables,
};
use super::weighted_average;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TsukamotoConsequent {
    pub variable: String,
    pub set: String,
}

impl TsukamotoConsequent {
    pub fn key(&self) -> String {
        format!("{}.{}", self.variable, self.set)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TsukamotoRule {
    pub id: Option<String>,
    #[serde(rename = "if")]
    pub when: FuzzyExpression,
    #[serde(rename = "then")]
    pub then: TsukamotoConsequent,
    pub weight: Option<f64>,
}

impl TsukamotoRule {
    pub fn weight(&self) -> f64 {
        self.weight.unwrap_or(1.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TsukamotoInferenceInput {
    pub inputs: BTreeMap<String, f64>,
    pub variables: Vec<LinguisticVariable>,
    pub rules: Vec<TsukamotoRule>,
}

impl TsukamotoInferenceInput {
    pub fn validate(&self) -> Result<()> {
        if self.inputs.is_empty() {
            return Err(converge_pack::GateError::invalid_input(
                "at least one crisp input is required",
            ));
        }
        if self.variables.is_empty() {
            return Err(converge_pack::GateError::invalid_input(
                "at least one linguistic variable is required",
            ));
        }
        if self.rules.is_empty() {
            return Err(converge_pack::GateError::invalid_input(
                "at least one tsukamoto rule is required",
            ));
        }

        for (name, value) in &self.inputs {
            if name.trim().is_empty() || !value.is_finite() {
                return Err(converge_pack::GateError::invalid_input(
                    "input names must be non-empty and values must be finite",
                ));
            }
        }

        let variable_sets = validate_variables(&self.variables)?;

        for input_name in self.inputs.keys() {
            if !variable_sets.contains_key(input_name.as_str()) {
                return Err(converge_pack::GateError::invalid_input(format!(
                    "input variable '{input_name}' is not defined"
                )));
            }
        }

        for (idx, rule) in self.rules.iter().enumerate() {
            if let Some(id) = &rule.id
                && id.trim().is_empty()
            {
                return Err(converge_pack::GateError::invalid_input(format!(
                    "rule {idx} has an empty id"
                )));
            }
            let weight = rule.weight();
            if !(0.0..=1.0).contains(&weight) || !weight.is_finite() {
                return Err(converge_pack::GateError::invalid_input(format!(
                    "rule {idx} weight must be finite and in [0, 1]"
                )));
            }
            validate_expression(&rule.when, &variable_sets, &self.inputs)?;

            let consequent_var = self
                .variables
                .iter()
                .find(|v| v.name == rule.then.variable)
                .ok_or_else(|| {
                    converge_pack::GateError::invalid_input(format!(
                        "rule {idx} consequent references unknown variable '{}'",
                        rule.then.variable
                    ))
                })?;
            let consequent_set = consequent_var
                .sets
                .iter()
                .find(|s| s.name == rule.then.set)
                .ok_or_else(|| {
                    converge_pack::GateError::invalid_input(format!(
                        "rule {idx} consequent references unknown set '{}' on '{}'",
                        rule.then.set, rule.then.variable
                    ))
                })?;
            if !consequent_set.function.is_monotonic() {
                return Err(converge_pack::GateError::invalid_input(format!(
                    "rule {idx} consequent '{}.{}' uses a non-monotonic membership \
                     function; tsukamoto requires monotonic consequents \
                     (left/right shoulder)",
                    rule.then.variable, rule.then.set
                )));
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TsukamotoActivatedRule {
    pub id: String,
    pub antecedent_strength: f64,
    pub weight: f64,
    pub firing_strength: f64,
    pub consequent: String,
    pub consequent_value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TsukamotoInferenceOutput {
    pub input_memberships: BTreeMap<String, BTreeMap<String, f64>>,
    pub activated_rules: Vec<TsukamotoActivatedRule>,
    pub output: Option<f64>,
    pub confidence: f64,
    pub total_rules: usize,
}

impl TsukamotoInferenceOutput {
    pub fn summary(&self) -> String {
        match self.output {
            Some(value) => format!(
                "Evaluated {} tsukamoto rules, {} fired, output: {:.6}",
                self.total_rules,
                self.activated_rules.len(),
                value
            ),
            None => format!(
                "Evaluated {} tsukamoto rules, no rules fired",
                self.total_rules
            ),
        }
    }
}

pub struct TsukamotoInferenceEngine;

impl TsukamotoInferenceEngine {
    pub fn solve(
        &self,
        input: &TsukamotoInferenceInput,
        spec: &ProblemSpec,
    ) -> Result<(TsukamotoInferenceOutput, SolverReport)> {
        input.validate()?;

        let input_memberships = evaluate_input_memberships(&input.inputs, &input.variables);
        let mut activated_rules = Vec::new();
        let mut weighted_pairs: Vec<(f64, f64)> = Vec::new();
        let mut max_firing_strength = 0.0_f64;

        for (idx, rule) in input.rules.iter().enumerate() {
            let antecedent_strength = evaluate_expression(&rule.when, &input_memberships)?;
            let weight = rule.weight();
            let firing_strength = (antecedent_strength * weight).clamp(0.0, 1.0);

            if firing_strength <= 0.0 {
                continue;
            }

            // Find the consequent set's MF and invert it at the firing strength.
            let consequent_var = input
                .variables
                .iter()
                .find(|v| v.name == rule.then.variable)
                .expect("validate ensures the variable exists");
            let consequent_set = consequent_var
                .sets
                .iter()
                .find(|s| s.name == rule.then.set)
                .expect("validate ensures the set exists");
            let consequent_value = consequent_set.function.inverse(firing_strength)?;

            if !consequent_value.is_finite() {
                return Err(converge_pack::GateError::invalid_input(format!(
                    "rule {idx} produced a non-finite consequent value"
                )));
            }

            activated_rules.push(TsukamotoActivatedRule {
                id: rule
                    .id
                    .clone()
                    .unwrap_or_else(|| format!("rule-{}", idx + 1)),
                antecedent_strength,
                weight,
                firing_strength,
                consequent: rule.then.key(),
                consequent_value,
            });
            weighted_pairs.push((firing_strength, consequent_value));
            max_firing_strength = max_firing_strength.max(firing_strength);
        }

        let output = weighted_average(&weighted_pairs);
        let confidence = if output.is_some() {
            max_firing_strength
        } else {
            0.0
        };

        let result = TsukamotoInferenceOutput {
            input_memberships,
            activated_rules,
            output,
            confidence,
            total_rules: input.rules.len(),
        };

        let replay = ReplayEnvelope::minimal(spec.seed());
        let report = SolverReport::optimal("tsukamoto-inference-v1", confidence, replay);

        Ok((result, report))
    }
}
