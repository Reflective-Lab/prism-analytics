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
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SugenoFunction {
    Constant {
        value: f64,
    },
    Linear {
        intercept: f64,
        coefficients: BTreeMap<String, f64>,
    },
}

impl SugenoFunction {
    pub fn evaluate(&self, inputs: &BTreeMap<String, f64>) -> Result<f64> {
        match self {
            Self::Constant { value } => {
                if !value.is_finite() {
                    return Err(converge_pack::GateError::invalid_input(
                        "sugeno constant must be finite",
                    ));
                }
                Ok(*value)
            }
            Self::Linear {
                intercept,
                coefficients,
            } => {
                if !intercept.is_finite() {
                    return Err(converge_pack::GateError::invalid_input(
                        "sugeno linear intercept must be finite",
                    ));
                }
                let mut value = *intercept;
                for (name, coeff) in coefficients {
                    if !coeff.is_finite() {
                        return Err(converge_pack::GateError::invalid_input(format!(
                            "sugeno coefficient '{name}' must be finite"
                        )));
                    }
                    let Some(input) = inputs.get(name) else {
                        return Err(converge_pack::GateError::invalid_input(format!(
                            "sugeno linear consequent references unknown input '{name}'"
                        )));
                    };
                    value += coeff * input;
                }
                Ok(value)
            }
        }
    }

    pub fn validate(&self, inputs: &BTreeMap<String, f64>) -> Result<()> {
        match self {
            Self::Constant { value } => {
                if !value.is_finite() {
                    return Err(converge_pack::GateError::invalid_input(
                        "sugeno constant must be finite",
                    ));
                }
            }
            Self::Linear {
                intercept,
                coefficients,
            } => {
                if !intercept.is_finite() {
                    return Err(converge_pack::GateError::invalid_input(
                        "sugeno linear intercept must be finite",
                    ));
                }
                for (name, coeff) in coefficients {
                    if name.trim().is_empty() {
                        return Err(converge_pack::GateError::invalid_input(
                            "sugeno coefficient names must be non-empty",
                        ));
                    }
                    if !coeff.is_finite() {
                        return Err(converge_pack::GateError::invalid_input(format!(
                            "sugeno coefficient '{name}' must be finite"
                        )));
                    }
                    if !inputs.contains_key(name) {
                        return Err(converge_pack::GateError::invalid_input(format!(
                            "sugeno linear consequent references unknown input '{name}'"
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SugenoRule {
    pub id: Option<String>,
    #[serde(rename = "if")]
    pub when: FuzzyExpression,
    #[serde(rename = "then")]
    pub then: SugenoFunction,
    pub weight: Option<f64>,
}

impl SugenoRule {
    pub fn weight(&self) -> f64 {
        self.weight.unwrap_or(1.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SugenoInferenceInput {
    pub inputs: BTreeMap<String, f64>,
    pub variables: Vec<LinguisticVariable>,
    pub rules: Vec<SugenoRule>,
}

impl SugenoInferenceInput {
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
                "at least one sugeno rule is required",
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
            rule.then.validate(&self.inputs)?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SugenoActivatedRule {
    pub id: String,
    pub antecedent_strength: f64,
    pub weight: f64,
    pub firing_strength: f64,
    pub consequent_value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SugenoInferenceOutput {
    pub input_memberships: BTreeMap<String, BTreeMap<String, f64>>,
    pub activated_rules: Vec<SugenoActivatedRule>,
    pub output: Option<f64>,
    pub confidence: f64,
    pub total_rules: usize,
}

impl SugenoInferenceOutput {
    pub fn summary(&self) -> String {
        match self.output {
            Some(value) => format!(
                "Evaluated {} sugeno rules, {} fired, output: {:.6}",
                self.total_rules,
                self.activated_rules.len(),
                value
            ),
            None => format!(
                "Evaluated {} sugeno rules, no rules fired",
                self.total_rules
            ),
        }
    }
}

pub struct SugenoInferenceEngine;

impl SugenoInferenceEngine {
    pub fn solve(
        &self,
        input: &SugenoInferenceInput,
        spec: &ProblemSpec,
    ) -> Result<(SugenoInferenceOutput, SolverReport)> {
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

            let consequent_value = rule.then.evaluate(&input.inputs)?;
            if !consequent_value.is_finite() {
                return Err(converge_pack::GateError::invalid_input(format!(
                    "rule {idx} produced a non-finite consequent value"
                )));
            }

            activated_rules.push(SugenoActivatedRule {
                id: rule
                    .id
                    .clone()
                    .unwrap_or_else(|| format!("rule-{}", idx + 1)),
                antecedent_strength,
                weight,
                firing_strength,
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

        let result = SugenoInferenceOutput {
            input_memberships,
            activated_rules,
            output,
            confidence,
            total_rules: input.rules.len(),
        };

        let replay = ReplayEnvelope::minimal(spec.seed());
        let report = SolverReport::optimal("sugeno-inference-v1", confidence, replay);

        Ok((result, report))
    }
}
