use std::collections::{BTreeMap, BTreeSet};

use converge_pack::gate::GateResult as Result;
use serde::{Deserialize, Serialize};

/// A fuzzy membership degree in [0.0, 1.0]. The constructor clamps the input.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MembershipDegree(f64);

impl MembershipDegree {
    pub fn new(v: f64) -> Self {
        Self(v.clamp(0.0, 1.0))
    }
    pub fn value(self) -> f64 {
        self.0
    }
    pub fn zero() -> Self {
        Self(0.0)
    }
    pub fn one() -> Self {
        Self(1.0)
    }
}

impl From<MembershipDegree> for f64 {
    fn from(m: MembershipDegree) -> f64 {
        m.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzySet {
    pub name: String,
    pub function: MembershipFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinguisticVariable {
    pub name: String,
    pub sets: Vec<FuzzySet>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MembershipFunction {
    Triangular {
        min: f64,
        peak: f64,
        max: f64,
    },
    Trapezoidal {
        min: f64,
        lower_peak: f64,
        upper_peak: f64,
        max: f64,
    },
    LeftShoulder {
        start: f64,
        end: f64,
    },
    RightShoulder {
        start: f64,
        end: f64,
    },
    Gaussian {
        center: f64,
        sigma: f64,
    },
}

impl MembershipFunction {
    pub fn evaluate(&self, value: f64) -> MembershipDegree {
        let membership = match *self {
            Self::Triangular { min, peak, max } => {
                if value <= min || value >= max {
                    0.0
                } else if value == peak {
                    1.0
                } else if value < peak {
                    (value - min) / (peak - min)
                } else {
                    (max - value) / (max - peak)
                }
            }
            Self::Trapezoidal {
                min,
                lower_peak,
                upper_peak,
                max,
            } => {
                if value <= min || value >= max {
                    0.0
                } else if (lower_peak..=upper_peak).contains(&value) {
                    1.0
                } else if value < lower_peak {
                    (value - min) / (lower_peak - min)
                } else {
                    (max - value) / (max - upper_peak)
                }
            }
            Self::LeftShoulder { start, end } => {
                if value <= start {
                    1.0
                } else if value >= end {
                    0.0
                } else {
                    (end - value) / (end - start)
                }
            }
            Self::RightShoulder { start, end } => {
                if value <= start {
                    0.0
                } else if value >= end {
                    1.0
                } else {
                    (value - start) / (end - start)
                }
            }
            Self::Gaussian { center, sigma } => {
                let dx = value - center;
                (-(dx * dx) / (2.0 * sigma * sigma)).exp()
            }
        };

        MembershipDegree::new(membership)
    }

    /// Whether this membership function is monotonic over its support.
    ///
    /// Tsukamoto inference requires monotonic consequent MFs so the firing
    /// strength can be uniquely inverted to a crisp consequent value.
    pub fn is_monotonic(&self) -> bool {
        matches!(self, Self::LeftShoulder { .. } | Self::RightShoulder { .. })
    }

    /// Inverse of the membership function: given a target membership in
    /// `[0, 1]`, return the crisp `x` such that `μ(x) = target`.
    ///
    /// Defined only for monotonic shapes. Returns `Err` for non-monotonic
    /// MFs (Triangular, Trapezoidal, Gaussian) where the inverse is not
    /// unique.
    pub fn inverse(&self, target: f64) -> Result<f64> {
        if !target.is_finite() || !(0.0..=1.0).contains(&target) {
            return Err(converge_pack::GateError::invalid_input(
                "inverse target must be a finite membership in [0, 1]",
            ));
        }
        match *self {
            Self::LeftShoulder { start, end } => {
                // μ(x) = 1 for x ≤ start, ramps 1 → 0 across [start, end], 0 for x ≥ end.
                // Inverse: x = end - target × (end - start).
                Ok(end - target * (end - start))
            }
            Self::RightShoulder { start, end } => {
                // μ(x) = 0 for x ≤ start, ramps 0 → 1 across [start, end], 1 for x ≥ end.
                // Inverse: x = start + target × (end - start).
                Ok(start + target * (end - start))
            }
            Self::Triangular { .. } | Self::Trapezoidal { .. } | Self::Gaussian { .. } => {
                Err(converge_pack::GateError::invalid_input(
                    "inverse is only defined for monotonic membership functions \
                     (left/right shoulder)",
                ))
            }
        }
    }

    pub fn validate(&self) -> Result<()> {
        match *self {
            Self::Triangular { min, peak, max } => {
                validate_finite(&[min, peak, max])?;
                if !(min < peak && peak < max) {
                    return Err(converge_pack::GateError::invalid_input(
                        "triangular membership requires min < peak < max",
                    ));
                }
            }
            Self::Trapezoidal {
                min,
                lower_peak,
                upper_peak,
                max,
            } => {
                validate_finite(&[min, lower_peak, upper_peak, max])?;
                if !(min < lower_peak && lower_peak <= upper_peak && upper_peak < max) {
                    return Err(converge_pack::GateError::invalid_input(
                        "trapezoidal membership requires min < lower_peak <= upper_peak < max",
                    ));
                }
            }
            Self::LeftShoulder { start, end } | Self::RightShoulder { start, end } => {
                validate_finite(&[start, end])?;
                if start >= end {
                    return Err(converge_pack::GateError::invalid_input(
                        "shoulder membership requires start < end",
                    ));
                }
            }
            Self::Gaussian { center, sigma } => {
                validate_finite(&[center, sigma])?;
                if sigma <= 0.0 {
                    return Err(converge_pack::GateError::invalid_input(
                        "gaussian membership requires sigma > 0",
                    ));
                }
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum FuzzyExpression {
    Is { variable: String, set: String },
    And { terms: Vec<FuzzyExpression> },
    Or { terms: Vec<FuzzyExpression> },
    Not { term: Box<FuzzyExpression> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyConsequent {
    pub variable: String,
    pub set: String,
}

impl FuzzyConsequent {
    pub fn key(&self) -> String {
        format!("{}.{}", self.variable, self.set)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyRule {
    pub id: Option<String>,
    #[serde(rename = "if")]
    pub when: FuzzyExpression,
    #[serde(rename = "then")]
    pub then: FuzzyConsequent,
    pub weight: Option<MembershipDegree>,
}

impl FuzzyRule {
    pub fn weight(&self) -> MembershipDegree {
        self.weight.unwrap_or_else(MembershipDegree::one)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyInferenceInput {
    pub inputs: BTreeMap<String, f64>,
    pub variables: Vec<LinguisticVariable>,
    pub rules: Vec<FuzzyRule>,
}

impl FuzzyInferenceInput {
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
                "at least one fuzzy rule is required",
            ));
        }

        for (name, value) in &self.inputs {
            if name.trim().is_empty() || !value.is_finite() {
                return Err(converge_pack::GateError::invalid_input(
                    "input names must be non-empty and values must be finite",
                ));
            }
        }

        let variable_sets = self.validate_variables()?;

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
            // weight() returns MembershipDegree which is always in [0,1]; validate
            // that the raw serialized value (if present) was finite.
            if let Some(w) = rule.weight
                && !w.value().is_finite()
            {
                return Err(converge_pack::GateError::invalid_input(format!(
                    "rule {idx} weight must be finite and in [0, 1]"
                )));
            }
            validate_expression(&rule.when, &variable_sets, &self.inputs)?;
            validate_consequent(&rule.then, &variable_sets)?;
        }

        Ok(())
    }

    fn validate_variables(&self) -> Result<BTreeMap<&str, BTreeSet<&str>>> {
        validate_variables(&self.variables)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivatedRule {
    pub id: String,
    pub antecedent_strength: MembershipDegree,
    pub weight: MembershipDegree,
    pub strength: MembershipDegree,
    pub consequent: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyInferenceOutput {
    pub input_memberships: BTreeMap<String, BTreeMap<String, MembershipDegree>>,
    pub memberships: BTreeMap<String, MembershipDegree>,
    pub activated_rules: Vec<ActivatedRule>,
    pub confidence: MembershipDegree,
    pub total_rules: usize,
}

impl FuzzyInferenceOutput {
    pub fn summary(&self) -> String {
        let top = self
            .memberships
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal));

        match top {
            Some((key, value)) => format!(
                "Evaluated {} fuzzy rules, top membership: {}={:.3}",
                self.total_rules,
                key,
                value.value()
            ),
            None => format!(
                "Evaluated {} fuzzy rules, no memberships fired",
                self.total_rules
            ),
        }
    }
}

pub(super) fn validate_variables(
    variables: &[LinguisticVariable],
) -> Result<BTreeMap<&str, BTreeSet<&str>>> {
    let mut variable_names = BTreeSet::new();
    let mut variable_sets = BTreeMap::new();

    for variable in variables {
        if variable.name.trim().is_empty() {
            return Err(converge_pack::GateError::invalid_input(
                "linguistic variable names must be non-empty",
            ));
        }
        if !variable_names.insert(variable.name.as_str()) {
            return Err(converge_pack::GateError::invalid_input(format!(
                "duplicate linguistic variable '{}'",
                variable.name
            )));
        }
        if variable.sets.is_empty() {
            return Err(converge_pack::GateError::invalid_input(format!(
                "linguistic variable '{}' must define at least one set",
                variable.name
            )));
        }

        let mut set_names = BTreeSet::new();
        for set in &variable.sets {
            if set.name.trim().is_empty() {
                return Err(converge_pack::GateError::invalid_input(format!(
                    "variable '{}' has an empty set name",
                    variable.name
                )));
            }
            if !set_names.insert(set.name.as_str()) {
                return Err(converge_pack::GateError::invalid_input(format!(
                    "duplicate set '{}' on variable '{}'",
                    set.name, variable.name
                )));
            }
            set.function.validate()?;
        }
        variable_sets.insert(variable.name.as_str(), set_names);
    }

    Ok(variable_sets)
}

pub(super) fn validate_expression(
    expression: &FuzzyExpression,
    variable_sets: &BTreeMap<&str, BTreeSet<&str>>,
    inputs: &BTreeMap<String, f64>,
) -> Result<()> {
    match expression {
        FuzzyExpression::Is { variable, set } => {
            validate_variable_set(variable, set, variable_sets)?;
            if !inputs.contains_key(variable) {
                return Err(converge_pack::GateError::invalid_input(format!(
                    "antecedent variable '{variable}' has no crisp input"
                )));
            }
        }
        FuzzyExpression::And { terms } | FuzzyExpression::Or { terms } => {
            if terms.is_empty() {
                return Err(converge_pack::GateError::invalid_input(
                    "and/or expressions require at least one term",
                ));
            }
            for term in terms {
                validate_expression(term, variable_sets, inputs)?;
            }
        }
        FuzzyExpression::Not { term } => validate_expression(term, variable_sets, inputs)?,
    }

    Ok(())
}

fn validate_consequent(
    consequent: &FuzzyConsequent,
    variable_sets: &BTreeMap<&str, BTreeSet<&str>>,
) -> Result<()> {
    validate_variable_set(&consequent.variable, &consequent.set, variable_sets)
}

fn validate_variable_set(
    variable: &str,
    set: &str,
    variable_sets: &BTreeMap<&str, BTreeSet<&str>>,
) -> Result<()> {
    let Some(sets) = variable_sets.get(variable) else {
        return Err(converge_pack::GateError::invalid_input(format!(
            "unknown variable '{variable}'"
        )));
    };
    if !sets.contains(set) {
        return Err(converge_pack::GateError::invalid_input(format!(
            "unknown set '{set}' on variable '{variable}'"
        )));
    }

    Ok(())
}

fn validate_finite(values: &[f64]) -> Result<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(converge_pack::GateError::invalid_input(
            "membership function bounds must be finite",
        ))
    }
}

pub(super) fn evaluate_input_memberships(
    inputs: &BTreeMap<String, f64>,
    variables: &[LinguisticVariable],
) -> BTreeMap<String, BTreeMap<String, MembershipDegree>> {
    let mut memberships = BTreeMap::new();

    for variable in variables {
        let Some(value) = inputs.get(&variable.name) else {
            continue;
        };

        let sets = variable
            .sets
            .iter()
            .map(|set| (set.name.clone(), set.function.evaluate(*value)))
            .collect();
        memberships.insert(variable.name.clone(), sets);
    }

    memberships
}

pub(super) fn evaluate_expression(
    expression: &FuzzyExpression,
    memberships: &BTreeMap<String, BTreeMap<String, MembershipDegree>>,
) -> Result<MembershipDegree> {
    match expression {
        FuzzyExpression::Is { variable, set } => memberships
            .get(variable)
            .and_then(|sets| sets.get(set))
            .copied()
            .ok_or_else(|| {
                converge_pack::GateError::invalid_input(format!(
                    "membership '{variable}.{set}' is not available"
                ))
            }),
        FuzzyExpression::And { terms } => {
            let mut value = MembershipDegree::one();
            for term in terms {
                let t = evaluate_expression(term, memberships)?;
                if t < value {
                    value = t;
                }
            }
            Ok(value)
        }
        FuzzyExpression::Or { terms } => {
            let mut value = MembershipDegree::zero();
            for term in terms {
                let t = evaluate_expression(term, memberships)?;
                if t > value {
                    value = t;
                }
            }
            Ok(value)
        }
        FuzzyExpression::Not { term } => Ok(MembershipDegree::new(
            1.0 - evaluate_expression(term, memberships)?.value(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    // ── MembershipFunction::evaluate ─────────────────────────────────────────

    #[test]
    fn triangular_peak_is_one() {
        let mf = MembershipFunction::Triangular {
            min: 0.0,
            peak: 0.5,
            max: 1.0,
        };
        assert!((mf.evaluate(0.5).value() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn triangular_rising_midpoint() {
        let mf = MembershipFunction::Triangular {
            min: 0.0,
            peak: 0.5,
            max: 1.0,
        };
        assert!((mf.evaluate(0.25).value() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn triangular_falling_midpoint() {
        let mf = MembershipFunction::Triangular {
            min: 0.0,
            peak: 0.5,
            max: 1.0,
        };
        assert!((mf.evaluate(0.75).value() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn triangular_boundaries_are_zero() {
        let mf = MembershipFunction::Triangular {
            min: 0.0,
            peak: 0.5,
            max: 1.0,
        };
        assert_eq!(mf.evaluate(0.0), MembershipDegree::zero());
        assert_eq!(mf.evaluate(1.0), MembershipDegree::zero());
        assert_eq!(mf.evaluate(-1.0), MembershipDegree::zero());
    }

    #[test]
    fn trapezoidal_plateau_is_one() {
        let mf = MembershipFunction::Trapezoidal {
            min: 0.0,
            lower_peak: 0.3,
            upper_peak: 0.7,
            max: 1.0,
        };
        assert_eq!(mf.evaluate(0.3), MembershipDegree::one());
        assert_eq!(mf.evaluate(0.5), MembershipDegree::one());
        assert_eq!(mf.evaluate(0.7), MembershipDegree::one());
    }

    #[test]
    fn trapezoidal_rising_ramp() {
        let mf = MembershipFunction::Trapezoidal {
            min: 0.0,
            lower_peak: 0.4,
            upper_peak: 0.6,
            max: 1.0,
        };
        assert!((mf.evaluate(0.2).value() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn trapezoidal_falling_ramp() {
        let mf = MembershipFunction::Trapezoidal {
            min: 0.0,
            lower_peak: 0.4,
            upper_peak: 0.6,
            max: 1.0,
        };
        assert!((mf.evaluate(0.8).value() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn trapezoidal_boundaries_are_zero() {
        let mf = MembershipFunction::Trapezoidal {
            min: 0.0,
            lower_peak: 0.3,
            upper_peak: 0.7,
            max: 1.0,
        };
        assert_eq!(mf.evaluate(0.0), MembershipDegree::zero());
        assert_eq!(mf.evaluate(1.0), MembershipDegree::zero());
    }

    #[test]
    fn left_shoulder_full_at_and_below_start() {
        let mf = MembershipFunction::LeftShoulder {
            start: 0.3,
            end: 0.7,
        };
        assert_eq!(mf.evaluate(0.0), MembershipDegree::one());
        assert_eq!(mf.evaluate(0.3), MembershipDegree::one());
    }

    #[test]
    fn left_shoulder_midpoint() {
        let mf = MembershipFunction::LeftShoulder {
            start: 0.3,
            end: 0.7,
        };
        assert!((mf.evaluate(0.5).value() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn left_shoulder_zero_at_and_beyond_end() {
        let mf = MembershipFunction::LeftShoulder {
            start: 0.3,
            end: 0.7,
        };
        assert_eq!(mf.evaluate(0.7), MembershipDegree::zero());
        assert_eq!(mf.evaluate(1.0), MembershipDegree::zero());
    }

    #[test]
    fn right_shoulder_zero_at_and_below_start() {
        let mf = MembershipFunction::RightShoulder {
            start: 0.3,
            end: 0.7,
        };
        assert_eq!(mf.evaluate(0.0), MembershipDegree::zero());
        assert_eq!(mf.evaluate(0.3), MembershipDegree::zero());
    }

    #[test]
    fn right_shoulder_midpoint() {
        let mf = MembershipFunction::RightShoulder {
            start: 0.3,
            end: 0.7,
        };
        assert!((mf.evaluate(0.5).value() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn right_shoulder_full_at_and_beyond_end() {
        let mf = MembershipFunction::RightShoulder {
            start: 0.3,
            end: 0.7,
        };
        assert_eq!(mf.evaluate(0.7), MembershipDegree::one());
        assert_eq!(mf.evaluate(1.0), MembershipDegree::one());
    }

    #[test]
    fn gaussian_center_is_one() {
        let mf = MembershipFunction::Gaussian {
            center: 0.5,
            sigma: 0.1,
        };
        assert!((mf.evaluate(0.5).value() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn gaussian_one_sigma_decays() {
        let mf = MembershipFunction::Gaussian {
            center: 0.0,
            sigma: 1.0,
        };
        assert!((mf.evaluate(1.0).value() - (-0.5_f64).exp()).abs() < 1e-10);
    }

    // ── MembershipFunction::validate ─────────────────────────────────────────

    #[test]
    fn triangular_validate_ok() {
        assert!(
            MembershipFunction::Triangular {
                min: 0.0,
                peak: 0.5,
                max: 1.0
            }
            .validate()
            .is_ok()
        );
    }

    #[test]
    fn triangular_validate_rejects_min_eq_peak() {
        assert!(
            MembershipFunction::Triangular {
                min: 0.5,
                peak: 0.5,
                max: 1.0
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn triangular_validate_rejects_peak_eq_max() {
        assert!(
            MembershipFunction::Triangular {
                min: 0.0,
                peak: 1.0,
                max: 1.0
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn trapezoidal_validate_equal_peaks_ok() {
        assert!(
            MembershipFunction::Trapezoidal {
                min: 0.0,
                lower_peak: 0.5,
                upper_peak: 0.5,
                max: 1.0
            }
            .validate()
            .is_ok()
        );
    }

    #[test]
    fn trapezoidal_validate_rejects_inverted_peaks() {
        assert!(
            MembershipFunction::Trapezoidal {
                min: 0.0,
                lower_peak: 0.7,
                upper_peak: 0.3,
                max: 1.0
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn shoulder_validate_rejects_start_ge_end() {
        assert!(
            MembershipFunction::LeftShoulder {
                start: 0.7,
                end: 0.3
            }
            .validate()
            .is_err()
        );
        assert!(
            MembershipFunction::RightShoulder {
                start: 0.5,
                end: 0.5
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn gaussian_validate_rejects_zero_sigma() {
        assert!(
            MembershipFunction::Gaussian {
                center: 0.5,
                sigma: 0.0
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn gaussian_validate_rejects_negative_sigma() {
        assert!(
            MembershipFunction::Gaussian {
                center: 0.5,
                sigma: -0.1
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn validate_rejects_non_finite_bounds() {
        assert!(
            MembershipFunction::Triangular {
                min: f64::NAN,
                peak: 0.5,
                max: 1.0
            }
            .validate()
            .is_err()
        );
        assert!(
            MembershipFunction::Gaussian {
                center: 0.0,
                sigma: f64::INFINITY
            }
            .validate()
            .is_err()
        );
    }

    // ── is_monotonic ──────────────────────────────────────────────────────────

    #[test]
    fn shoulder_mfs_are_monotonic() {
        assert!(
            MembershipFunction::LeftShoulder {
                start: 0.0,
                end: 1.0
            }
            .is_monotonic()
        );
        assert!(
            MembershipFunction::RightShoulder {
                start: 0.0,
                end: 1.0
            }
            .is_monotonic()
        );
    }

    #[test]
    fn non_shoulder_mfs_are_not_monotonic() {
        assert!(
            !MembershipFunction::Triangular {
                min: 0.0,
                peak: 0.5,
                max: 1.0
            }
            .is_monotonic()
        );
        assert!(
            !MembershipFunction::Trapezoidal {
                min: 0.0,
                lower_peak: 0.3,
                upper_peak: 0.7,
                max: 1.0
            }
            .is_monotonic()
        );
        assert!(
            !MembershipFunction::Gaussian {
                center: 0.5,
                sigma: 0.1
            }
            .is_monotonic()
        );
    }

    // ── inverse ───────────────────────────────────────────────────────────────

    #[test]
    fn left_shoulder_inverse_endpoints() {
        let mf = MembershipFunction::LeftShoulder {
            start: 0.3,
            end: 0.7,
        };
        assert!((mf.inverse(0.0).unwrap() - 0.7).abs() < 1e-10);
        assert!((mf.inverse(1.0).unwrap() - 0.3).abs() < 1e-10);
    }

    #[test]
    fn left_shoulder_inverse_midpoint() {
        let mf = MembershipFunction::LeftShoulder {
            start: 0.3,
            end: 0.7,
        };
        assert!((mf.inverse(0.5).unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn right_shoulder_inverse_endpoints() {
        let mf = MembershipFunction::RightShoulder {
            start: 0.3,
            end: 0.7,
        };
        assert!((mf.inverse(0.0).unwrap() - 0.3).abs() < 1e-10);
        assert!((mf.inverse(1.0).unwrap() - 0.7).abs() < 1e-10);
    }

    #[test]
    fn right_shoulder_inverse_midpoint() {
        let mf = MembershipFunction::RightShoulder {
            start: 0.3,
            end: 0.7,
        };
        assert!((mf.inverse(0.5).unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn non_monotonic_inverse_returns_err() {
        assert!(
            MembershipFunction::Triangular {
                min: 0.0,
                peak: 0.5,
                max: 1.0
            }
            .inverse(0.5)
            .is_err()
        );
        assert!(
            MembershipFunction::Trapezoidal {
                min: 0.0,
                lower_peak: 0.3,
                upper_peak: 0.7,
                max: 1.0
            }
            .inverse(0.5)
            .is_err()
        );
        assert!(
            MembershipFunction::Gaussian {
                center: 0.5,
                sigma: 0.1
            }
            .inverse(0.5)
            .is_err()
        );
    }

    #[test]
    fn inverse_rejects_out_of_range_target() {
        let mf = MembershipFunction::LeftShoulder {
            start: 0.0,
            end: 1.0,
        };
        assert!(mf.inverse(2.0).is_err());
        assert!(mf.inverse(-0.1).is_err());
        assert!(mf.inverse(f64::NAN).is_err());
    }

    // ── evaluate_expression ───────────────────────────────────────────────────

    fn sample_memberships() -> BTreeMap<String, BTreeMap<String, MembershipDegree>> {
        let mut outer = BTreeMap::new();
        let mut sets = BTreeMap::new();
        sets.insert("hot".to_string(), MembershipDegree::new(0.8));
        sets.insert("cold".to_string(), MembershipDegree::new(0.2));
        outer.insert("temperature".to_string(), sets);
        outer
    }

    #[test]
    fn is_expression_returns_membership() {
        let expr = FuzzyExpression::Is {
            variable: "temperature".to_string(),
            set: "hot".to_string(),
        };
        assert!(
            (evaluate_expression(&expr, &sample_memberships())
                .unwrap()
                .value()
                - 0.8)
                .abs()
                < 1e-10
        );
    }

    #[test]
    fn and_expression_returns_minimum() {
        let expr = FuzzyExpression::And {
            terms: vec![
                FuzzyExpression::Is {
                    variable: "temperature".to_string(),
                    set: "hot".to_string(),
                },
                FuzzyExpression::Is {
                    variable: "temperature".to_string(),
                    set: "cold".to_string(),
                },
            ],
        };
        assert!(
            (evaluate_expression(&expr, &sample_memberships())
                .unwrap()
                .value()
                - 0.2)
                .abs()
                < 1e-10
        );
    }

    #[test]
    fn or_expression_returns_maximum() {
        let expr = FuzzyExpression::Or {
            terms: vec![
                FuzzyExpression::Is {
                    variable: "temperature".to_string(),
                    set: "hot".to_string(),
                },
                FuzzyExpression::Is {
                    variable: "temperature".to_string(),
                    set: "cold".to_string(),
                },
            ],
        };
        assert!(
            (evaluate_expression(&expr, &sample_memberships())
                .unwrap()
                .value()
                - 0.8)
                .abs()
                < 1e-10
        );
    }

    #[test]
    fn not_expression_returns_complement() {
        let expr = FuzzyExpression::Not {
            term: Box::new(FuzzyExpression::Is {
                variable: "temperature".to_string(),
                set: "hot".to_string(),
            }),
        };
        assert!(
            (evaluate_expression(&expr, &sample_memberships())
                .unwrap()
                .value()
                - 0.2)
                .abs()
                < 1e-10
        );
    }

    #[test]
    fn is_expression_missing_variable_errors() {
        let expr = FuzzyExpression::Is {
            variable: "humidity".to_string(),
            set: "high".to_string(),
        };
        assert!(evaluate_expression(&expr, &sample_memberships()).is_err());
    }

    // ── FuzzyInferenceOutput::summary ─────────────────────────────────────────

    #[test]
    fn summary_with_memberships() {
        let mut memberships = BTreeMap::new();
        memberships.insert("comfort.high".to_string(), MembershipDegree::new(0.75));
        let output = FuzzyInferenceOutput {
            input_memberships: BTreeMap::new(),
            memberships,
            activated_rules: vec![],
            confidence: MembershipDegree::new(0.75),
            total_rules: 2,
        };
        let s = output.summary();
        assert!(s.contains('2'));
        assert!(s.contains("comfort.high"));
    }

    #[test]
    fn summary_with_no_memberships() {
        let output = FuzzyInferenceOutput {
            input_memberships: BTreeMap::new(),
            memberships: BTreeMap::new(),
            activated_rules: vec![],
            confidence: MembershipDegree::zero(),
            total_rules: 1,
        };
        assert!(output.summary().contains("no memberships fired"));
    }

    // ── FuzzyConsequent::key ──────────────────────────────────────────────────

    #[test]
    fn consequent_key_format() {
        let c = FuzzyConsequent {
            variable: "foo".to_string(),
            set: "bar".to_string(),
        };
        assert_eq!(c.key(), "foo.bar");
    }
}
