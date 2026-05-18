use std::collections::BTreeMap;

use converge_pack::gate::GateResult as Result;
use converge_pack::gate::{ProblemSpec, ReplayEnvelope, SolverReport};

use super::types::*;

pub struct FuzzyInferenceEngine;

impl FuzzyInferenceEngine {
    pub fn solve(
        &self,
        input: &FuzzyInferenceInput,
        spec: &ProblemSpec,
    ) -> Result<(FuzzyInferenceOutput, SolverReport)> {
        input.validate()?;

        let input_memberships = evaluate_input_memberships(&input.inputs, &input.variables);
        let mut memberships: BTreeMap<String, MembershipDegree> = BTreeMap::new();
        let mut activated_rules = Vec::new();

        for (idx, rule) in input.rules.iter().enumerate() {
            let antecedent_strength = evaluate_expression(&rule.when, &input_memberships)?;
            let weight = rule.weight();
            let strength = MembershipDegree::new(antecedent_strength.value() * weight.value());
            let consequent = rule.then.key();

            memberships
                .entry(consequent.clone())
                .and_modify(|current| {
                    if strength > *current {
                        *current = strength;
                    }
                })
                .or_insert(strength);

            if strength > MembershipDegree::zero() {
                activated_rules.push(ActivatedRule {
                    id: rule
                        .id
                        .clone()
                        .unwrap_or_else(|| format!("rule-{}", idx + 1)),
                    antecedent_strength,
                    weight,
                    strength,
                    consequent,
                });
            }
        }

        let confidence = memberships
            .values()
            .copied()
            .fold(MembershipDegree::zero(), |a, b| if b > a { b } else { a });
        let output = FuzzyInferenceOutput {
            input_memberships,
            memberships,
            activated_rules,
            confidence,
            total_rules: input.rules.len(),
        };

        let replay = ReplayEnvelope::minimal(spec.seed());
        let report = SolverReport::optimal("fuzzy-inference-v1", confidence.value(), replay);

        Ok((output, report))
    }
}
