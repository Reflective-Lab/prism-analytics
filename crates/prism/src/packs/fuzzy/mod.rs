mod sugeno;

pub use crate::fuzzy::*;
pub use sugeno::*;

use converge_optimization::packs::{
    InvariantDef, InvariantResult, Pack, PackSolveResult, default_gate_evaluation,
};
use converge_pack::gate::GateResult as Result;
use converge_pack::gate::{KernelTraceLink, ProblemSpec, PromotionGate, ProposedPlan};

pub struct FuzzyInferencePack;

impl Pack for FuzzyInferencePack {
    fn name(&self) -> &'static str {
        "fuzzy-inference"
    }

    fn version(&self) -> &'static str {
        "1.0.0"
    }

    fn validate_inputs(&self, inputs: &serde_json::Value) -> Result<()> {
        let input: FuzzyInferenceInput = serde_json::from_value(inputs.clone())
            .map_err(|e| converge_pack::GateError::invalid_input(format!("Invalid input: {e}")))?;
        input.validate()
    }

    fn invariants(&self) -> &[InvariantDef] {
        static INVARIANTS: std::sync::LazyLock<Vec<InvariantDef>> =
            std::sync::LazyLock::new(|| {
                vec![
                    InvariantDef::critical(
                        "valid-memberships",
                        "All fuzzy memberships must be in [0, 1]",
                    ),
                    InvariantDef::advisory(
                        "rule-activation",
                        "No fuzzy rule fired - output carries no active expectation signal",
                    ),
                ]
            });
        &INVARIANTS
    }

    fn solve(&self, spec: &ProblemSpec) -> Result<PackSolveResult> {
        let input: FuzzyInferenceInput = spec.inputs_as()?;
        input.validate()?;

        let solver = FuzzyInferenceEngine;
        let (output, report) = solver.solve(&input, spec)?;

        let trace = KernelTraceLink::audit_only(format!("trace-{}", spec.problem_id));
        let confidence = output.confidence;
        let plan = ProposedPlan::from_payload(
            format!("plan-{}", spec.problem_id),
            self.name(),
            output.summary(),
            &output,
            confidence,
            trace,
        )?;

        Ok(PackSolveResult::new(plan, report))
    }

    fn check_invariants(&self, plan: &ProposedPlan) -> Result<Vec<InvariantResult>> {
        let output: FuzzyInferenceOutput = serde_json::from_value(plan.plan.clone())
            .map_err(|e| converge_pack::GateError::invalid_input(e.to_string()))?;

        let mut results = vec![];
        let outputs_valid = output
            .memberships
            .values()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value));
        let inputs_valid = output.input_memberships.values().all(|sets| {
            sets.values()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
        });

        if outputs_valid && inputs_valid {
            results.push(InvariantResult::pass("valid-memberships"));
        } else {
            results.push(InvariantResult::fail(
                "valid-memberships",
                converge_pack::gate::Violation::new(
                    "valid-memberships",
                    1.0,
                    "A fuzzy membership was outside [0, 1]",
                ),
            ));
        }

        if output.activated_rules.is_empty() {
            results.push(InvariantResult::fail(
                "rule-activation",
                converge_pack::gate::Violation::new("rule-activation", 0.0, "No fuzzy rules fired"),
            ));
        } else {
            results.push(InvariantResult::pass("rule-activation"));
        }

        Ok(results)
    }

    fn evaluate_gate(
        &self,
        _plan: &ProposedPlan,
        invariant_results: &[InvariantResult],
    ) -> PromotionGate {
        default_gate_evaluation(invariant_results, self.invariants())
    }
}
