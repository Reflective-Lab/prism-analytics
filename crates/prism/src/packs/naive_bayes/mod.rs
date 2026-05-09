mod solver;
mod types;

pub use solver::*;
pub use types::*;

use converge_optimization::packs::{
    InvariantDef, InvariantResult, Pack, PackSolveResult, default_gate_evaluation,
};
use converge_pack::gate::GateResult as Result;
use converge_pack::gate::{KernelTraceLink, ProblemSpec, PromotionGate, ProposedPlan};

pub struct NaiveBayesPack;

impl Pack for NaiveBayesPack {
    fn name(&self) -> &'static str {
        "naive-bayes"
    }

    fn version(&self) -> &'static str {
        "1.0.0"
    }

    fn validate_inputs(&self, inputs: &serde_json::Value) -> Result<()> {
        let input: NaiveBayesInput = serde_json::from_value(inputs.clone())
            .map_err(|e| converge_pack::GateError::invalid_input(format!("Invalid input: {e}")))?;
        input.validate()
    }

    fn invariants(&self) -> &[InvariantDef] {
        static INVARIANTS: std::sync::LazyLock<Vec<InvariantDef>> =
            std::sync::LazyLock::new(|| {
                vec![
                    InvariantDef::critical(
                        "valid-probabilities",
                        "All class probabilities must be in [0, 1] and sum to 1",
                    ),
                    InvariantDef::advisory(
                        "dominant-class",
                        "Top class has > 99% probability — consider whether priors are balanced",
                    ),
                ]
            });
        &INVARIANTS
    }

    fn solve(&self, spec: &ProblemSpec) -> Result<PackSolveResult> {
        let input: NaiveBayesInput = spec.inputs_as()?;
        input.validate()?;

        let solver = GaussianNaiveBayes;
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
        let output: NaiveBayesOutput = serde_json::from_value(plan.plan.clone())
            .map_err(|e| converge_pack::GateError::invalid_input(e.to_string()))?;

        let mut results = vec![];

        let all_valid = output
            .probabilities
            .iter()
            .all(|p| (0.0..=1.0).contains(&p.probability));
        let sum: f64 = output.probabilities.iter().map(|p| p.probability).sum();

        if all_valid && (sum - 1.0).abs() < 1e-6 {
            results.push(InvariantResult::pass("valid-probabilities"));
        } else {
            results.push(InvariantResult::fail(
                "valid-probabilities",
                converge_pack::gate::Violation::new(
                    "valid-probabilities",
                    1.0,
                    format!("Probabilities invalid or sum to {sum:.6}"),
                ),
            ));
        }

        if output.confidence > 0.99 {
            results.push(InvariantResult::fail(
                "dominant-class",
                converge_pack::gate::Violation::new(
                    "dominant-class",
                    output.confidence,
                    format!("Top class probability {:.1}%", output.confidence * 100.0),
                ),
            ));
        } else {
            results.push(InvariantResult::pass("dominant-class"));
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
