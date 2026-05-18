use crate::primitives::UnitFraction;
use converge_pack::gate::GateResult as Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastingInput {
    pub values: Vec<f64>,
    pub horizon: usize,
    #[serde(default = "default_alpha")]
    pub alpha: UnitFraction,
}

fn default_alpha() -> UnitFraction {
    UnitFraction::new(0.3).expect("0.3 is a valid UnitFraction")
}

impl ForecastingInput {
    pub fn validate(&self) -> Result<()> {
        if self.values.len() < 2 {
            return Err(converge_pack::GateError::invalid_input(
                "At least 2 historical values required",
            ));
        }
        if self.horizon == 0 {
            return Err(converge_pack::GateError::invalid_input(
                "Horizon must be >= 1",
            ));
        }
        // alpha constraint guaranteed by UnitFraction construction.
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastPoint {
    pub step: usize,
    pub value: f64,
    pub lower: f64,
    pub upper: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastingOutput {
    pub predictions: Vec<ForecastPoint>,
    pub residual_std: f64,
    pub horizon: usize,
}

impl ForecastingOutput {
    pub fn summary(&self) -> String {
        format!(
            "Forecasted {} steps ahead (residual std: {:.3})",
            self.horizon, self.residual_std,
        )
    }
}
