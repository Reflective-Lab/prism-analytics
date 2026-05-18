use crate::primitives::UnitFraction;
use converge_pack::gate::GateResult as Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationInput {
    pub records: Vec<Vec<f64>>,
    pub weights: Vec<f64>,
    pub bias: f64,
    pub threshold: UnitFraction,
    pub labels: Option<(String, String)>,
}

impl ClassificationInput {
    pub fn validate(&self) -> Result<()> {
        if self.records.is_empty() {
            return Err(converge_pack::GateError::invalid_input(
                "At least one record required",
            ));
        }
        let dim = self.weights.len();
        if dim == 0 {
            return Err(converge_pack::GateError::invalid_input(
                "At least one weight (feature) required",
            ));
        }
        for (i, record) in self.records.iter().enumerate() {
            if record.len() != dim {
                return Err(converge_pack::GateError::invalid_input(format!(
                    "Record {} has {} features, expected {}",
                    i,
                    record.len(),
                    dim
                )));
            }
        }
        // threshold constraint guaranteed by UnitFraction construction.
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifiedRecord {
    pub index: usize,
    pub probability: f64,
    pub label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationOutput {
    pub predictions: Vec<ClassifiedRecord>,
    pub positive_count: usize,
    pub negative_count: usize,
    pub total: usize,
}

impl ClassificationOutput {
    pub fn summary(&self) -> String {
        format!(
            "Classified {} records: {} positive, {} negative",
            self.total, self.positive_count, self.negative_count,
        )
    }
}
