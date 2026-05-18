use crate::primitives::ZScoreThreshold;
use converge_pack::gate::GateResult as Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionInput {
    pub values: Vec<f64>,
    #[serde(default = "default_threshold")]
    pub threshold: ZScoreThreshold,
    pub labels: Option<Vec<String>>,
}

fn default_threshold() -> ZScoreThreshold {
    ZScoreThreshold::new(2.0).expect("2.0 is a valid ZScoreThreshold")
}

impl AnomalyDetectionInput {
    pub fn validate(&self) -> Result<()> {
        if self.values.is_empty() {
            return Err(converge_pack::GateError::invalid_input(
                "At least one value required",
            ));
        }
        // threshold > 0 guaranteed by ZScoreThreshold construction.
        if let Some(labels) = &self.labels
            && labels.len() != self.values.len()
        {
            return Err(converge_pack::GateError::invalid_input(
                "Labels length must match values length",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyRecord {
    pub index: usize,
    pub value: f64,
    pub z_score: f64,
    pub label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionOutput {
    pub anomalies: Vec<AnomalyRecord>,
    pub mean: f64,
    pub std_dev: f64,
    pub total_points: usize,
    pub anomaly_count: usize,
}

impl AnomalyDetectionOutput {
    pub fn summary(&self) -> String {
        format!(
            "Detected {} anomalies in {} points (threshold z>{})",
            self.anomaly_count, self.total_points, self.std_dev
        )
    }
}
