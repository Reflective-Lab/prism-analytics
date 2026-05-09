use converge_pack::gate::GateResult as Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianParams {
    pub mean: f64,
    pub std_dev: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassDef {
    pub name: String,
    pub prior: f64,
    pub feature_params: Vec<GaussianParams>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NaiveBayesInput {
    pub classes: Vec<ClassDef>,
    pub features: Vec<f64>,
}

impl NaiveBayesInput {
    pub fn validate(&self) -> Result<()> {
        if self.classes.len() < 2 {
            return Err(converge_pack::GateError::invalid_input(
                "At least two classes required",
            ));
        }
        if self.features.is_empty() {
            return Err(converge_pack::GateError::invalid_input(
                "At least one feature required",
            ));
        }
        let n_features = self.features.len();
        for class in &self.classes {
            if class.prior <= 0.0 {
                return Err(converge_pack::GateError::invalid_input(format!(
                    "Class '{}' prior must be positive",
                    class.name
                )));
            }
            if class.feature_params.len() != n_features {
                return Err(converge_pack::GateError::invalid_input(format!(
                    "Class '{}' has {} feature params, expected {}",
                    class.name,
                    class.feature_params.len(),
                    n_features
                )));
            }
            for (i, params) in class.feature_params.iter().enumerate() {
                if params.std_dev <= 0.0 {
                    return Err(converge_pack::GateError::invalid_input(format!(
                        "Class '{}' feature {} std_dev must be positive",
                        class.name, i
                    )));
                }
                if !params.mean.is_finite() || !params.std_dev.is_finite() {
                    return Err(converge_pack::GateError::invalid_input(format!(
                        "Class '{}' feature {} has non-finite params",
                        class.name, i
                    )));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassProbability {
    pub class: String,
    pub probability: f64,
    pub log_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NaiveBayesOutput {
    pub predicted: String,
    pub confidence: f64,
    pub probabilities: Vec<ClassProbability>,
}

impl NaiveBayesOutput {
    pub fn summary(&self) -> String {
        format!(
            "Predicted class '{}' with {:.1}% confidence",
            self.predicted,
            self.confidence * 100.0
        )
    }
}
