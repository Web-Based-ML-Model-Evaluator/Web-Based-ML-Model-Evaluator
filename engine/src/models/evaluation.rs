//! Types for evaluating a model the user uploaded (ONNX), run on-device by the browser.
//!
//! The browser runs the model; the engine supplies its inputs and scores its predictions
//! against the ground truth it reads from the dataset itself.

use crate::models::execution::ExecutionResponse;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Which dataset columns to feed the model, and which column holds the answers.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelInputsRequest {
    pub dataset_id: String,
    /// "classification" or "regression"
    pub task_type: String,
    pub target_column: String,
    pub feature_columns: Vec<String>,
}

/// Feature matrix for the model, in the order of `feature_columns`.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelInputs {
    pub rows: usize,
    pub feature_columns: Vec<String>,
    pub features: Vec<Vec<f64>>,
    /// Classification only: the target's distinct labels, sorted.
    pub classes: Vec<String>,
}

/// How to read numeric classification outputs.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PredictionKind {
    /// The model outputs the label itself (text, or a number such as 0/1).
    #[default]
    Label,
    /// The model outputs a position in the sorted class list (e.g. argmax of scores).
    ClassIndex,
}

/// Predictions produced by an uploaded model, one per dataset row.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ExternalPredictions {
    pub dataset_id: String,
    /// Shown in results and the leaderboard, e.g. "Uploaded ONNX: model.onnx".
    pub model_name: String,
    pub task_type: String,
    pub target_column: String,
    pub predictions: Vec<Value>,
    #[serde(default)]
    pub kind: PredictionKind,
    /// Time the model took to predict, measured by the browser.
    pub inference_ms: f64,
}

/// Metrics for an uploaded model, plus how its outputs were matched to labels.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ExternalEvaluation {
    #[serde(flatten)]
    pub execution: ExecutionResponse,
    /// Set when numeric outputs had to be matched to text labels.
    pub class_mapping: Option<String>,
}
