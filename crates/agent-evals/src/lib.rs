use async_trait::async_trait;
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EvalError {
    #[error("evaluation failed: {0}")]
    Failed(String),
}

#[async_trait]
pub trait StepEvaluator: Send + Sync {
    async fn evaluate(&self, step_output: &Value) -> Result<f32, EvalError>;
}

#[async_trait]
pub trait OutputEvaluator: Send + Sync {
    async fn evaluate(&self, final_output: &Value) -> Result<f32, EvalError>;
}

#[async_trait]
pub trait PlanEvaluator: Send + Sync {
    async fn rank(&self, plans: &[Value]) -> Result<Vec<usize>, EvalError>;
}

pub struct JsonValidityEvaluator;

#[async_trait]
impl StepEvaluator for JsonValidityEvaluator {
    async fn evaluate(&self, step_output: &Value) -> Result<f32, EvalError> {
        if step_output.is_object() || step_output.is_array() {
            Ok(1.0)
        } else {
            Ok(0.0)
        }
    }
}

pub struct PassThroughPlanEvaluator;

#[async_trait]
impl PlanEvaluator for PassThroughPlanEvaluator {
    async fn rank(&self, plans: &[Value]) -> Result<Vec<usize>, EvalError> {
        Ok((0..plans.len()).collect())
    }
}
