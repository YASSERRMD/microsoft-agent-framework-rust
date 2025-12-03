use async_trait::async_trait;
use serde_json::{json, Value};
use thiserror::Error;

/// Standardized result shape shared by all evaluators.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationResult {
    /// Whether the evaluator considers the candidate output acceptable.
    pub passed: bool,
    /// Normalized score in the range `[0.0, 1.0]`.
    pub score: f32,
    /// Optional textual rationale.
    pub reason: Option<String>,
    /// Free-form metadata for downstream consumers.
    pub details: Value,
}

impl EvaluationResult {
    pub fn pass(score: f32, reason: impl IntoReason) -> Self {
        Self {
            passed: true,
            score: score.clamp(0.0, 1.0),
            reason: reason.into_reason(),
            details: Value::Null,
        }
    }

    pub fn fail(reason: impl IntoReason) -> Self {
        Self {
            passed: false,
            score: 0.0,
            reason: reason.into_reason(),
            details: Value::Null,
        }
    }

    pub fn with_details(mut self, details: Value) -> Self {
        self.details = details;
        self
    }
}

pub trait IntoReason {
    fn into_reason(self) -> Option<String>;
}

impl IntoReason for Option<String> {
    fn into_reason(self) -> Option<String> {
        self
    }
}

impl IntoReason for &str {
    fn into_reason(self) -> Option<String> {
        Some(self.to_string())
    }
}

impl IntoReason for String {
    fn into_reason(self) -> Option<String> {
        Some(self)
    }
}

/// Ranked ordering for a list of candidate plans.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanRanking {
    /// Ordered plan indices, highest-ranked first.
    pub order: Vec<usize>,
    /// Optional rationale per plan index.
    pub rationales: Vec<Option<String>>,
}

impl PlanRanking {
    pub fn new(order: Vec<usize>) -> Self {
        let rationales = order.iter().map(|_| None).collect();
        Self { order, rationales }
    }

    pub fn with_rationales(mut self, rationales: Vec<Option<String>>) -> Self {
        self.rationales = rationales;
        self
    }
}

#[derive(Debug, Error)]
pub enum EvalError {
    #[error("evaluation failed: {0}")]
    Failed(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

#[async_trait]
pub trait StepEvaluator: Send + Sync {
    /// Evaluate a single step output (self-evaluation use case).
    async fn evaluate(&self, step_output: &Value) -> Result<EvaluationResult, EvalError>;
}

#[async_trait]
pub trait OutputEvaluator: Send + Sync {
    /// Evaluate the final output of an agent run.
    async fn evaluate(&self, final_output: &Value) -> Result<EvaluationResult, EvalError>;
}

#[async_trait]
pub trait PlanEvaluator: Send + Sync {
    /// Rank candidate plans for execution.
    async fn rank(&self, plans: &[Value]) -> Result<PlanRanking, EvalError>;
}

#[async_trait]
pub trait RewardEvaluator: Send + Sync {
    /// Assign a reward signal for reinforcement-style feedback.
    async fn reward(&self, context: &Value) -> Result<EvaluationResult, EvalError>;
}

#[async_trait]
pub trait GuardrailEvaluator: Send + Sync {
    /// Validate a candidate output against safety guardrails.
    async fn validate(&self, candidate: &Value) -> Result<EvaluationResult, EvalError>;
}

/// Ensures step outputs remain structured as JSON objects or arrays.
pub struct JsonValidityEvaluator;

#[async_trait]
impl StepEvaluator for JsonValidityEvaluator {
    async fn evaluate(&self, step_output: &Value) -> Result<EvaluationResult, EvalError> {
        let passed = step_output.is_object() || step_output.is_array();
        let value_type = match step_output {
            Value::Null => "null",
            Value::Bool(_) => "boolean",
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
        };

        let result = if passed {
            EvaluationResult::pass(1.0, "output is valid JSON structure")
        } else {
            EvaluationResult::fail("output must be an object or array")
        }
        .with_details(json!({ "type": value_type }));

        Ok(result)
    }
}

/// Simple heuristics to flag obviously toxic content.
pub struct ToxicityEvaluator {
    disallowed_terms: Vec<&'static str>,
}

impl Default for ToxicityEvaluator {
    fn default() -> Self {
        Self {
            disallowed_terms: vec!["hate", "violence", "kill", "racist", "terror"],
        }
    }
}

#[async_trait]
impl GuardrailEvaluator for ToxicityEvaluator {
    async fn validate(&self, candidate: &Value) -> Result<EvaluationResult, EvalError> {
        let text = candidate
            .as_str()
            .ok_or_else(|| EvalError::InvalidInput("candidate must be a string".into()))?;

        let lowered = text.to_lowercase();
        let offending: Vec<&str> = self
            .disallowed_terms
            .iter()
            .copied()
            .filter(|term| lowered.contains(term))
            .collect();

        if offending.is_empty() {
            Ok(EvaluationResult::pass(1.0, "no toxic terms detected"))
        } else {
            Ok(EvaluationResult::fail("toxic language detected")
                .with_details(json!({"offending_terms": offending})))
        }
    }
}

/// Flags outputs that appear ungrounded or speculative.
pub struct HallucinationEvaluator;

#[async_trait]
impl GuardrailEvaluator for HallucinationEvaluator {
    async fn validate(&self, candidate: &Value) -> Result<EvaluationResult, EvalError> {
        let text = candidate
            .as_str()
            .ok_or_else(|| EvalError::InvalidInput("candidate must be a string".into()))?;

        let lowered = text.to_lowercase();
        let signals = ["made up", "fictional", "not sure", "guessing", "probably"].to_vec();
        let hallucinated: Vec<&str> = signals
            .into_iter()
            .filter(|signal| lowered.contains(signal))
            .collect();

        if hallucinated.is_empty() {
            Ok(EvaluationResult::pass(
                1.0,
                "no hallucination markers detected",
            ))
        } else {
            Ok(
                EvaluationResult::fail("possible hallucination markers present")
                    .with_details(json!({"markers": hallucinated})),
            )
        }
    }
}

/// Validates tool-call payloads for structure and argument shapes.
pub struct ToolCallCorrectnessEvaluator;

#[async_trait]
impl StepEvaluator for ToolCallCorrectnessEvaluator {
    async fn evaluate(&self, step_output: &Value) -> Result<EvaluationResult, EvalError> {
        let Some(obj) = step_output.as_object() else {
            return Ok(EvaluationResult::fail("expected tool call object"));
        };

        let tool = obj.get("tool");
        let args = obj.get("arguments");

        if tool.is_none() || args.is_none() {
            return Ok(EvaluationResult::fail(
                "tool call must contain 'tool' and 'arguments' fields",
            ));
        }

        if !args.unwrap().is_object() {
            return Ok(EvaluationResult::fail("'arguments' must be a JSON object"));
        }

        Ok(EvaluationResult::pass(1.0, "tool call structure is valid"))
    }
}

/// Ensures hidden chain-of-thought is not leaked into the final answer.
pub struct ChainOfThoughtGuardrail;

#[async_trait]
impl GuardrailEvaluator for ChainOfThoughtGuardrail {
    async fn validate(&self, candidate: &Value) -> Result<EvaluationResult, EvalError> {
        let text = candidate
            .as_str()
            .ok_or_else(|| EvalError::InvalidInput("candidate must be a string".into()))?;

        let lowered = text.to_lowercase();
        if lowered.contains("chain-of-thought") || lowered.contains("reasoning:") {
            Ok(EvaluationResult::fail(
                "chain-of-thought markers should be hidden from the user",
            ))
        } else {
            Ok(EvaluationResult::pass(
                1.0,
                "no chain-of-thought markers exposed to the user",
            ))
        }
    }
}

/// Allows a model or agent to provide a self-scored reflection for the step.
pub struct SelfAssessmentEvaluator;

#[async_trait]
impl StepEvaluator for SelfAssessmentEvaluator {
    async fn evaluate(&self, step_output: &Value) -> Result<EvaluationResult, EvalError> {
        let Some(assessment) = step_output.get("self_assessment") else {
            return Ok(EvaluationResult::fail(
                "self_assessment field missing from step output",
            ));
        };

        let score = assessment
            .get("score")
            .and_then(Value::as_f64)
            .map(|v| v as f32)
            .unwrap_or(0.0);
        let notes = assessment
            .get("notes")
            .and_then(Value::as_str)
            .map(|s| s.to_string());

        Ok(
            EvaluationResult::pass(score, "step provided self-assessment")
                .with_details(json!({ "notes": notes })),
        )
    }
}

/// Ranks plans deterministically in their original order.
pub struct PassThroughPlanEvaluator;

#[async_trait]
impl PlanEvaluator for PassThroughPlanEvaluator {
    async fn rank(&self, plans: &[Value]) -> Result<PlanRanking, EvalError> {
        Ok(PlanRanking::new((0..plans.len()).collect()))
    }
}

/// Basic reward evaluator that converts a score field into a normalized reward.
pub struct ScoreRewardEvaluator;

#[async_trait]
impl RewardEvaluator for ScoreRewardEvaluator {
    async fn reward(&self, context: &Value) -> Result<EvaluationResult, EvalError> {
        let score = context
            .get("score")
            .and_then(Value::as_f64)
            .map(|v| v as f32)
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);

        let reason = context
            .get("reason")
            .and_then(Value::as_str)
            .map(|s| s.to_string());

        Ok(EvaluationResult::pass(score, reason))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn json_validity_passes_for_objects() {
        let evaluator = JsonValidityEvaluator;
        let result = evaluator
            .evaluate(&json!({ "message": "ok" }))
            .await
            .unwrap();

        assert!(result.passed);
        assert!(result.score > 0.9);
    }

    #[tokio::test]
    async fn toxicity_blocks_offending_terms() {
        let evaluator = ToxicityEvaluator::default();
        let result = evaluator
            .validate(&Value::String("This message encourages violence".into()))
            .await
            .unwrap();

        assert!(!result.passed);
        assert!(result.details["offending_terms"].is_array());
    }

    #[tokio::test]
    async fn tool_call_validator_requires_structure() {
        let evaluator = ToolCallCorrectnessEvaluator;
        let result = evaluator
            .evaluate(&json!({ "tool": "search", "arguments": {"query": "rust"} }))
            .await
            .unwrap();

        assert!(result.passed);
    }

    #[tokio::test]
    async fn chain_of_thought_guardrail_detects_markers() {
        let evaluator = ChainOfThoughtGuardrail;
        let result = evaluator
            .validate(&Value::String(
                "Chain-of-thought: I reasoned about X".into(),
            ))
            .await
            .unwrap();

        assert!(!result.passed);
    }

    #[tokio::test]
    async fn reward_evaluator_uses_score() {
        let evaluator = ScoreRewardEvaluator;
        let result = evaluator
            .reward(&json!({"score": 0.8, "reason": "high-quality"}))
            .await
            .unwrap();

        assert!(result.passed);
        assert_eq!(result.score, 0.8);
    }
}
