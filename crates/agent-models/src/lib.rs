use async_trait::async_trait;
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

pub type Token = String;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UsageMetrics {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelMetadata {
    pub provider: String,
    pub model: String,
    pub supports_tools: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToolCallInfo {
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CompletionChunk {
    pub token: Token,
    pub index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LLMResponse {
    pub content: String,
    pub usage: UsageMetrics,
    pub tool_calls: Vec<ToolCallInfo>,
    pub metadata: ModelMetadata,
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("request failed: {0}")]
    Request(String),
}

#[async_trait]
pub trait LLMModel: Send + Sync {
    async fn generate(&self, prompt: &str) -> Result<LLMResponse, ModelError>;
    async fn stream(&self, prompt: &str) -> Result<Box<dyn tokio_stream::Stream<Item = Token> + Send + Unpin>, ModelError>;
    fn supports_tools(&self) -> bool;
}

pub struct StubModel;

#[async_trait]
impl LLMModel for StubModel {
    async fn generate(&self, prompt: &str) -> Result<LLMResponse, ModelError> {
        Ok(LLMResponse {
            content: format!("echo: {prompt}"),
            usage: UsageMetrics { prompt_tokens: prompt.len(), completion_tokens: 2 },
            tool_calls: Vec::new(),
            metadata: ModelMetadata { provider: "stub".into(), model: "stub".into(), supports_tools: false },
        })
    }

    async fn stream(&self, prompt: &str) -> Result<Box<dyn tokio_stream::Stream<Item = Token> + Send + Unpin>, ModelError> {
        let mut tokens = vec!["echo".to_string(), prompt.to_string()];
        Ok(Box::new(tokio_stream::iter(tokens.drain(..))))
    }

    fn supports_tools(&self) -> bool {
        false
    }
}

pub struct RandomReasoner;

#[async_trait]
impl LLMModel for RandomReasoner {
    async fn generate(&self, prompt: &str) -> Result<LLMResponse, ModelError> {
        let mut rng = rand::thread_rng();
        let calls = if rng.gen_bool(0.2) {
            vec![ToolCallInfo { name: "math".into(), arguments: serde_json::json!({"expression": "1+1"}) }]
        } else {
            Vec::new()
        };
        Ok(LLMResponse {
            content: format!("reasoned: {prompt}"),
            usage: UsageMetrics { prompt_tokens: prompt.len(), completion_tokens: 3 },
            tool_calls: calls,
            metadata: ModelMetadata { provider: "random".into(), model: "reasoner".into(), supports_tools: true },
        })
    }

    async fn stream(&self, prompt: &str) -> Result<Box<dyn tokio_stream::Stream<Item = Token> + Send + Unpin>, ModelError> {
        let chunks = vec!["reasoned".to_string(), prompt.to_string()];
        Ok(Box::new(tokio_stream::iter(chunks)))
    }

    fn supports_tools(&self) -> bool {
        true
    }
}
