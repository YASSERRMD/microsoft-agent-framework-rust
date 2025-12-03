use std::pin::Pin;

use async_trait::async_trait;
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_stream::{self as stream, Stream};

pub type Token = String;
pub type TokenStream = Pin<Box<dyn Stream<Item = Token> + Send>>;

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
    pub is_reasoning: bool,
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

#[async_trait]
pub trait LLMModel: Send + Sync {
    async fn generate(&self, prompt: &str) -> LLMResponse;
    async fn stream(&self, prompt: &str) -> TokenStream;
    fn supports_tools(&self) -> bool;
}

fn build_usage(prompt: &str, completion: &str) -> UsageMetrics {
    UsageMetrics {
        prompt_tokens: prompt.split_whitespace().count(),
        completion_tokens: completion.split_whitespace().count(),
    }
}

fn token_stream_from_content(content: &str) -> TokenStream {
    let tokens: Vec<String> = content.split_whitespace().map(ToOwned::to_owned).collect();
    Box::pin(stream::iter(tokens))
}

pub struct OpenAIChatModel {
    pub model: String,
    pub supports_tools: bool,
    pub reasoning: bool,
}

impl OpenAIChatModel {
    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            provider: "openai".into(),
            model: self.model.clone(),
            supports_tools: self.supports_tools,
            is_reasoning: self.reasoning,
        }
    }
}

#[async_trait]
impl LLMModel for OpenAIChatModel {
    async fn generate(&self, prompt: &str) -> LLMResponse {
        let content = if self.reasoning {
            format!("[reasoning:{}] {}", self.model, prompt)
        } else {
            format!("[chat:{}] {}", self.model, prompt)
        };

        let tool_calls = if self.supports_tools {
            vec![ToolCallInfo {
                name: "auto_tool".into(),
                arguments: serde_json::json!({"prompt": prompt}),
            }]
        } else {
            Vec::new()
        };

        LLMResponse {
            usage: build_usage(prompt, &content),
            content,
            tool_calls,
            metadata: self.metadata(),
        }
    }

    async fn stream(&self, prompt: &str) -> TokenStream {
        let content = if self.reasoning {
            format!("reasoning {}", prompt)
        } else {
            format!("chat {}", prompt)
        };

        token_stream_from_content(&content)
    }

    fn supports_tools(&self) -> bool {
        self.supports_tools
    }
}

pub struct AzureOpenAIModel {
    pub deployment: String,
    pub supports_tools: bool,
    pub reasoning: bool,
}

impl AzureOpenAIModel {
    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            provider: "azure_openai".into(),
            model: self.deployment.clone(),
            supports_tools: self.supports_tools,
            is_reasoning: self.reasoning,
        }
    }
}

#[async_trait]
impl LLMModel for AzureOpenAIModel {
    async fn generate(&self, prompt: &str) -> LLMResponse {
        let content = if self.reasoning {
            format!("[azure-reasoning:{}] {}", self.deployment, prompt)
        } else {
            format!("[azure-chat:{}] {}", self.deployment, prompt)
        };

        let tool_calls = if self.supports_tools {
            vec![ToolCallInfo {
                name: "azure_tool".into(),
                arguments: serde_json::json!({"input": prompt}),
            }]
        } else {
            Vec::new()
        };

        LLMResponse {
            usage: build_usage(prompt, &content),
            content,
            tool_calls,
            metadata: self.metadata(),
        }
    }

    async fn stream(&self, prompt: &str) -> TokenStream {
        let content = format!("azure {}", prompt);
        token_stream_from_content(&content)
    }

    fn supports_tools(&self) -> bool {
        self.supports_tools
    }
}

pub struct OllamaModel {
    pub model: String,
    pub supports_tools: bool,
}

impl OllamaModel {
    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            provider: "ollama".into(),
            model: self.model.clone(),
            supports_tools: self.supports_tools,
            is_reasoning: false,
        }
    }
}

#[async_trait]
impl LLMModel for OllamaModel {
    async fn generate(&self, prompt: &str) -> LLMResponse {
        let content = format!("[ollama:{}] {}", self.model, prompt);
        LLMResponse {
            usage: build_usage(prompt, &content),
            content,
            tool_calls: Vec::new(),
            metadata: self.metadata(),
        }
    }

    async fn stream(&self, prompt: &str) -> TokenStream {
        token_stream_from_content(prompt)
    }

    fn supports_tools(&self) -> bool {
        self.supports_tools
    }
}

pub struct RestModel {
    pub endpoint: String,
    pub model: String,
    pub supports_tools: bool,
}

impl RestModel {
    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            provider: "rest".into(),
            model: self.model.clone(),
            supports_tools: self.supports_tools,
            is_reasoning: false,
        }
    }
}

#[async_trait]
impl LLMModel for RestModel {
    async fn generate(&self, prompt: &str) -> LLMResponse {
        let content = format!("[rest:{}] {} => {}", self.model, self.endpoint, prompt);
        LLMResponse {
            usage: build_usage(prompt, &content),
            content,
            tool_calls: Vec::new(),
            metadata: self.metadata(),
        }
    }

    async fn stream(&self, prompt: &str) -> TokenStream {
        token_stream_from_content(&format!("{} {}", self.model, prompt))
    }

    fn supports_tools(&self) -> bool {
        self.supports_tools
    }
}

pub struct EmbeddingModel {
    pub model: String,
}

impl EmbeddingModel {
    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            provider: "embedding".into(),
            model: self.model.clone(),
            supports_tools: false,
            is_reasoning: false,
        }
    }
}

#[async_trait]
impl LLMModel for EmbeddingModel {
    async fn generate(&self, prompt: &str) -> LLMResponse {
        let embedding = format!("embedding:{}", self.model);
        LLMResponse {
            usage: build_usage(prompt, &embedding),
            content: embedding,
            tool_calls: Vec::new(),
            metadata: self.metadata(),
        }
    }

    async fn stream(&self, prompt: &str) -> TokenStream {
        token_stream_from_content(&format!("embedding {}", prompt))
    }

    fn supports_tools(&self) -> bool {
        false
    }
}

pub struct StubModel;

#[async_trait]
impl LLMModel for StubModel {
    async fn generate(&self, prompt: &str) -> LLMResponse {
        let content = format!("echo: {prompt}");
        LLMResponse {
            content: content.clone(),
            usage: UsageMetrics {
                prompt_tokens: prompt.len(),
                completion_tokens: 2,
            },
            tool_calls: Vec::new(),
            metadata: ModelMetadata {
                provider: "stub".into(),
                model: "stub".into(),
                supports_tools: false,
                is_reasoning: false,
            },
        }
    }

    async fn stream(&self, prompt: &str) -> TokenStream {
        let tokens = vec!["echo".to_string(), prompt.to_string()];
        Box::pin(stream::iter(tokens))
    }

    fn supports_tools(&self) -> bool {
        false
    }
}

pub struct RandomReasoner;

#[async_trait]
impl LLMModel for RandomReasoner {
    async fn generate(&self, prompt: &str) -> LLMResponse {
        let mut rng = rand::thread_rng();
        let calls = if rng.gen_bool(0.2) {
            vec![ToolCallInfo {
                name: "math".into(),
                arguments: serde_json::json!({"expression": "1+1"}),
            }]
        } else {
            Vec::new()
        };
        let content = format!("reasoned: {prompt}");
        LLMResponse {
            content: content.clone(),
            usage: UsageMetrics {
                prompt_tokens: prompt.len(),
                completion_tokens: 3,
            },
            tool_calls: calls,
            metadata: ModelMetadata {
                provider: "random".into(),
                model: "reasoner".into(),
                supports_tools: true,
                is_reasoning: true,
            },
        }
    }

    async fn stream(&self, prompt: &str) -> TokenStream {
        let chunks = vec!["reasoned".to_string(), prompt.to_string()];
        Box::pin(stream::iter(chunks))
    }

    fn supports_tools(&self) -> bool {
        true
    }
}
