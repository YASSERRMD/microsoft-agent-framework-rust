use async_trait::async_trait;
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("invalid arguments: {0}")]
    InvalidArgs(String),
    #[error("execution failed: {0}")]
    Execution(String),
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    fn input_schema(&self) -> Value;
    fn output_schema(&self) -> Value;
    async fn execute(&self, args: Value) -> Result<Value, ToolError>;
}

#[derive(Default)]
pub struct ToolRegistry {
    tools: BTreeMap<String, Arc<dyn Tool>>, // deterministic ordering
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<T: Tool + 'static>(&mut self, tool: T) {
        self.tools.insert(tool.name().to_string(), Arc::new(tool));
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    pub fn list(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }
}

pub mod builtins {
    use super::{Tool, ToolError};
    use async_trait::async_trait;
    use serde_json::Value;

    pub struct TimeTool;

    #[async_trait]
    impl Tool for TimeTool {
        fn name(&self) -> &'static str {
            "time"
        }

        fn input_schema(&self) -> Value {
            serde_json::json!({"type": "object", "properties": {}})
        }

        fn output_schema(&self) -> Value {
            serde_json::json!({"type": "string", "description": "ISO-8601 timestamp"})
        }

        async fn execute(&self, _args: Value) -> Result<Value, ToolError> {
            let now = chrono::Utc::now().to_rfc3339();
            Ok(Value::String(now))
        }
    }

    pub struct MathTool;

    #[async_trait]
    impl Tool for MathTool {
        fn name(&self) -> &'static str {
            "math"
        }

        fn input_schema(&self) -> Value {
            serde_json::json!({
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"]
            })
        }

        fn output_schema(&self) -> Value {
            serde_json::json!({"type": "number"})
        }

        async fn execute(&self, args: Value) -> Result<Value, ToolError> {
            let expr = args
                .get("expression")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::InvalidArgs("expression missing".into()))?;
            let value: f64 = meval::eval_str(expr).map_err(|e| ToolError::Execution(e.to_string()))?;
            Ok(Value::from(value))
        }
    }

    pub struct LogTool;

    #[async_trait]
    impl Tool for LogTool {
        fn name(&self) -> &'static str {
            "log"
        }

        fn input_schema(&self) -> Value {
            serde_json::json!({"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]})
        }

        fn output_schema(&self) -> Value {
            serde_json::json!({"type": "object", "properties": {"ack": {"type": "boolean"}}})
        }

        async fn execute(&self, args: Value) -> Result<Value, ToolError> {
            if let Some(msg) = args.get("message").and_then(|v| v.as_str()) {
                tracing::info!(target: "agent-tools::log", message = msg);
                Ok(serde_json::json!({"ack": true}))
            } else {
                Err(ToolError::InvalidArgs("message missing".into()))
            }
        }
    }

    pub struct HttpFetchTool {
        client: reqwest::Client,
    }

    impl HttpFetchTool {
        pub fn new() -> Self {
            Self { client: reqwest::Client::new() }
        }
    }

    #[async_trait]
    impl Tool for HttpFetchTool {
        fn name(&self) -> &'static str {
            "http_fetch"
        }

        fn input_schema(&self) -> Value {
            serde_json::json!({
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"]
            })
        }

        fn output_schema(&self) -> Value {
            serde_json::json!({"type": "object", "properties": {"status": {"type": "number"}, "body": {"type": "string"}}})
        }

        async fn execute(&self, args: Value) -> Result<Value, ToolError> {
            let url = args
                .get("url")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::InvalidArgs("url missing".into()))?;
            let resp = self
                .client
                .get(url)
                .send()
                .await
                .map_err(|e| ToolError::Execution(e.to_string()))?;
            let status = resp.status().as_u16();
            let body = resp
                .text()
                .await
                .map_err(|e| ToolError::Execution(e.to_string()))?;
            Ok(serde_json::json!({"status": status, "body": body}))
        }
    }
}
