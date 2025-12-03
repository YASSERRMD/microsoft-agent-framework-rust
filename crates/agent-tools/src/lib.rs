use async_trait::async_trait;
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
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

#[derive(Debug, Clone, Default)]
pub struct ToolMetadata {
    pub description: Option<String>,
    pub tags: Vec<String>,
    pub allowed_roles: Vec<String>,
    pub cooldown: Option<Duration>,
    pub access_controller: Option<AccessController>,
    pub rate_limit: Option<RateLimitPolicy>,
}

#[derive(Debug, Clone, Default)]
pub struct AccessController {
    pub required_roles: Vec<String>,
    pub policy_name: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RateLimitPolicy {
    pub max_calls: u64,
    pub per: Duration,
}

struct ToolEntry {
    tool: Arc<dyn Tool>,
    metadata: ToolMetadata,
}

#[derive(Default)]
pub struct ToolRegistry {
    tools: BTreeMap<String, ToolEntry>, // deterministic ordering
    last_invoked: Mutex<BTreeMap<String, Instant>>, // cooldown tracking
    rate_windows: Mutex<BTreeMap<String, RateWindow>>, // rate limiter
}

#[derive(Debug, Clone)]
struct RateWindow {
    started_at: Instant,
    calls: u64,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<T: Tool + 'static>(&mut self, tool: T) {
        self.register_with_metadata(tool, ToolMetadata::default());
    }

    pub fn register_with_metadata<T: Tool + 'static>(&mut self, tool: T, metadata: ToolMetadata) {
        self.tools.insert(
            tool.name().to_string(),
            ToolEntry {
                tool: Arc::new(tool),
                metadata,
            },
        );
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).map(|entry| entry.tool.clone())
    }

    pub fn get_metadata(&self, name: &str) -> Option<ToolMetadata> {
        self.tools.get(name).map(|entry| entry.metadata.clone())
    }

    pub fn list(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    pub fn list_with_metadata(&self) -> Vec<(String, ToolMetadata)> {
        self.tools
            .iter()
            .map(|(name, entry)| (name.clone(), entry.metadata.clone()))
            .collect()
    }

    pub async fn invoke(
        &self,
        name: &str,
        args: Value,
        caller_roles: &[String],
    ) -> Result<Value, ToolInvocationError> {
        let entry = self
            .tools
            .get(name)
            .ok_or_else(|| ToolInvocationError::NotFound(name.to_string()))?;

        self.enforce_access(name, &entry.metadata, caller_roles)?;
        self.enforce_cooldown(name, &entry.metadata)?;
        self.enforce_rate_limit(name, &entry.metadata)?;

        Ok(entry.tool.execute(args).await?)
    }

    fn enforce_access(
        &self,
        name: &str,
        metadata: &ToolMetadata,
        caller_roles: &[String],
    ) -> Result<(), ToolInvocationError> {
        let role_permitted = metadata.allowed_roles.is_empty()
            || caller_roles
                .iter()
                .any(|role| metadata.allowed_roles.iter().any(|r| r == role));

        let access_controller_permitted = metadata
            .access_controller
            .as_ref()
            .map(|controller| {
                controller.required_roles.is_empty()
                    || controller
                        .required_roles
                        .iter()
                        .any(|role| caller_roles.iter().any(|caller| caller == role))
            })
            .unwrap_or(true);

        if role_permitted && access_controller_permitted {
            Ok(())
        } else {
            let reason = metadata
                .access_controller
                .as_ref()
                .and_then(|c| c.policy_name.clone())
                .unwrap_or_else(|| "caller lacks required role".into());
            Err(ToolInvocationError::AccessDenied {
                tool: name.to_string(),
                reason,
            })
        }
    }

    fn enforce_rate_limit(
        &self,
        name: &str,
        metadata: &ToolMetadata,
    ) -> Result<(), ToolInvocationError> {
        let Some(policy) = &metadata.rate_limit else {
            return Ok(());
        };

        let mut guard = self
            .rate_windows
            .lock()
            .expect("rate limiter mutex poisoned");
        let window = guard.entry(name.to_string()).or_insert_with(|| RateWindow {
            started_at: Instant::now(),
            calls: 0,
        });

        if window.started_at.elapsed() > policy.per {
            window.started_at = Instant::now();
            window.calls = 0;
        }

        if window.calls >= policy.max_calls {
            return Err(ToolInvocationError::RateLimited {
                tool: name.to_string(),
                retry_after_ms: policy
                    .per
                    .saturating_sub(window.started_at.elapsed())
                    .as_millis() as u64,
            });
        }

        window.calls += 1;
        Ok(())
    }

    fn enforce_cooldown(
        &self,
        name: &str,
        metadata: &ToolMetadata,
    ) -> Result<(), ToolInvocationError> {
        if let Some(cooldown) = metadata.cooldown {
            let mut guard = self.last_invoked.lock().expect("cooldown mutex poisoned");
            if let Some(last) = guard.get(name) {
                let elapsed = last.elapsed();
                if elapsed < cooldown {
                    return Err(ToolInvocationError::CoolingDown {
                        tool: name.to_string(),
                        remaining_ms: cooldown.saturating_sub(elapsed).as_millis() as u64,
                    });
                }
            }
            guard.insert(name.to_string(), Instant::now());
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum ToolInvocationError {
    #[error("tool {0} not found")]
    NotFound(String),
    #[error("tool {tool} access denied: {reason}")]
    AccessDenied { tool: String, reason: String },
    #[error("tool {tool} cooling down, try again in {remaining_ms}ms")]
    CoolingDown { tool: String, remaining_ms: u64 },
    #[error("tool {tool} rate limited, retry after {retry_after_ms}ms")]
    RateLimited { tool: String, retry_after_ms: u64 },
    #[error(transparent)]
    Tool(#[from] ToolError),
}

pub mod builtins {
    use super::{Tool, ToolError};
    use async_trait::async_trait;
    use serde::{Deserialize, Serialize};
    use serde_json::Value;
    use tokio::fs;

    use std::fs as stdfs;
    use std::path::PathBuf;

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

    pub struct FileTool {
        root: PathBuf,
    }

    impl FileTool {
        pub fn new(root: impl AsRef<std::path::Path>) -> Self {
            Self {
                root: root.as_ref().to_path_buf(),
            }
        }

        fn canonical_root(&self) -> Result<PathBuf, ToolError> {
            stdfs::create_dir_all(&self.root)
                .map_err(|e| ToolError::Execution(format!("failed to create root: {e}")))?;
            self.root
                .canonicalize()
                .map_err(|e| ToolError::Execution(format!("failed to canonicalize root: {e}")))
        }

        fn resolve(&self, path: &str, allow_create: bool) -> Result<PathBuf, ToolError> {
            let root = self.canonical_root()?;
            let candidate = root.join(path);

            let resolved = if allow_create {
                let parent = candidate
                    .parent()
                    .ok_or_else(|| ToolError::InvalidArgs("invalid path".into()))?;

                let canonical_parent = parent.canonicalize().or_else(|_| {
                    stdfs::create_dir_all(parent).map_err(|e| {
                        ToolError::Execution(format!("failed to create parent: {e}"))
                    })?;
                    parent
                        .canonicalize()
                        .map_err(|e| ToolError::Execution(format!("failed to access path: {e}")))
                })?;

                let file_name = candidate
                    .file_name()
                    .ok_or_else(|| ToolError::InvalidArgs("invalid path".into()))?;

                canonical_parent.join(file_name)
            } else {
                candidate
                    .canonicalize()
                    .map_err(|e| ToolError::Execution(format!("failed to access path: {e}")))?
            };

            if !resolved.starts_with(&root) {
                return Err(ToolError::InvalidArgs("path escapes sandbox".into()));
            }
            Ok(resolved)
        }
    }

    #[async_trait]
    impl Tool for FileTool {
        fn name(&self) -> &'static str {
            "file"
        }

        fn input_schema(&self) -> Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "operation": {"type": "string", "enum": ["read", "write"]},
                    "content": {"type": "string"}
                },
                "required": ["path", "operation"],
                "additionalProperties": false
            })
        }

        fn output_schema(&self) -> Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "operation": {"type": "string"},
                    "content": {"type": "string"},
                    "bytes": {"type": "integer"}
                },
                "required": ["path", "operation"],
                "additionalProperties": false
            })
        }

        async fn execute(&self, args: Value) -> Result<Value, ToolError> {
            let path = args
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::InvalidArgs("path missing".into()))?;
            let op = args
                .get("operation")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::InvalidArgs("operation missing".into()))?;
            let resolved = self.resolve(path, op == "write")?;

            match op {
                "read" => {
                    let content = fs::read_to_string(&resolved)
                        .await
                        .map_err(|e| ToolError::Execution(format!("read failed: {e}")))?;
                    Ok(serde_json::json!({
                        "path": resolved.display().to_string(),
                        "operation": "read",
                        "content": content
                    }))
                }
                "write" => {
                    let content =
                        args.get("content")
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| {
                                ToolError::InvalidArgs("content missing for write".into())
                            })?;
                    if let Some(parent) = resolved.parent() {
                        fs::create_dir_all(parent).await.map_err(|e| {
                            ToolError::Execution(format!("failed to create directories: {e}"))
                        })?;
                    }
                    fs::write(&resolved, content)
                        .await
                        .map_err(|e| ToolError::Execution(format!("write failed: {e}")))?;
                    Ok(serde_json::json!({
                        "path": resolved.display().to_string(),
                        "operation": "write",
                        "bytes": content.len()
                    }))
                }
                _ => Err(ToolError::InvalidArgs("unsupported operation".into())),
            }
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
            let value: f64 =
                meval::eval_str(expr).map_err(|e| ToolError::Execution(e.to_string()))?;
            Ok(Value::from(value))
        }
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct SearchResult {
        pub title: String,
        pub url: String,
        pub snippet: String,
    }

    #[async_trait]
    pub trait SearchProvider: Send + Sync {
        async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>, ToolError>;
    }

    pub struct SearchTool<P: SearchProvider> {
        provider: std::sync::Arc<P>,
    }

    impl<P: SearchProvider> SearchTool<P> {
        pub fn new(provider: std::sync::Arc<P>) -> Self {
            Self { provider }
        }
    }

    #[async_trait]
    impl<P: SearchProvider + 'static> Tool for SearchTool<P> {
        fn name(&self) -> &'static str {
            "search"
        }

        fn input_schema(&self) -> Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50}
                },
                "required": ["query"]
            })
        }

        fn output_schema(&self) -> Value {
            serde_json::json!({
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "snippet": {"type": "string"}
                    },
                    "required": ["title", "url", "snippet"]
                }
            })
        }

        async fn execute(&self, args: Value) -> Result<Value, ToolError> {
            let query = args
                .get("query")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::InvalidArgs("query missing".into()))?;
            let limit = args
                .get("limit")
                .and_then(|v| v.as_u64())
                .unwrap_or(5)
                .min(50) as usize;

            let results = self.provider.search(query, limit).await?;
            Ok(serde_json::to_value(results).map_err(|e| ToolError::Execution(e.to_string()))?)
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
            Self {
                client: reqwest::Client::new(),
            }
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

#[cfg(test)]
mod tests {
    use super::builtins::{FileTool, SearchProvider, SearchResult, SearchTool};
    use super::{ToolError, ToolInvocationError, ToolMetadata, ToolRegistry};
    use crate::Tool;
    use serde_json::json;
    use std::sync::Arc;
    use std::time::Duration;

    #[tokio::test]
    async fn file_tool_read_write_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let tool = FileTool::new(dir.path());

        let write_result = tool
            .execute(
                json!({"path": "notes/hello.txt", "operation": "write", "content": "hi there"}),
            )
            .await
            .unwrap();
        assert_eq!(write_result.get("operation").unwrap(), "write");

        let read_result = tool
            .execute(json!({"path": "notes/hello.txt", "operation": "read"}))
            .await
            .unwrap();
        assert_eq!(read_result.get("operation").unwrap(), "read");
        assert_eq!(read_result.get("content").unwrap(), "hi there");
    }

    #[tokio::test]
    async fn file_tool_rejects_traversal() {
        let dir = tempfile::tempdir().unwrap();
        let tool = FileTool::new(dir.path());

        let result = tool
            .execute(json!({"path": "../evil.txt", "operation": "write", "content": "nope"}))
            .await;

        assert!(matches!(result, Err(ToolError::InvalidArgs(_))));
    }

    struct StaticSearchProvider {
        results: Vec<SearchResult>,
    }

    #[async_trait::async_trait]
    impl SearchProvider for StaticSearchProvider {
        async fn search(&self, _query: &str, limit: usize) -> Result<Vec<SearchResult>, ToolError> {
            Ok(self.results.iter().take(limit).cloned().collect())
        }
    }

    #[tokio::test]
    async fn search_tool_returns_results() {
        let provider = Arc::new(StaticSearchProvider {
            results: vec![SearchResult {
                title: "Example".into(),
                url: "https://example.com".into(),
                snippet: "Example domain".into(),
            }],
        });

        let tool = SearchTool::new(provider);
        let output = tool
            .execute(json!({"query": "example", "limit": 3}))
            .await
            .unwrap();

        assert_eq!(output.as_array().unwrap().len(), 1);
        assert_eq!(output[0]["title"], "Example");
    }

    #[tokio::test]
    async fn registry_enforces_cooldown_and_access() {
        struct NoopTool;

        #[async_trait::async_trait]
        impl super::Tool for NoopTool {
            fn name(&self) -> &'static str {
                "noop"
            }

            fn input_schema(&self) -> serde_json::Value {
                json!({"type": "object"})
            }

            fn output_schema(&self) -> serde_json::Value {
                json!({"type": "null"})
            }

            async fn execute(
                &self,
                _args: serde_json::Value,
            ) -> Result<serde_json::Value, ToolError> {
                Ok(json!(null))
            }
        }

        let mut registry = ToolRegistry::new();
        registry.register_with_metadata(
            NoopTool,
            ToolMetadata {
                allowed_roles: vec!["admin".into()],
                cooldown: Some(Duration::from_millis(50)),
                description: None,
                tags: vec![],
            },
        );

        let denied = registry
            .invoke("noop", json!({}), &["guest".into()])
            .await
            .unwrap_err();
        assert!(matches!(denied, ToolInvocationError::AccessDenied { .. }));

        let first = registry
            .invoke("noop", json!({}), &["admin".into()])
            .await
            .unwrap();
        assert!(first.is_null());

        let cooldown = registry
            .invoke("noop", json!({}), &["admin".into()])
            .await
            .unwrap_err();
        assert!(matches!(cooldown, ToolInvocationError::CoolingDown { .. }));
    }
}
