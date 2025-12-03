use agent_memory::MemoryStore;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{fmt::Debug, sync::Arc};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentConfig {
    pub name: String,
    pub description: Option<String>,
    pub max_iterations: usize,
    pub retry_policy: RetryPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentState {
    pub plan: Option<Plan>,
    pub memory_keys: Vec<String>,
    pub iteration: usize,
    pub step_history: Vec<StepOutcome>,
    pub chain_of_thought: Option<ChainOfThought>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentContext {
    pub config: AgentConfig,
    pub state: AgentState,
    pub metadata: Value,
    #[serde(skip_serializing, skip_deserializing)]
    pub memory: Option<Arc<dyn MemoryStore>>,
    #[serde(skip_serializing, skip_deserializing)]
    pub tool_permissions: ToolPermissions,
}

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("planning failed: {0}")]
    Planning(String),
    #[error("execution failed: {0}")]
    Execution(String),
    #[error("tool failure: {0}")]
    Tool(String),
    #[error("memory failure: {0}")]
    Memory(String),
    #[error("safety violation: {0}")]
    Safety(String),
    #[error("timeout")]
    Timeout,
    #[error("validation failed: {0}")]
    Validation(String),
    #[error("retry exhausted after {attempts} attempts")]
    RetryExhausted { attempts: usize },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub max_retries: usize,
    pub backoff_ms: u64,
    pub jitter: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 0,
            backoff_ms: 0,
            jitter: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    pub goal: String,
    pub steps: Vec<Step>,
    pub metadata: Value,
}

impl Plan {
    pub fn executable(self) -> ExecutablePlan {
        ExecutablePlan {
            plan: self,
            current: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    pub id: String,
    pub description: String,
    pub tool: Option<String>,
    pub args: Value,
    pub subtasks: Vec<Subtask>,
    pub policies: StepPolicies,
    #[serde(skip_serializing, skip_deserializing)]
    pub chain_of_thought: Option<ChainOfThought>,
}

impl Step {
    pub fn with_tool<T: Into<String>>(mut self, tool: T, args: Value) -> Self {
        self.tool = Some(tool.into());
        self.args = args;
        self
    }

    pub fn add_cot_note<T: Into<String>>(&mut self, note: T) {
        let mut cot = self.chain_of_thought.take().unwrap_or_default();
        cot.push(note);
        self.chain_of_thought = Some(cot);
    }

    pub fn record_subtask<T: Into<String>>(&mut self, id: T, description: T) {
        self.subtasks.push(Subtask {
            id: id.into(),
            description: description.into(),
        });
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StepPolicies {
    pub retry: RetryPolicy,
    pub fallback: Option<FallbackPolicy>,
    pub safety: SafetyPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subtask {
    pub id: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutablePlan {
    pub plan: Plan,
    pub current: usize,
}

impl ExecutablePlan {
    pub fn next(&mut self) -> Option<Step> {
        if self.current < self.plan.steps.len() {
            let step = self.plan.steps[self.current].clone();
            self.current += 1;
            Some(step)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SafetyPolicy {
    pub allow_tool_execution: bool,
    pub redaction_rules: Vec<String>,
    pub rbac_roles: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackPolicy {
    pub strategy: FallbackStrategy,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum FallbackStrategy {
    Skip,
    RetryWithLimit { max_additional_retries: usize },
    AlternateTool { tool: String },
    Abort,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChainOfThought {
    notes: Vec<String>,
}

impl ChainOfThought {
    pub fn new() -> Self {
        Self { notes: Vec::new() }
    }

    pub fn push<T: Into<String>>(&mut self, note: T) {
        self.notes.push(note.into());
    }

    pub fn notes(&self) -> &[String] {
        &self.notes
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolPermissions {
    pub allowed: Vec<String>,
    pub denied: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepOutcome {
    pub step_id: String,
    pub output: Value,
    pub observations: Vec<String>,
    pub success: bool,
    pub retries: usize,
    pub fallback_used: bool,
    pub control_notes: Vec<String>,
}

impl StepOutcome {
    pub fn success(step_id: String, output: Value) -> Self {
        Self {
            step_id,
            output,
            observations: Vec::new(),
            success: true,
            retries: 0,
            fallback_used: false,
            control_notes: Vec::new(),
        }
    }

    pub fn failure(step_id: String, error: AgentError) -> Self {
        Self {
            step_id,
            output: serde_json::json!({ "error": error.to_string() }),
            observations: vec!["step failed".to_string()],
            success: false,
            retries: 0,
            fallback_used: false,
            control_notes: vec!["failure".to_string()],
        }
    }
}

#[async_trait]
pub trait Agent: Send + Sync + Debug {
    async fn plan(&self, ctx: &AgentContext) -> Result<Plan, AgentError>;
    async fn execute_step(
        &self,
        step: &Step,
        ctx: &mut AgentContext,
    ) -> Result<StepOutcome, AgentError>;

    async fn initialize(&self, ctx: &mut AgentContext) -> Result<(), AgentError> {
        tracing::debug!(agent = %ctx.config.name, "initializing agent");
        Ok(())
    }

    async fn think(&self, ctx: &AgentContext) -> Result<Plan, AgentError> {
        self.plan(ctx).await
    }

    async fn act(&self, step: &Step, ctx: &mut AgentContext) -> Result<StepOutcome, AgentError> {
        self.execute_step(step, ctx).await
    }

    async fn observe(
        &self,
        _outcome: &StepOutcome,
        _ctx: &mut AgentContext,
    ) -> Result<(), AgentError> {
        Ok(())
    }

    async fn reflect(&self, _ctx: &mut AgentContext) -> Result<(), AgentError> {
        Ok(())
    }
}
