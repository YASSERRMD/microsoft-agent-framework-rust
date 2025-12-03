use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt::Debug;
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentConfig {
    pub name: String,
    pub description: Option<String>,
    pub max_iterations: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentState {
    pub plan: Option<Plan>,
    pub memory_keys: Vec<String>,
    pub iteration: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentContext {
    pub config: AgentConfig,
    pub state: AgentState,
    pub metadata: Value,
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
    #[error("timeout")]
    Timeout,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    pub goal: String,
    pub steps: Vec<Step>,
}

impl Plan {
    pub fn executable(self) -> ExecutablePlan {
        ExecutablePlan { plan: self, current: 0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    pub id: String,
    pub description: String,
    pub tool: Option<String>,
    pub args: Value,
    pub subtasks: Vec<Subtask>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepOutcome {
    pub step_id: String,
    pub output: Value,
    pub observations: Vec<String>,
    pub success: bool,
}

#[async_trait]
pub trait Agent: Send + Sync + Debug {
    async fn plan(&self, ctx: &AgentContext) -> Result<Plan, AgentError>;
    async fn execute_step(&self, step: &Step, ctx: &mut AgentContext) -> Result<StepOutcome, AgentError>;

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

    async fn observe(&self, _outcome: &StepOutcome, _ctx: &mut AgentContext) -> Result<(), AgentError> {
        Ok(())
    }

    async fn reflect(&self, _ctx: &mut AgentContext) -> Result<(), AgentError> {
        Ok(())
    }
}
