use agent_core::{Agent, AgentContext, AgentError, ExecutablePlan, Plan, Step, StepOutcome};
use async_trait::async_trait;
use tokio::time::{sleep, Duration};
use tracing::instrument;

pub struct StepExecutor;

impl StepExecutor {
    pub async fn run_step<A: Agent>(step: Step, agent: &A, ctx: &mut AgentContext) -> StepOutcome {
        match agent.act(&step, ctx).await {
            Ok(outcome) => outcome,
            Err(err) => StepOutcome {
                step_id: step.id,
                output: serde_json::json!({"error": err.to_string()}),
                observations: vec!["step failed".to_string()],
                success: false,
            },
        }
    }
}

#[derive(Default)]
pub struct ControlLoop {
    pub max_iterations: usize,
    pub delay: Duration,
}

impl ControlLoop {
    #[instrument(skip_all)]
    pub async fn run<A: Agent>(&self, agent: &A, ctx: &mut AgentContext) -> Result<Vec<StepOutcome>, AgentError> {
        agent.initialize(ctx).await?;
        let plan: Plan = agent.think(ctx).await?;
        let mut executable = plan.executable();
        let mut results = Vec::new();

        for iteration in 0..self.max_iterations {
            if let Some(step) = executable.next() {
                ctx.state.iteration = iteration;
                let outcome = StepExecutor::run_step(step.clone(), agent, ctx).await;
                agent.observe(&outcome, ctx).await?;
                results.push(outcome);
            } else {
                break;
            }
            if self.delay > Duration::from_millis(0) {
                sleep(self.delay).await;
            }
        }

        agent.reflect(ctx).await?;
        Ok(results)
    }
}

#[async_trait]
pub trait MessageBus {
    async fn send(&self, recipient: &str, message: serde_json::Value) -> Result<(), AgentError>;
    async fn recv(&self, recipient: &str) -> Result<Option<serde_json::Value>, AgentError>;
}

pub struct InMemoryBus {
    messages: tokio::sync::Mutex<Vec<(String, serde_json::Value)>>,
}

impl InMemoryBus {
    pub fn new() -> Self {
        Self {
            messages: tokio::sync::Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl MessageBus for InMemoryBus {
    async fn send(&self, recipient: &str, message: serde_json::Value) -> Result<(), AgentError> {
        self.messages
            .lock()
            .await
            .push((recipient.to_string(), message));
        Ok(())
    }

    async fn recv(&self, recipient: &str) -> Result<Option<serde_json::Value>, AgentError> {
        let mut messages = self.messages.lock().await;
        if let Some(pos) = messages.iter().position(|(r, _)| r == recipient) {
            Ok(Some(messages.remove(pos).1))
        } else {
            Ok(None)
        }
    }
}
