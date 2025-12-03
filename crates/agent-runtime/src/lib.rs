use agent_core::{
    Agent, AgentContext, AgentError, ExecutablePlan, Plan, RetryPolicy, Step, StepOutcome,
};
use async_trait::async_trait;
use rand::Rng;
use std::{collections::HashMap, sync::Arc};
use tokio::time::{sleep, Duration};
use tracing::instrument;

use agent_memory::MemoryStore;

pub struct StepExecutor;

impl StepExecutor {
    pub async fn run_step<A: Agent>(step: Step, agent: &A, ctx: &mut AgentContext) -> StepOutcome {
        let retry_policy = resolve_retry_policy(&step, &ctx.config.retry_policy);
        let mut retries = 0usize;

        loop {
            match agent.act(&step, ctx).await {
                Ok(mut outcome) => {
                    outcome.retries = retries;
                    return outcome;
                }
                Err(err) => {
                    if retries < retry_policy.max_retries {
                        let delay = backoff_delay(&retry_policy, retries);
                        retries += 1;
                        if delay > Duration::from_millis(0) {
                            sleep(delay).await;
                        }
                        continue;
                    }

                    return Self::apply_fallback(step.clone(), agent, ctx, err, retries).await;
                }
            }
        }
    }

    async fn apply_fallback<A: Agent>(
        step: Step,
        agent: &A,
        ctx: &mut AgentContext,
        error: AgentError,
        retries: usize,
    ) -> StepOutcome {
        match &step.policies.fallback {
            Some(policy) => match &policy.strategy {
                agent_core::FallbackStrategy::Skip => StepOutcome {
                    step_id: step.id,
                    output: serde_json::json!({"skipped": true, "error": error.to_string()}),
                    observations: vec!["skipped via fallback".to_string()],
                    success: false,
                    retries,
                    fallback_used: true,
                    control_notes: vec!["fallback: skip".to_string()],
                },
                agent_core::FallbackStrategy::Abort => StepOutcome {
                    step_id: step.id,
                    output: serde_json::json!({"error": error.to_string()}),
                    observations: vec!["aborted via fallback".to_string()],
                    success: false,
                    retries,
                    fallback_used: true,
                    control_notes: vec!["fallback: abort".to_string()],
                },
                agent_core::FallbackStrategy::RetryWithLimit {
                    max_additional_retries,
                } => {
                    let mut total_retries = retries;
                    for attempt in 0..=*max_additional_retries {
                        if attempt > 0 {
                            total_retries += 1;
                        }

                        match agent.act(&step, ctx).await {
                            Ok(mut outcome) => {
                                outcome.retries = total_retries;
                                outcome.fallback_used = true;
                                outcome.control_notes.push("fallback: retry".to_string());
                                return outcome;
                            }
                            Err(err) => {
                                if attempt == *max_additional_retries {
                                    return StepOutcome {
                                        step_id: step.id.clone(),
                                        output: serde_json::json!({"error": err.to_string()}),
                                        observations: vec!["retry fallback exhausted".to_string()],
                                        success: false,
                                        retries: total_retries,
                                        fallback_used: true,
                                        control_notes: vec!["fallback: retry exhausted".to_string()],
                                    };
                                }
                            }
                        }
                    }

                    StepOutcome::failure(step.id, error)
                }
                agent_core::FallbackStrategy::AlternateTool { tool } => {
                    let mut alternate = step.clone();
                    alternate.tool = Some(tool.clone());
                    let mut outcome = match agent.act(&alternate, ctx).await {
                        Ok(outcome) => outcome,
                        Err(err) => {
                            return StepOutcome {
                                step_id: alternate.id,
                                output: serde_json::json!({"error": err.to_string()}),
                                observations: vec!["alternate tool failed".to_string()],
                                success: false,
                                retries,
                                fallback_used: true,
                                control_notes: vec!["fallback: alternate tool".to_string()],
                            }
                        }
                    };

                    outcome.retries = retries;
                    outcome.fallback_used = true;
                    outcome
                        .control_notes
                        .push("fallback: alternate tool".to_string());
                    outcome
                }
            },
            None => StepOutcome::failure(step.id, error),
        }
    }
}

fn resolve_retry_policy(step: &Step, default_policy: &RetryPolicy) -> RetryPolicy {
    if step.policies.retry.max_retries > 0
        || step.policies.retry.backoff_ms > 0
        || step.policies.retry.jitter
    {
        step.policies.retry.clone()
    } else {
        default_policy.clone()
    }
}

fn backoff_delay(policy: &RetryPolicy, retry_count: usize) -> Duration {
    let base = policy.backoff_ms * (retry_count as u64 + 1);
    if base == 0 {
        return Duration::from_millis(0);
    }

    if policy.jitter {
        let jitter: u64 = rand::thread_rng().gen_range(0..=policy.backoff_ms.max(1));
        Duration::from_millis(base + jitter)
    } else {
        Duration::from_millis(base)
    }
}

#[derive(Default)]
pub struct ControlLoop {
    pub max_iterations: usize,
    pub delay: Duration,
    pub mode: ControlMode,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ControlMode {
    #[default]
    Deterministic,
    Reactive,
    Procedural,
    ReflectionEnabled,
}

impl ControlLoop {
    #[instrument(skip_all)]
    pub async fn run<A: Agent>(
        &self,
        agent: &A,
        ctx: &mut AgentContext,
    ) -> Result<Vec<StepOutcome>, AgentError> {
        agent.initialize(ctx).await?;
        let mut executable: Option<ExecutablePlan> = None;
        if matches!(
            self.mode,
            ControlMode::Deterministic | ControlMode::ReflectionEnabled
        ) {
            let plan: Plan = agent.think(ctx).await?;
            executable = Some(plan.executable());
        }
        let mut results = Vec::new();

        for iteration in 0..self.max_iterations {
            ctx.state.iteration = iteration;

            let next_step = match self.mode {
                ControlMode::Deterministic | ControlMode::ReflectionEnabled => {
                    executable.as_mut().and_then(|plan| plan.next())
                }
                ControlMode::Reactive => {
                    let plan: Plan = agent.think(ctx).await?;
                    let mut plan_exec = plan.executable();
                    plan_exec.next()
                }
                ControlMode::Procedural => {
                    if let Some(step) = executable.as_mut().and_then(|plan| plan.next()) {
                        Some(step)
                    } else {
                        let plan: Plan = agent.think(ctx).await?;
                        executable = Some(plan.executable());
                        executable.as_mut().and_then(|plan| plan.next())
                    }
                }
            };

            if let Some(step) = next_step {
                let outcome = StepExecutor::run_step(step.clone(), agent, ctx).await;
                agent.observe(&outcome, ctx).await?;
                results.push(outcome);

                if matches!(self.mode, ControlMode::ReflectionEnabled) {
                    agent.reflect(ctx).await?;
                }
            } else {
                break;
            }
            if self.delay > Duration::from_millis(0) {
                sleep(self.delay).await;
            }
        }

        if matches!(self.mode, ControlMode::ReflectionEnabled) {
            agent.reflect(ctx).await?;
        }

        if !matches!(self.mode, ControlMode::ReflectionEnabled) {
            agent.reflect(ctx).await?;
        }
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

pub enum MemoryTopology {
    Shared(Arc<dyn MemoryStore>),
    Isolated,
}

pub struct MultiAgentOrchestrator<B: MessageBus> {
    bus: Arc<B>,
    memory_topology: MemoryTopology,
    agents: HashMap<String, AgentContext>,
}

impl<B: MessageBus> MultiAgentOrchestrator<B> {
    pub fn new(bus: B, memory_topology: MemoryTopology) -> Self {
        Self {
            bus: Arc::new(bus),
            memory_topology,
            agents: HashMap::new(),
        }
    }

    pub fn register_agent<T: Into<String>>(&mut self, name: T, ctx: AgentContext) {
        self.agents.insert(name.into(), ctx);
    }

    pub fn prepare_context(&self, ctx: &mut AgentContext) {
        if let MemoryTopology::Shared(store) = &self.memory_topology {
            ctx.memory = Some(store.clone());
        }
    }

    pub async fn call_agent<A: Agent>(
        &self,
        name: &str,
        agent: &A,
        control: &ControlLoop,
    ) -> Result<Vec<StepOutcome>, AgentError> {
        let mut ctx = self
            .agents
            .get(name)
            .cloned()
            .unwrap_or_else(|| AgentContext {
                config: agent_core::AgentConfig::default(),
                state: agent_core::AgentState::default(),
                metadata: serde_json::json!({}),
                memory: None,
                tool_permissions: agent_core::ToolPermissions::default(),
            });
        self.prepare_context(&mut ctx);
        control.run(agent, &mut ctx).await
    }

    pub async fn send_message(
        &self,
        recipient: &str,
        message: serde_json::Value,
    ) -> Result<(), AgentError> {
        self.bus.send(recipient, message).await
    }

    pub async fn recv_message(
        &self,
        recipient: &str,
    ) -> Result<Option<serde_json::Value>, AgentError> {
        self.bus.recv(recipient).await
    }
}
