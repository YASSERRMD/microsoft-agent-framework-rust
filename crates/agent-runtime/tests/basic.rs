use agent_core::{
    Agent, AgentConfig, AgentContext, AgentError, AgentState, Plan, RetryPolicy, Step, StepOutcome,
    StepPolicies, ToolPermissions,
};
use agent_runtime::{
    ControlLoop, ControlMode, InMemoryBus, MemoryTopology, MultiAgentOrchestrator, StepExecutor,
};
use serde_json::json;
use std::sync::Arc;
use std::sync::Mutex;

#[derive(Debug)]
struct TestAgent;

#[async_trait::async_trait]
impl Agent for TestAgent {
    async fn plan(&self, _ctx: &agent_core::AgentContext) -> Result<Plan, AgentError> {
        Ok(Plan {
            goal: "test".into(),
            steps: vec![Step {
                id: "one".into(),
                description: "noop".into(),
                tool: None,
                args: json!({}),
                subtasks: vec![],
                policies: StepPolicies::default(),
                chain_of_thought: None,
            }],
            metadata: json!({}),
        })
    }

    async fn execute_step(
        &self,
        step: &Step,
        _ctx: &mut AgentContext,
    ) -> Result<StepOutcome, AgentError> {
        Ok(StepOutcome {
            step_id: step.id.clone(),
            output: json!({"ok": true}),
            observations: vec![],
            success: true,
            retries: 0,
            fallback_used: false,
            control_notes: vec![],
        })
    }
}

#[tokio::test]
async fn runs_control_loop() {
    let agent = TestAgent;
    let mut ctx = AgentContext {
        config: AgentConfig {
            name: "t".into(),
            description: None,
            max_iterations: 2,
            retry_policy: RetryPolicy::default(),
        },
        state: AgentState::default(),
        metadata: json!({}),
        memory: None,
        tool_permissions: ToolPermissions::default(),
    };
    let loop_ctrl = ControlLoop {
        max_iterations: 2,
        delay: std::time::Duration::from_millis(0),
        mode: ControlMode::Deterministic,
    };
    let outcomes = loop_ctrl.run(&agent, &mut ctx).await.expect("loop to run");
    assert_eq!(outcomes.len(), 1);
    assert!(outcomes[0].success);
}

#[derive(Debug)]
struct FlakyAgent {
    attempts: Arc<Mutex<usize>>,
}

#[async_trait::async_trait]
impl Agent for FlakyAgent {
    async fn plan(&self, _ctx: &agent_core::AgentContext) -> Result<Plan, AgentError> {
        Ok(Plan {
            goal: "flaky".into(),
            steps: vec![Step {
                id: "retry".into(),
                description: "sometimes fails".into(),
                tool: None,
                args: json!({}),
                subtasks: vec![],
                policies: StepPolicies {
                    retry: RetryPolicy {
                        max_retries: 1,
                        backoff_ms: 0,
                        jitter: false,
                    },
                    ..Default::default()
                },
                chain_of_thought: None,
            }],
            metadata: json!({}),
        })
    }

    async fn execute_step(
        &self,
        step: &Step,
        _ctx: &mut AgentContext,
    ) -> Result<StepOutcome, AgentError> {
        let mut attempts = self.attempts.lock().unwrap();
        *attempts += 1;
        if *attempts == 1 {
            return Err(AgentError::Execution("first attempt fails".into()));
        }
        Ok(StepOutcome::success(step.id.clone(), json!({"ok": true})))
    }
}

#[tokio::test]
async fn step_executor_retries_and_records_counts() {
    let agent = FlakyAgent {
        attempts: Arc::new(Mutex::new(0)),
    };
    let mut ctx = AgentContext {
        config: AgentConfig::default(),
        state: AgentState::default(),
        metadata: json!({}),
        memory: None,
        tool_permissions: ToolPermissions::default(),
    };
    let plan = agent.plan(&ctx).await.expect("plan available");
    let step = plan.steps.first().cloned().expect("step present");
    let outcome = StepExecutor::run_step(step, &agent, &mut ctx).await;
    assert!(outcome.success);
    assert_eq!(outcome.retries, 1);
}

#[derive(Debug)]
struct AlternateToolAgent;

#[async_trait::async_trait]
impl Agent for AlternateToolAgent {
    async fn plan(&self, _ctx: &agent_core::AgentContext) -> Result<Plan, AgentError> {
        Ok(Plan {
            goal: "fallback".into(),
            steps: vec![Step {
                id: "main".into(),
                description: "use alternate".into(),
                tool: None,
                args: json!({}),
                subtasks: vec![],
                policies: StepPolicies {
                    fallback: Some(agent_core::FallbackPolicy {
                        strategy: agent_core::FallbackStrategy::AlternateTool {
                            tool: "alt".into(),
                        },
                        reason: None,
                    }),
                    ..Default::default()
                },
                chain_of_thought: None,
            }],
            metadata: json!({}),
        })
    }

    async fn execute_step(
        &self,
        step: &Step,
        _ctx: &mut AgentContext,
    ) -> Result<StepOutcome, AgentError> {
        if step.tool.as_deref() == Some("alt") {
            Ok(StepOutcome::success(step.id.clone(), json!({"alt": true})))
        } else {
            Err(AgentError::Execution("primary tool unavailable".into()))
        }
    }
}

#[tokio::test]
async fn fallback_switches_tool() {
    let agent = AlternateToolAgent;
    let mut ctx = AgentContext {
        config: AgentConfig::default(),
        state: AgentState::default(),
        metadata: json!({}),
        memory: None,
        tool_permissions: ToolPermissions::default(),
    };
    let plan = agent.plan(&ctx).await.expect("plan available");
    let step = plan.steps.first().cloned().expect("step present");
    let outcome = StepExecutor::run_step(step, &agent, &mut ctx).await;
    assert!(outcome.success);
    assert!(outcome.fallback_used);
    assert_eq!(outcome.output["alt"], json!(true));
}

#[derive(Debug)]
struct ModeAwareAgent;

#[async_trait::async_trait]
impl Agent for ModeAwareAgent {
    async fn plan(&self, ctx: &agent_core::AgentContext) -> Result<Plan, AgentError> {
        Ok(Plan {
            goal: "mode".into(),
            steps: vec![Step {
                id: format!("{}", ctx.state.iteration),
                description: "id matches iteration".into(),
                tool: None,
                args: json!({}),
                subtasks: vec![],
                policies: StepPolicies::default(),
                chain_of_thought: None,
            }],
            metadata: json!({}),
        })
    }

    async fn execute_step(
        &self,
        step: &Step,
        _ctx: &mut AgentContext,
    ) -> Result<StepOutcome, AgentError> {
        Ok(StepOutcome::success(step.id.clone(), json!({"ok": true})))
    }
}

#[tokio::test]
async fn reactive_mode_replans_each_iteration() {
    let agent = ModeAwareAgent;
    let mut ctx = AgentContext {
        config: AgentConfig {
            name: "reactive".into(),
            description: None,
            max_iterations: 2,
            retry_policy: RetryPolicy::default(),
        },
        state: AgentState::default(),
        metadata: json!({}),
        memory: None,
        tool_permissions: ToolPermissions::default(),
    };
    let loop_ctrl = ControlLoop {
        max_iterations: 2,
        delay: std::time::Duration::from_millis(0),
        mode: ControlMode::Reactive,
    };
    let outcomes = loop_ctrl.run(&agent, &mut ctx).await.expect("loop to run");
    assert_eq!(outcomes.len(), 2);
    assert_eq!(outcomes[0].step_id, "0");
    assert_eq!(outcomes[1].step_id, "1");
}

#[derive(Debug)]
struct ReflectiveAgent {
    reflections: Arc<Mutex<usize>>,
}

#[async_trait::async_trait]
impl Agent for ReflectiveAgent {
    async fn plan(&self, _ctx: &agent_core::AgentContext) -> Result<Plan, AgentError> {
        Ok(Plan {
            goal: "reflect".into(),
            steps: vec![Step {
                id: "r1".into(),
                description: "single".into(),
                tool: None,
                args: json!({}),
                subtasks: vec![],
                policies: StepPolicies::default(),
                chain_of_thought: None,
            }],
            metadata: json!({}),
        })
    }

    async fn execute_step(
        &self,
        step: &Step,
        _ctx: &mut AgentContext,
    ) -> Result<StepOutcome, AgentError> {
        Ok(StepOutcome::success(step.id.clone(), json!({"ok": true})))
    }

    async fn reflect(&self, _ctx: &mut AgentContext) -> Result<(), AgentError> {
        let mut count = self.reflections.lock().unwrap();
        *count += 1;
        Ok(())
    }
}

#[tokio::test]
async fn reflection_enabled_mode_reflects_per_step() {
    let agent = ReflectiveAgent {
        reflections: Arc::new(Mutex::new(0)),
    };
    let mut ctx = AgentContext {
        config: AgentConfig::default(),
        state: AgentState::default(),
        metadata: json!({}),
        memory: None,
        tool_permissions: ToolPermissions::default(),
    };
    let loop_ctrl = ControlLoop {
        max_iterations: 1,
        delay: std::time::Duration::from_millis(0),
        mode: ControlMode::ReflectionEnabled,
    };
    loop_ctrl.run(&agent, &mut ctx).await.expect("loop to run");
    assert_eq!(*agent.reflections.lock().unwrap(), 2);
}

#[tokio::test]
async fn orchestrator_shares_memory_and_routes_messages() {
    let shared_store = Arc::new(agent_memory::InMemoryStore::new());
    let shared_dyn: Arc<dyn agent_memory::MemoryStore> = shared_store.clone();
    let bus = InMemoryBus::new();
    let mut orchestrator =
        MultiAgentOrchestrator::new(bus, MemoryTopology::Shared(shared_dyn.clone()));

    let base_ctx = AgentContext {
        config: AgentConfig::default(),
        state: AgentState::default(),
        metadata: json!({}),
        memory: None,
        tool_permissions: ToolPermissions::default(),
    };

    orchestrator.register_agent("alpha", base_ctx.clone());
    orchestrator.register_agent("beta", base_ctx.clone());

    let mut prepared = base_ctx.clone();
    orchestrator.prepare_context(&mut prepared);
    assert!(prepared.memory.is_some());
    let shared_memory = prepared.memory.unwrap();
    assert!(Arc::ptr_eq(&shared_memory, &shared_dyn));

    orchestrator
        .send_message("beta", json!({"ping": true}))
        .await
        .expect("message sent");
    let received = orchestrator
        .recv_message("beta")
        .await
        .expect("message received");
    assert_eq!(received.unwrap()["ping"], json!(true));
}
