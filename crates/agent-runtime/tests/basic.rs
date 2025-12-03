use agent_core::{Agent, AgentConfig, AgentContext, AgentError, AgentState, Plan, Step, StepOutcome};
use agent_runtime::ControlLoop;
use serde_json::json;

#[derive(Debug)]
struct TestAgent;

#[async_trait::async_trait]
impl Agent for TestAgent {
    async fn plan(&self, _ctx: &agent_core::AgentContext) -> Result<Plan, AgentError> {
        Ok(Plan {
            goal: "test".into(),
            steps: vec![Step { id: "one".into(), description: "noop".into(), tool: None, args: json!({}), subtasks: vec![] }],
        })
    }

    async fn execute_step(&self, step: &Step, _ctx: &mut AgentContext) -> Result<StepOutcome, AgentError> {
        Ok(StepOutcome { step_id: step.id.clone(), output: json!({"ok": true}), observations: vec![], success: true })
    }
}

#[tokio::test]
async fn runs_control_loop() {
    let agent = TestAgent;
    let mut ctx = AgentContext { config: AgentConfig { name: "t".into(), description: None, max_iterations: 2 }, state: AgentState::default(), metadata: json!({}) };
    let loop_ctrl = ControlLoop { max_iterations: 2, delay: std::time::Duration::from_millis(0) };
    let outcomes = loop_ctrl.run(&agent, &mut ctx).await.expect("loop to run");
    assert_eq!(outcomes.len(), 1);
    assert!(outcomes[0].success);
}
