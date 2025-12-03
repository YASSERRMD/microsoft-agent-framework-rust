use agent_core::{
    Agent, AgentContext, AgentError, FallbackPolicy, FallbackStrategy, Plan, Step, StepOutcome,
};
use agent_examples::common::{base_context, default_policies, deterministic_loop};
use agent_models::{LLMModel, StubModel};
use agent_runtime::ControlLoop;
use serde_json::json;
use std::fmt;
use std::sync::Arc;

struct PlanningExecutionAgent {
    tools: Arc<agent_tools::ToolRegistry>,
    model: StubModel,
}

impl fmt::Debug for PlanningExecutionAgent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PlanningExecutionAgent").finish()
    }
}

fn planned_step(id: &str, description: &str, tool: Option<&str>, args: serde_json::Value) -> Step {
    let mut policies = default_policies();
    if id == "fallback-example" {
        policies.fallback = Some(FallbackPolicy {
            strategy: FallbackStrategy::Skip,
            reason: Some("Non-critical step".into()),
        });
    }
    Step {
        id: id.into(),
        description: description.into(),
        tool: tool.map(|t| t.to_string()),
        args,
        subtasks: vec![],
        policies,
        chain_of_thought: None,
    }
}

#[async_trait::async_trait]
impl Agent for PlanningExecutionAgent {
    async fn plan(&self, _ctx: &AgentContext) -> Result<Plan, AgentError> {
        Ok(Plan {
            goal: "Plan tasks then execute with fallbacks".into(),
            steps: vec![
                planned_step(
                    "outline",
                    "Log the execution outline",
                    Some("log"),
                    json!({"message": "Planning+Execution run started"}),
                ),
                planned_step(
                    "calc",
                    "Run a calculation",
                    Some("math"),
                    json!({"expression": "10/2"}),
                ),
                planned_step(
                    "fallback-example",
                    "Attempt a non-critical HTTP call",
                    Some("http_fetch"),
                    json!({"url": "https://example.com/noncritical"}),
                ),
                planned_step(
                    "finalize",
                    "Generate final report",
                    None,
                    json!({"prompt": "Summarize results"}),
                ),
            ],
            metadata: json!({}),
        })
    }

    async fn execute_step(
        &self,
        step: &Step,
        ctx: &mut AgentContext,
    ) -> Result<StepOutcome, AgentError> {
        if let Some(tool_name) = &step.tool {
            let output = self
                .tools
                .invoke(tool_name, step.args.clone(), &ctx.tool_permissions.allowed)
                .await
                .map_err(|e| AgentError::Tool(e.to_string()))?;
            return Ok(StepOutcome {
                step_id: step.id.clone(),
                output,
                observations: vec!["planned_action".into()],
                success: true,
                retries: 0,
                fallback_used: false,
                control_notes: vec!["plan+execute".into()],
            });
        }

        let prompt = step
            .args
            .get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or("Summarize plan");
        let response = self.model.generate(prompt).await;
        Ok(StepOutcome {
            step_id: step.id.clone(),
            output: json!({"message": response.content}),
            observations: vec!["planned_action".into()],
            success: true,
            retries: 0,
            fallback_used: false,
            control_notes: vec!["plan+execute".into()],
        })
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Running planning+execution example...");
    let tools = Arc::new(agent_examples::common::default_tools());
    let agent = PlanningExecutionAgent {
        tools,
        model: StubModel,
    };
    let mut ctx = base_context("plan-execute");
    let loop_ctrl: ControlLoop = deterministic_loop(4);
    let outcomes = loop_ctrl.run(&agent, &mut ctx).await?;
    for outcome in outcomes {
        println!("{} => {}", outcome.step_id, outcome.output);
    }
    Ok(())
}
