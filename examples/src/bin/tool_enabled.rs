use agent_core::{Agent, AgentContext, AgentError, Plan, Step, StepOutcome};
use agent_examples::common::{base_context, default_policies, deterministic_loop};
use agent_models::{LLMModel, StubModel};
use agent_runtime::ControlLoop;
use serde_json::json;
use std::fmt;
use std::sync::Arc;

struct ToolEnabledAgent {
    tools: Arc<agent_tools::ToolRegistry>,
    model: StubModel,
}

impl fmt::Debug for ToolEnabledAgent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolEnabledAgent").finish()
    }
}

#[async_trait::async_trait]
impl Agent for ToolEnabledAgent {
    async fn plan(&self, _ctx: &AgentContext) -> Result<Plan, AgentError> {
        Ok(Plan {
            goal: "Use several tools to respond to a query".into(),
            steps: vec![
                Step {
                    id: "time".into(),
                    description: "Check the current time".into(),
                    tool: Some("time".into()),
                    args: json!({}),
                    subtasks: vec![],
                    policies: default_policies(),
                    chain_of_thought: None,
                },
                Step {
                    id: "compute".into(),
                    description: "Compute a quick estimate".into(),
                    tool: Some("math".into()),
                    args: json!({"expression": "3*7"}),
                    subtasks: vec![],
                    policies: default_policies(),
                    chain_of_thought: None,
                },
                Step {
                    id: "respond".into(),
                    description: "Use model to craft the reply".into(),
                    tool: None,
                    args: json!({"prompt": "Summarize the tool outputs"}),
                    subtasks: vec![],
                    policies: default_policies(),
                    chain_of_thought: None,
                },
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
                observations: vec!["tool".into()],
                success: true,
                retries: 0,
                fallback_used: false,
                control_notes: vec!["tool-enabled".into()],
            });
        }

        let prompt = step
            .args
            .get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or("Summarize");
        let content = self.model.generate(prompt).await;
        Ok(StepOutcome {
            step_id: step.id.clone(),
            output: json!({"message": content.content}),
            observations: vec!["model".into()],
            success: true,
            retries: 0,
            fallback_used: false,
            control_notes: vec!["tool-enabled".into()],
        })
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Running tool-enabled example...");
    let tools = Arc::new(agent_examples::common::default_tools());
    let agent = ToolEnabledAgent {
        tools,
        model: StubModel,
    };
    let mut ctx = base_context("tool-enabled");
    let loop_ctrl: ControlLoop = deterministic_loop(3);
    let outcomes = loop_ctrl.run(&agent, &mut ctx).await?;
    for outcome in outcomes {
        println!("{} => {}", outcome.step_id, outcome.output);
    }
    Ok(())
}
