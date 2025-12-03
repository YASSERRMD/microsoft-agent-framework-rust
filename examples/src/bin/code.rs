use agent_core::{Agent, AgentContext, AgentError, Plan, Step, StepOutcome};
use agent_examples::common::{base_context, default_policies, deterministic_loop};
use agent_models::{LLMModel, StubModel};
use agent_runtime::ControlLoop;
use serde_json::json;
use std::fmt;
use std::sync::Arc;

struct CodeAgent {
    model: StubModel,
    tools: Arc<agent_tools::ToolRegistry>,
}

impl fmt::Debug for CodeAgent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CodeAgent").finish()
    }
}

#[async_trait::async_trait]
impl Agent for CodeAgent {
    async fn plan(&self, _ctx: &AgentContext) -> Result<Plan, AgentError> {
        Ok(Plan {
            goal: "Produce a small Rust snippet".into(),
            steps: vec![
                Step {
                    id: "analyze".into(),
                    description: "Log the requirements".into(),
                    tool: Some("log".into()),
                    args: json!({"message": "Generate a hello_world function"}),
                    subtasks: vec![],
                    policies: default_policies(),
                    chain_of_thought: None,
                },
                Step {
                    id: "draft".into(),
                    description: "Draft the function body".into(),
                    tool: None,
                    args: json!({"goal": "fn hello_world() -> String {\"Hello\".into()}"}),
                    subtasks: vec![],
                    policies: default_policies(),
                    chain_of_thought: None,
                },
                Step {
                    id: "write".into(),
                    description: "Persist the snippet".into(),
                    tool: Some("file".into()),
                    args: json!({
                        "path": "target/generated/hello.rs",
                        "operation": "write",
                        "content": "pub fn hello_world() -> String { \"Hello from agent\".into() }",
                    }),
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
                control_notes: vec!["codegen".into()],
            });
        }

        let prompt = step
            .args
            .get("goal")
            .and_then(|v| v.as_str())
            .unwrap_or("Write code");
        let completion = self.model.generate(prompt).await;
        Ok(StepOutcome {
            step_id: step.id.clone(),
            output: json!({"draft": completion.content}),
            observations: vec!["drafted".into()],
            success: true,
            retries: 0,
            fallback_used: false,
            control_notes: vec!["codegen".into()],
        })
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Running codegen example...");
    let tools = Arc::new(agent_examples::common::default_tools());
    let agent = CodeAgent {
        model: StubModel,
        tools,
    };
    let mut ctx = base_context("code-agent");
    let loop_ctrl: ControlLoop = deterministic_loop(3);
    let outcomes = loop_ctrl.run(&agent, &mut ctx).await?;
    for outcome in outcomes {
        println!("{} => {}", outcome.step_id, outcome.output);
    }
    Ok(())
}
