use agent_core::{Agent, AgentContext, AgentError, Plan, Step, StepOutcome};
use agent_examples::common::{base_context, default_policies, reactive_loop};
use agent_models::{LLMModel, StubModel};
use agent_runtime::ControlLoop;
use serde_json::json;
use std::fmt;
use std::sync::Arc;

struct ReactAgent {
    tools: Arc<agent_tools::ToolRegistry>,
    model: StubModel,
}

impl fmt::Debug for ReactAgent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReactAgent").finish()
    }
}

#[async_trait::async_trait]
impl Agent for ReactAgent {
    async fn plan(&self, ctx: &AgentContext) -> Result<Plan, AgentError> {
        // Alternates between reflection and action based on iteration.
        let step = match ctx.state.iteration {
            0 => Step {
                id: "thought".into(),
                description: "Reflect on the task".into(),
                tool: None,
                args: json!({"prompt": "Think about how to answer"}),
                subtasks: vec![],
                policies: default_policies(),
                chain_of_thought: Some({
                    let mut cot = agent_core::ChainOfThought::new();
                    cot.push("Need context before acting");
                    cot
                }),
            },
            1 => Step {
                id: "action".into(),
                description: "Take an action using tools".into(),
                tool: Some("time".into()),
                args: json!({}),
                subtasks: vec![],
                policies: default_policies(),
                chain_of_thought: None,
            },
            _ => Step {
                id: "answer".into(),
                description: "Provide the final answer".into(),
                tool: None,
                args: json!({"prompt": "Summarize observation and action"}),
                subtasks: vec![],
                policies: default_policies(),
                chain_of_thought: None,
            },
        };

        Ok(Plan {
            goal: "Demonstrate ReAct-style iteration".into(),
            steps: vec![step],
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
                observations: vec!["acted".into()],
                success: true,
                retries: 0,
                fallback_used: false,
                control_notes: vec!["react".into()],
            });
        }

        let prompt = step
            .args
            .get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or("Reflect");
        let response = self.model.generate(prompt).await;
        Ok(StepOutcome {
            step_id: step.id.clone(),
            output: json!({"message": response.content}),
            observations: vec!["reflected".into()],
            success: true,
            retries: 0,
            fallback_used: false,
            control_notes: vec!["react".into()],
        })
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Running ReAct example...");
    let tools = Arc::new(agent_examples::common::default_tools());
    let agent = ReactAgent {
        tools,
        model: StubModel,
    };
    let mut ctx = base_context("react");
    let loop_ctrl: ControlLoop = reactive_loop(3);
    let outcomes = loop_ctrl.run(&agent, &mut ctx).await?;
    for outcome in outcomes {
        println!("{} => {}", outcome.step_id, outcome.output);
    }
    Ok(())
}
