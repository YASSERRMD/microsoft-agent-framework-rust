use agent_core::{Agent, AgentContext, AgentError, Plan, Step, StepOutcome};
use agent_examples::common::{
    base_context, default_policies, deterministic_loop, shared_tools_arc,
};
use agent_models::{LLMModel, StubModel};
use agent_runtime::ControlLoop;
use serde_json::json;
use std::fmt;
use std::sync::Arc;

struct ChatbotAgent {
    model: StubModel,
    system_prompt: String,
    tools: Arc<agent_tools::ToolRegistry>,
}

impl fmt::Debug for ChatbotAgent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChatbotAgent")
            .field("system_prompt", &self.system_prompt)
            .finish()
    }
}

#[async_trait::async_trait]
impl Agent for ChatbotAgent {
    async fn plan(&self, _ctx: &AgentContext) -> Result<Plan, AgentError> {
        Ok(Plan {
            goal: "Hold a short conversation".into(),
            steps: vec![Step {
                id: "reply".into(),
                description: "Answer the greeting".into(),
                tool: None,
                args: json!({"user": "Hi there!"}),
                subtasks: vec![],
                policies: default_policies(),
                chain_of_thought: None,
            }],
            metadata: json!({"agent": self.system_prompt}),
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
                observations: vec!["tool_invocation".into()],
                success: true,
                retries: 0,
                fallback_used: false,
                control_notes: vec![],
            });
        }

        let prompt = format!(
            "{}\nUser: {}\nAssistant:",
            self.system_prompt,
            step.args
                .get("user")
                .and_then(|v| v.as_str())
                .unwrap_or("Hello")
        );
        let reply = self.model.generate(&prompt).await;
        Ok(StepOutcome {
            step_id: step.id.clone(),
            output: json!({
                "message": reply.content,
                "usage": reply.usage,
            }),
            observations: vec!["chat_response".into()],
            success: true,
            retries: 0,
            fallback_used: false,
            control_notes: vec!["chatbot".into()],
        })
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Running chatbot example...");
    let tools = shared_tools_arc();
    let agent = ChatbotAgent {
        model: StubModel,
        system_prompt: "You are a friendly Rust-based chatbot.".into(),
        tools,
    };
    let mut ctx = base_context("chatbot");
    let loop_ctrl: ControlLoop = deterministic_loop(1);
    let outcomes = loop_ctrl.run(&agent, &mut ctx).await?;
    for outcome in outcomes {
        println!("{} => {}", outcome.step_id, outcome.output);
    }
    Ok(())
}
