use agent_core::{Agent, AgentContext, AgentError, Plan, Step, StepOutcome};
use agent_examples::common::{base_context, default_policies, deterministic_loop};
use agent_models::{LLMModel, StubModel};
use agent_runtime::ControlLoop;
use serde_json::json;
use std::fmt;
use std::sync::Arc;

struct ResearchPartner {
    model: StubModel,
}

impl ResearchPartner {
    async fn investigate(&self, topic: &str) -> String {
        self.model
            .generate(&format!("Collect findings about {topic}"))
            .await
            .content
    }
}

struct BuilderPartner {
    model: StubModel,
}

impl BuilderPartner {
    async fn propose(&self, idea: &str) -> String {
        self.model
            .generate(&format!("Draft an implementation for {idea}"))
            .await
            .content
    }
}

struct MultiAgentCoordinator {
    researcher: ResearchPartner,
    builder: BuilderPartner,
    tools: Arc<agent_tools::ToolRegistry>,
}

impl fmt::Debug for MultiAgentCoordinator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MultiAgentCoordinator").finish()
    }
}

#[async_trait::async_trait]
impl Agent for MultiAgentCoordinator {
    async fn plan(&self, _ctx: &AgentContext) -> Result<Plan, AgentError> {
        Ok(Plan {
            goal: "Coordinate two agents to ship a feature".into(),
            steps: vec![
                Step {
                    id: "kickoff".into(),
                    description: "Log the team kickoff".into(),
                    tool: Some("log".into()),
                    args: json!({"message": "Researcher + Builder online"}),
                    subtasks: vec![],
                    policies: default_policies(),
                    chain_of_thought: None,
                },
                Step {
                    id: "research".into(),
                    description: "Ask researcher for findings".into(),
                    tool: None,
                    args: json!({"topic": "Rust agent orchestration"}),
                    subtasks: vec![],
                    policies: default_policies(),
                    chain_of_thought: None,
                },
                Step {
                    id: "build".into(),
                    description: "Ask builder for a plan".into(),
                    tool: None,
                    args: json!({"idea": "streaming control loop"}),
                    subtasks: vec![],
                    policies: default_policies(),
                    chain_of_thought: None,
                },
                Step {
                    id: "debrief".into(),
                    description: "Log the combined result".into(),
                    tool: Some("log".into()),
                    args: json!({"message": "Team debrief published"}),
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
                observations: vec!["team_log".into()],
                success: true,
                retries: 0,
                fallback_used: false,
                control_notes: vec!["multi-agent".into()],
            });
        }

        let (message, control_note) = match step.id.as_str() {
            "research" => {
                let topic = step
                    .args
                    .get("topic")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown topic");
                (
                    self.researcher.investigate(topic).await,
                    "researcher".to_string(),
                )
            }
            "build" => {
                let idea = step
                    .args
                    .get("idea")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown idea");
                (self.builder.propose(idea).await, "builder".to_string())
            }
            _ => ("noop".into(), "noop".into()),
        };

        Ok(StepOutcome {
            step_id: step.id.clone(),
            output: json!({"message": message}),
            observations: vec!["collaboration".into()],
            success: true,
            retries: 0,
            fallback_used: false,
            control_notes: vec![control_note],
        })
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Running multi-agent example...");
    let tools = Arc::new(agent_examples::common::default_tools());
    let agent = MultiAgentCoordinator {
        researcher: ResearchPartner { model: StubModel },
        builder: BuilderPartner { model: StubModel },
        tools,
    };
    let mut ctx = base_context("multi-agent");
    let loop_ctrl: ControlLoop = deterministic_loop(4);
    let outcomes = loop_ctrl.run(&agent, &mut ctx).await?;
    for outcome in outcomes {
        println!("{} => {}", outcome.step_id, outcome.output);
    }
    Ok(())
}
