use agent_core::{
    Agent, AgentConfig, AgentContext, AgentState, Plan, RetryPolicy, SafetyPolicy, Step,
    StepOutcome, StepPolicies, ToolPermissions,
};
use agent_models::StubModel;
use agent_runtime::{ControlLoop, ControlMode};
use agent_tools::builtins::{FileTool, LogTool, MathTool, TimeTool};
use agent_tools::ToolRegistry;
use clap::{Parser, Subcommand};
use serde_json::json;
use std::sync::Arc;
use tracing::info;

#[derive(Parser)]
#[command(name = "agent", about = "Microsoft Agent Framework (Rust Edition) CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Scaffold a new agent project
    New { name: String },
    /// Run a demo agent loop
    Run,
    /// List available tools
    Tools,
    /// List available models
    Models,
}

#[derive(Debug)]
struct DemoAgent {
    model: StubModel,
    tools: Arc<ToolRegistry>,
}

#[async_trait::async_trait]
impl Agent for DemoAgent {
    async fn plan(&self, _ctx: &agent_core::AgentContext) -> Result<Plan, agent_core::AgentError> {
        Ok(Plan {
            goal: "Say hello and compute a sum".into(),
            steps: vec![
                Step {
                    id: "hello".into(),
                    description: "log a greeting".into(),
                    tool: Some("log".into()),
                    args: json!({"message": "hello from agent"}),
                    subtasks: vec![],
                    policies: StepPolicies {
                        retry: RetryPolicy::default(),
                        fallback: None,
                        safety: SafetyPolicy {
                            allow_tool_execution: true,
                            redaction_rules: vec![],
                            rbac_roles: vec![],
                        },
                    },
                    chain_of_thought: None,
                },
                Step {
                    id: "add".into(),
                    description: "compute a math expression".into(),
                    tool: Some("math".into()),
                    args: json!({"expression": "1+1"}),
                    subtasks: vec![],
                    policies: StepPolicies {
                        retry: RetryPolicy::default(),
                        fallback: None,
                        safety: SafetyPolicy {
                            allow_tool_execution: true,
                            redaction_rules: vec![],
                            rbac_roles: vec![],
                        },
                    },
                    chain_of_thought: None,
                },
            ],
            metadata: json!({}),
        })
    }

    async fn execute_step(
        &self,
        step: &Step,
        _ctx: &mut agent_core::AgentContext,
    ) -> Result<StepOutcome, agent_core::AgentError> {
        if let Some(tool_name) = &step.tool {
            if let Some(tool) = self.tools.get(tool_name) {
                let output = tool
                    .execute(step.args.clone())
                    .await
                    .map_err(|e| agent_core::AgentError::Tool(e.to_string()))?;
                return Ok(StepOutcome {
                    step_id: step.id.clone(),
                    output,
                    observations: vec![],
                    success: true,
                    retries: 0,
                    fallback_used: false,
                    control_notes: vec![],
                });
            }
        }
        Ok(StepOutcome {
            step_id: step.id.clone(),
            output: json!({"note": "no-op"}),
            observations: vec![],
            success: true,
            retries: 0,
            fallback_used: false,
            control_notes: vec![],
        })
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    match cli.command {
        Commands::New { name } => {
            println!("Scaffolded new agent project: {name}");
        }
        Commands::Run => {
            let mut registry = ToolRegistry::new();
            registry.register(TimeTool);
            registry.register(LogTool);
            registry.register(MathTool);
            let pwd = std::env::current_dir()?;
            registry.register(FileTool::new(pwd));

            let mut ctx = AgentContext {
                config: AgentConfig {
                    name: "demo".into(),
                    description: None,
                    max_iterations: 4,
                    retry_policy: RetryPolicy::default(),
                },
                state: AgentState::default(),
                metadata: json!({}),
                memory: None,
                tool_permissions: ToolPermissions::default(),
            };
            let agent = DemoAgent {
                model: StubModel,
                tools: Arc::new(registry),
            };
            let loop_ctrl = ControlLoop {
                max_iterations: 4,
                delay: std::time::Duration::from_millis(0),
                mode: ControlMode::Deterministic,
            };
            let outcomes = loop_ctrl.run(&agent, &mut ctx).await?;
            for outcome in outcomes {
                info!(step = %outcome.step_id, output = %outcome.output, "step completed");
            }
        }
        Commands::Tools => {
            println!("Built-in tools: log, math, time, http_fetch, file");
        }
        Commands::Models => {
            println!("Models: stub, random_reasoner");
        }
    }
    Ok(())
}
