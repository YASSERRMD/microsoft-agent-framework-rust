use agent_core::{Agent, AgentConfig, AgentContext, AgentState, Plan, Step, StepOutcome};
use agent_models::StubModel;
use agent_runtime::ControlLoop;
use agent_tools::builtins::{LogTool, MathTool, TimeTool};
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
            steps: vec![Step {
                id: "hello".into(),
                description: "log a greeting".into(),
                tool: Some("log".into()),
                args: json!({"message": "hello from agent"}),
                subtasks: vec![],
            }, Step {
                id: "add".into(),
                description: "compute a math expression".into(),
                tool: Some("math".into()),
                args: json!({"expression": "1+1"}),
                subtasks: vec![],
            }],
        })
    }

    async fn execute_step(&self, step: &Step, _ctx: &mut agent_core::AgentContext) -> Result<StepOutcome, agent_core::AgentError> {
        if let Some(tool_name) = &step.tool {
            if let Some(tool) = self.tools.get(tool_name) {
                let output = tool
                    .execute(step.args.clone())
                    .await
                    .map_err(|e| agent_core::AgentError::Tool(e.to_string()))?;
                return Ok(StepOutcome { step_id: step.id.clone(), output, observations: vec![], success: true });
            }
        }
        Ok(StepOutcome { step_id: step.id.clone(), output: json!({"note": "no-op"}), observations: vec![], success: true })
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

            let mut ctx = AgentContext {
                config: AgentConfig { name: "demo".into(), description: None, max_iterations: 4 },
                state: AgentState::default(),
                metadata: json!({}),
            };
            let agent = DemoAgent { model: StubModel, tools: Arc::new(registry) };
            let loop_ctrl = ControlLoop { max_iterations: 4, delay: std::time::Duration::from_millis(0) };
            let outcomes = loop_ctrl.run(&agent, &mut ctx).await?;
            for outcome in outcomes {
                info!(step = %outcome.step_id, output = %outcome.output, "step completed");
            }
        }
        Commands::Tools => {
            println!("Built-in tools: log, math, time, http_fetch");
        }
        Commands::Models => {
            println!("Models: stub, random_reasoner");
        }
    }
    Ok(())
}
