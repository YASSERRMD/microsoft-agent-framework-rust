use agent_core::{Agent, AgentContext, AgentError, Plan, Step, StepOutcome};
use agent_examples::common::{base_context, default_policies, deterministic_loop};
use agent_runtime::ControlLoop;
use agent_tools::builtins::{SearchProvider, SearchResult, SearchTool};
use serde_json::json;
use std::fmt;
use std::sync::Arc;

struct StaticSearchProvider;

#[async_trait::async_trait]
impl SearchProvider for StaticSearchProvider {
    async fn search(
        &self,
        _query: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>, agent_tools::ToolError> {
        let items = vec![
            SearchResult {
                title: "Rust agent patterns".into(),
                url: "https://contoso.example/rust-agents".into(),
                snippet: "Overview of designing agents in Rust.".into(),
            },
            SearchResult {
                title: "Async orchestration".into(),
                url: "https://contoso.example/async".into(),
                snippet: "Practical tips for async control loops.".into(),
            },
        ];
        let take = limit.min(items.len());
        Ok(items.into_iter().take(take).collect())
    }
}

struct ResearchAgent {
    tools: Arc<agent_tools::ToolRegistry>,
}

impl fmt::Debug for ResearchAgent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResearchAgent").finish()
    }
}

#[async_trait::async_trait]
impl Agent for ResearchAgent {
    async fn plan(&self, _ctx: &AgentContext) -> Result<Plan, AgentError> {
        Ok(Plan {
            goal: "Research an assigned topic".into(),
            steps: vec![
                Step {
                    id: "search".into(),
                    description: "Find background references".into(),
                    tool: Some("search".into()),
                    args: json!({"query": "Rust agent frameworks", "limit": 3}),
                    subtasks: vec![],
                    policies: default_policies(),
                    chain_of_thought: None,
                },
                Step {
                    id: "synthesize".into(),
                    description: "Summarize the key points".into(),
                    tool: None,
                    args: json!({
                        "notes": [
                            "Rust excels at deterministic orchestration.",
                            "Tooling is strongly typed for safety.",
                            "Async runtimes make streaming simple.",
                        ]
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
                observations: vec!["search_completed".into()],
                success: true,
                retries: 0,
                fallback_used: false,
                control_notes: vec!["research".into()],
            });
        }

        let notes = step
            .args
            .get("notes")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let content = format!("Key findings: {} items", notes.len());
        Ok(StepOutcome {
            step_id: step.id.clone(),
            output: json!({"summary": content, "notes": notes}),
            observations: vec!["synthesis".into()],
            success: true,
            retries: 0,
            fallback_used: false,
            control_notes: vec!["research".into()],
        })
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Running research example...");
    let mut registry = agent_examples::common::default_tools();
    let search = SearchTool::new(Arc::new(StaticSearchProvider));
    registry.register(search);
    let tools = Arc::new(registry);

    let agent = ResearchAgent { tools };
    let mut ctx = base_context("research");
    let loop_ctrl: ControlLoop = deterministic_loop(2);
    let outcomes = loop_ctrl.run(&agent, &mut ctx).await?;
    for outcome in outcomes {
        println!("{} => {}", outcome.step_id, outcome.output);
    }
    Ok(())
}
