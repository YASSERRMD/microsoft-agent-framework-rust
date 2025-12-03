use agent_core::{Agent, AgentContext, AgentError, Plan, Step, StepOutcome};
use agent_examples::common::{base_context, default_policies, deterministic_loop};
use agent_models::{LLMModel, StubModel};
use agent_runtime::ControlLoop;
use agent_tools::builtins::{SearchProvider, SearchResult, SearchTool};
use serde_json::json;
use std::fmt;
use std::sync::Arc;

struct DemoSearchProvider;

#[async_trait::async_trait]
impl SearchProvider for DemoSearchProvider {
    async fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>, agent_tools::ToolError> {
        Ok(vec![SearchResult {
            title: format!("Top hit for {query}"),
            url: "https://example.com".into(),
            snippet: "Mock search result demonstrating the search tool.".into(),
        }]
        .into_iter()
        .take(limit)
        .collect())
    }
}

struct WebSearchAgent {
    model: StubModel,
    tools: Arc<agent_tools::ToolRegistry>,
}

impl fmt::Debug for WebSearchAgent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WebSearchAgent").finish()
    }
}

#[async_trait::async_trait]
impl Agent for WebSearchAgent {
    async fn plan(&self, _ctx: &AgentContext) -> Result<Plan, AgentError> {
        Ok(Plan {
            goal: "Search the web and summarize a hit".into(),
            steps: vec![
                Step {
                    id: "lookup".into(),
                    description: "Search for Rust agent frameworks".into(),
                    tool: Some("search".into()),
                    args: json!({"query": "rust agent runtime", "limit": 1}),
                    subtasks: vec![],
                    policies: default_policies(),
                    chain_of_thought: None,
                },
                Step {
                    id: "summarize".into(),
                    description: "Summarize the first hit".into(),
                    tool: None,
                    args: json!({"title": "Rust agent runtime", "url": "https://example.com"}),
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
                observations: vec!["web_search".into()],
                success: true,
                retries: 0,
                fallback_used: false,
                control_notes: vec!["web".into()],
            });
        }

        let title = step
            .args
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("Result");
        let url = step
            .args
            .get("url")
            .and_then(|v| v.as_str())
            .unwrap_or("https://example.com");
        let summary = self
            .model
            .generate(&format!("Summarize {title} at {url}"))
            .await;
        Ok(StepOutcome {
            step_id: step.id.clone(),
            output: json!({"summary": summary.content, "source": url}),
            observations: vec!["summary".into()],
            success: true,
            retries: 0,
            fallback_used: false,
            control_notes: vec!["web".into()],
        })
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Running web-search example...");
    let mut registry = agent_examples::common::default_tools();
    registry.register(SearchTool::new(Arc::new(DemoSearchProvider)));
    let tools = Arc::new(registry);

    let agent = WebSearchAgent {
        model: StubModel,
        tools,
    };
    let mut ctx = base_context("web-search");
    let loop_ctrl: ControlLoop = deterministic_loop(2);
    let outcomes = loop_ctrl.run(&agent, &mut ctx).await?;
    for outcome in outcomes {
        println!("{} => {}", outcome.step_id, outcome.output);
    }
    Ok(())
}
