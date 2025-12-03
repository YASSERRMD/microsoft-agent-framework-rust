use agent_core::{
    AgentConfig, AgentContext, AgentState, RetryPolicy, SafetyPolicy, StepPolicies, ToolPermissions,
};
use agent_runtime::{ControlLoop, ControlMode};
use agent_tools::{
    builtins::{FileTool, HttpFetchTool, LogTool, MathTool, TimeTool},
    ToolRegistry,
};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;

pub fn base_context(name: &str) -> AgentContext {
    AgentContext {
        config: AgentConfig {
            name: name.to_string(),
            description: None,
            max_iterations: 8,
            retry_policy: RetryPolicy::default(),
        },
        state: AgentState::default(),
        metadata: json!({}),
        memory: None,
        tool_permissions: ToolPermissions::default(),
    }
}

pub fn default_tools() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(TimeTool);
    registry.register(LogTool);
    registry.register(MathTool);
    registry.register(HttpFetchTool::new());
    let root = std::env::current_dir().expect("cwd");
    registry.register(FileTool::new(root));
    registry
}

pub fn deterministic_loop(iterations: usize) -> ControlLoop {
    ControlLoop {
        max_iterations: iterations,
        delay: Duration::from_millis(0),
        mode: ControlMode::Deterministic,
    }
}

pub fn reactive_loop(iterations: usize) -> ControlLoop {
    ControlLoop {
        max_iterations: iterations,
        delay: Duration::from_millis(0),
        mode: ControlMode::Reactive,
    }
}

pub fn default_policies() -> StepPolicies {
    StepPolicies {
        retry: RetryPolicy::default(),
        fallback: None,
        safety: SafetyPolicy {
            allow_tool_execution: true,
            redaction_rules: vec![],
            rbac_roles: vec![],
        },
    }
}

pub fn shared_tools_arc() -> Arc<ToolRegistry> {
    Arc::new(default_tools())
}
