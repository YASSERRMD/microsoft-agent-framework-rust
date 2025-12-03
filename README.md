# Microsoft Agent Framework (Rust Edition)

This repository hosts the Rust implementation of the Microsoft Agent Framework—a high-performance, modular, and safety-first platform for building AI agents. The workspace is organized around the official pillars of the framework: stateful agents, planning, tools, models, evaluation, orchestration, streaming, telemetry, pluggable memory, safety hooks, and tool protocols.

## Workspace crates
- `agent-core` – Core agent definitions, lifecycle hooks, plans, and steps.
- `agent-runtime` – Step executor, control loop, and a lightweight message bus for multi-agent flows.
- `agent-tools` – Tool trait, deterministic registry, and built-in tools (time, math, logging, HTTP fetch).
- `agent-models` – LLM model abstractions, usage tracking, tool call metadata, and stub providers.
- `agent-memory` – Memory trait with in-memory and null backends.
- `agent-evals` – Evaluator traits and basic validators.
- `agent-telemetry` – Tracing, metrics, and audit helpers.
- `agent-cli` – Demo CLI that scaffolds projects, runs sample agents, lists tools/models, and validates tool schemas via `agent new`, `agent run`, `agent tools`, `agent models`, and `agent test` commands.

## Safety system
- Input validation pipelines, prompt filters, and guardrail LLM hooks that keep agents within policy.
- Tool sandboxing, per-tool access controllers, and RBAC metadata aligned with the `agno-rust` model.
- Rate limiters, cooldowns, and policy-based access rules to prevent abuse or runaway loops.
- Redaction rules, retry/fallback directives, and output policy validators to ensure compliant responses.

## Repository layout
- `crates/` – Individual framework crates (core, runtime, tools, models, memory, evals, telemetry, CLI).
- `examples/` – Ready-to-run agent templates.
- `docs/` – Design notes and scope references.

See [`docs/scope.md`](docs/scope.md) for the full development scope, including required features, performance goals, testing expectations, and extensibility targets.

## Examples
Run any of the pre-built example agents to explore different orchestration patterns:

```
cargo run -p agent-examples --bin chatbot
cargo run -p agent-examples --bin research
cargo run -p agent-examples --bin code
cargo run -p agent-examples --bin web_search
cargo run -p agent-examples --bin multi_agent
cargo run -p agent-examples --bin tool_enabled
cargo run -p agent-examples --bin react
cargo run -p agent-examples --bin planning_execution
```

Each example sets up its own `Agent` implementation, registers built-in tools, and drives the control loop so you can quickly try different orchestration patterns.
