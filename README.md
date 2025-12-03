# Microsoft Agent Framework (Rust Edition)

This repository tracks the Rust implementation of the Microsoft Agent Framework, focused on high-performance, modular, and safety-first AI agents. The project aligns with the official pillars: stateful agents, plans and steps, tools, models, evaluators, orchestration, streaming, telemetry, pluggable memory, safety hooks, and tool protocols.

The workspace now includes scaffolded crates for each pillar:

- `agent-core` – core agent definitions, lifecycle hooks, plans, and steps.
- `agent-runtime` – step executor, control loop, and a lightweight message bus for multi-agent flows.
- `agent-tools` – tool trait, deterministic registry, and built-in tools (time, math, logging, HTTP fetch).
- `agent-models` – LLM model abstractions, usage tracking, tool call metadata, and stub providers.
- `agent-memory` – memory trait with in-memory and null backends.
- `agent-evals` – evaluator traits and basic validators.
- `agent-telemetry` – tracing, metrics, and audit helpers.
- `agent-cli` – a demo CLI that scaffolds projects, runs sample agents, lists tools/models, and validates tool schemas via
  `agent new`, `agent run`, `agent tools`, `agent models`, and `agent test` commands.

## Safety System (Microsoft Parity)
- Input validation pipelines, prompt filters, and guardrail LLM hooks to keep agents within policy.
- Tool sandboxing, per-tool access controllers, and RBAC metadata aligned with the agno-rust model.
- Rate limiters, cooldowns, and policy-based access rules to prevent abuse or runaway loops.
- Redaction rules, retry/fallback directives, and output policy validators to keep responses compliant.

## Repository Layout
- `crates/` – Individual framework crates (core, runtime, tools, models, memory, evals, telemetry, CLI).
- `examples/` – Ready-to-run agent templates.
- `docs/` – Design notes and scope references.
- `tests/` – Unit and integration coverage across pillars.

See [`docs/scope.md`](docs/scope.md) for the full development scope, including required features, performance goals, testing expectations, and extensibility targets.
