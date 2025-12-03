# Rust Agent Framework – Scope Summary

This document captures the full development scope for the Rust edition of the Microsoft Agent Framework. The Rust version must uphold the same pillars as the Microsoft baseline while pushing for higher performance and modularity:

- Stateful agents with pluggable memory
- Plans and steps (including hidden chain-of-thought)
- Tools and tool protocols
- Model abstraction and providers
- Evaluators and safety hooks
- Orchestration/streaming runtime with telemetry
- CLI and developer tooling

## 1. Project Structure
```
agent-framework-rs/
├── crates/
│   ├── agent-core
│   ├── agent-runtime
│   ├── agent-tools
│   ├── agent-models
│   ├── agent-memory
│   ├── agent-evals
│   ├── agent-telemetry
│   └── agent-cli
├── examples/
├── docs/
└── tests/
```

## 2. Agent Core (agent-core)
- Core structs: `Agent`, `AgentConfig`, `AgentState`, `AgentContext`, `AgentError`.
- Planning: `Plan`, `Step`, `Subtask`, hidden Chain-of-Thought, `ExecutablePlan`.
- Lifecycle: `initialize()`, `think()`, `act()`, `observe()`, `reflect()` with memory and tool orchestration, retry/fallback policies.
- Traits: `Agent` trait with `plan` and `execute_step` methods.

## 3. Models Layer (agent-models)
- Abstraction trait `LLMModel` with `generate`, `stream`, and `supports_tools`.
- Providers: OpenAI, Azure OpenAI, Ollama, generic REST, embedding provider.
- Structures: `LLMResponse`, `CompletionChunk`, `ModelMetadata`, `ToolCallInfo`, `UsageMetrics`.
- Features: chat and reasoning models, token usage tracking, streaming, tool calling with JSON schema.

## 4. Tools Infrastructure (agent-tools)
- Tool trait for JSON schema-based definitions and async execution.
- Registry: global/agent-specific, deterministic ordering, access control metadata.
- Built-ins: HTTP fetch, sandboxed file IO, time/date, math, logging, search, and user-defined tools.
- Supports automatic discovery, routing, cooldowns, safety wrappers, and error propagation.

## 5. Runtime Engine (agent-runtime)
- Step executor and control loop (deterministic, reactive, procedural, reflection-enabled).
- Tool resolution, observability hooks, timeout management, token budget enforcement, safety interceptors.
- Multi-agent orchestration with message passing and shared/isolated memory.

## 6. Memory Subsystem (agent-memory)
- Stores: in-memory, SQLite, Postgres, Redis, vector DBs (Qdrant/Milvus/local HNSW).
- Trait: `MemoryStore` with `put`, `get`, and `search`.
- Capabilities: short/long-term memory, conversation transcripts, retrieval hooks.

## 7. Evaluators (agent-evals)
- Types: `StepEvaluator`, `OutputEvaluator`, `PlanEvaluator`.
- Use cases: hallucination and toxicity checks, JSON validity, tool call correctness, hidden chain-of-thought validation.

## 8. Telemetry + Observability (agent-telemetry)
- Instrumentation with `tracing`, `prometheus`, and `opentelemetry`.
- Metrics: LLM calls, tool usage, token counters, per-step summaries.
- Structured logs, spans, and audit log writer.

## 9. Safety System
- Input validation, prompt filtering, and policy-based request scrubbing.
- Tool sandboxing with per-tool access controllers, rate limiters, and RBAC aligned with agno-rust.
- Redaction rules, retry/fallback directives, and guardrail LLMs for both inputs and outputs.
- Output policy validators per tool with configurable blocking behavior.

## 10. CLI + Developer Tools (agent-cli)
- Commands: `agent new`, `agent run`, `agent test`, `agent tools`, `agent models` for scaffolding, execution, testing, and debugging.

## 11. Examples & Templates
- Ready-to-run agents: chatbot, research, code, web-search, multi-agent, tool-enabled, ReAct, planning + execution.

## 12. Documentation Scope
- Full API reference and Rustdoc, diagrams for plans/loops, tutorials for tools, models, memory, safety, and RBAC integration.

## 13. Testing Scope
- Unit, integration, and stress tests; model mocking with deterministic `StubModel`; tool call replay; memory persistence; multi-agent and async/concurrency coverage.

## 14. Performance Requirements
- Zero-copy serde JSON, tokio async runtime, high-performance vector search, minimal allocations, non-blocking model calls.

## 15. Extensibility Requirements
- Modular, plugin-friendly design supporting new tools, memory stores, agents, evaluators, and models through clear traits and registries.

## 16. Target Outcome
- Production-grade, high-performance Rust implementation that matches Microsoft Agent Framework capabilities while delivering higher throughput and stronger safety defaults.
- Multi-agent ready with consistent tool protocols, safety controls, and telemetry.
- Fully documented with examples, CLI workflows, and reference traits to make extension straightforward for new models, tools, memory stores, and evaluators.
