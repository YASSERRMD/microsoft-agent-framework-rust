# Agent Examples

This package hosts ready-to-run binaries that demonstrate different agent profiles using the Microsoft Agent Framework (Rust Edition).

Run any example with Cargo:

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

Each binary configures a simple `Agent` implementation, sets up built-in tools, and drives the control loop to completion.
