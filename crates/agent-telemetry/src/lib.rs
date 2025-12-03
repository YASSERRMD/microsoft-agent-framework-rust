use opentelemetry::trace::{Span, TraceContextExt, Tracer};
use opentelemetry::Context;
use prometheus::{Encoder, IntCounterVec, Opts, Registry, TextEncoder};
use serde_json::Value;
use tracing::{event, Level};

pub struct Telemetry {
    tracer: opentelemetry::sdk::trace::Tracer,
    registry: Registry,
    llm_calls: IntCounterVec,
    tool_calls: IntCounterVec,
}

impl Telemetry {
    pub fn new() -> Self {
        let tracer = opentelemetry::sdk::trace::TracerProvider::builder()
            .build()
            .versioned_tracer("agent-framework", Some(env!("CARGO_PKG_VERSION")), None);
        let registry = Registry::new();
        let llm_calls = IntCounterVec::new(Opts::new("llm_calls", "LLM call count"), &["model"])
            .expect("metric");
        let tool_calls = IntCounterVec::new(Opts::new("tool_calls", "Tool call count"), &["tool"])
            .expect("metric");
        registry.register(Box::new(llm_calls.clone())).unwrap();
        registry.register(Box::new(tool_calls.clone())).unwrap();

        Self {
            tracer,
            registry,
            llm_calls,
            tool_calls,
        }
    }

    pub fn record_llm_call(&self, model: &str) {
        self.llm_calls.with_label_values(&[model]).inc();
        event!(Level::INFO, %model, "llm call recorded");
    }

    pub fn record_tool_call(&self, tool: &str) {
        self.tool_calls.with_label_values(&[tool]).inc();
        event!(Level::INFO, %tool, "tool call recorded");
    }

    pub fn start_span(&self, name: &str) -> (Context, Span) {
        let span = self.tracer.start(name);
        let cx = Context::current_with_span(span.clone());
        (cx, span)
    }

    pub fn export_metrics(&self) -> String {
        let mut buffer = Vec::new();
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        encoder.encode(&metric_families, &mut buffer).unwrap();
        String::from_utf8(buffer).unwrap_or_default()
    }

    pub fn audit(&self, event_name: &str, payload: &Value) {
        event!(Level::INFO, %event_name, payload = %payload, "audit event");
    }
}
