use chrono::Utc;
use opentelemetry::trace::{Span, TraceContextExt, Tracer};
use opentelemetry::Context;
use prometheus::{
    Encoder, HistogramOpts, HistogramVec, IntCounterVec, Opts, Registry, TextEncoder,
};
use serde_json::Value;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use tracing::{event, Level};

pub struct Telemetry {
    tracer: opentelemetry::sdk::trace::Tracer,
    registry: Registry,
    llm_calls: IntCounterVec,
    tool_calls: IntCounterVec,
    llm_input_tokens: IntCounterVec,
    llm_output_tokens: IntCounterVec,
    llm_latency_ms: HistogramVec,
    tool_latency_ms: HistogramVec,
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
        let llm_input_tokens = IntCounterVec::new(
            Opts::new("llm_input_tokens", "Tokens sent to LLMs"),
            &["model"],
        )
        .expect("metric");
        let llm_output_tokens = IntCounterVec::new(
            Opts::new("llm_output_tokens", "Tokens returned by LLMs"),
            &["model"],
        )
        .expect("metric");
        let llm_latency_ms = HistogramVec::new(
            HistogramOpts::new(
                "llm_call_latency_ms",
                "LLM call latency distribution (milliseconds)",
            ),
            &["model"],
        )
        .expect("metric");
        let tool_latency_ms = HistogramVec::new(
            HistogramOpts::new(
                "tool_call_latency_ms",
                "Tool call latency distribution (milliseconds)",
            ),
            &["tool"],
        )
        .expect("metric");
        registry.register(Box::new(llm_calls.clone())).unwrap();
        registry.register(Box::new(tool_calls.clone())).unwrap();
        registry
            .register(Box::new(llm_input_tokens.clone()))
            .unwrap();
        registry
            .register(Box::new(llm_output_tokens.clone()))
            .unwrap();
        registry.register(Box::new(llm_latency_ms.clone())).unwrap();
        registry
            .register(Box::new(tool_latency_ms.clone()))
            .unwrap();

        Self {
            tracer,
            registry,
            llm_calls,
            tool_calls,
            llm_input_tokens,
            llm_output_tokens,
            llm_latency_ms,
            tool_latency_ms,
        }
    }

    pub fn record_llm_call(
        &self,
        model: &str,
        input_tokens: u64,
        output_tokens: u64,
        duration_ms: Option<f64>,
    ) {
        self.llm_calls.with_label_values(&[model]).inc();
        self.llm_input_tokens
            .with_label_values(&[model])
            .inc_by(input_tokens);
        self.llm_output_tokens
            .with_label_values(&[model])
            .inc_by(output_tokens);
        if let Some(value) = duration_ms {
            self.llm_latency_ms
                .with_label_values(&[model])
                .observe(value);
        }
        event!(
            Level::INFO,
            %model,
            input_tokens,
            output_tokens,
            duration_ms = duration_ms.unwrap_or_default(),
            "llm call recorded"
        );
    }

    pub fn record_tool_call(&self, tool: &str, duration_ms: Option<f64>) {
        self.tool_calls.with_label_values(&[tool]).inc();
        if let Some(value) = duration_ms {
            self.tool_latency_ms
                .with_label_values(&[tool])
                .observe(value);
        }
        event!(Level::INFO, %tool, duration_ms = duration_ms.unwrap_or_default(), "tool call recorded");
    }

    pub fn log_tool_step(&self, tool: &str, step: &str, summary: &str, payload: Option<&Value>) {
        if let Some(payload) = payload {
            event!(
                Level::INFO,
                %tool,
                %step,
                summary = %summary,
                payload = %payload,
                "tool step"
            );
        } else {
            event!(Level::INFO, %tool, %step, summary = %summary, "tool step");
        }
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

    pub fn record_step_summary(
        &self,
        step_name: &str,
        summary: &str,
        status: &str,
        metadata: Option<&Value>,
    ) {
        if let Some(metadata) = metadata {
            event!(
                Level::INFO,
                %step_name,
                %status,
                summary = %summary,
                metadata = %metadata,
                "step summary"
            );
        } else {
            event!(Level::INFO, %step_name, %status, summary = %summary, "step summary");
        }
    }
}

pub struct AuditLogWriter {
    file: Mutex<std::fs::File>,
}

impl AuditLogWriter {
    pub fn new(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            file: Mutex::new(file),
        })
    }

    pub fn write_event(&self, event_name: &str, payload: &Value) -> std::io::Result<()> {
        let mut file = self.file.lock().expect("audit file poisoned");
        let timestamp = Utc::now().to_rfc3339();
        let record = serde_json::json!({
            "timestamp": timestamp,
            "event_name": event_name,
            "payload": payload,
        });
        writeln!(file, "{}", record.to_string())
    }

    pub fn flush(&self) -> std::io::Result<()> {
        let mut file = self.file.lock().expect("audit file poisoned");
        file.flush()
    }
}
