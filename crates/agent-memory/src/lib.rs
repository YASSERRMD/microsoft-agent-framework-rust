use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("key not found: {0}")]
    NotFound(String),
    #[error("backend failure: {0}")]
    Backend(String),
}

#[async_trait]
pub trait MemoryStore: Send + Sync {
    async fn put(&self, key: &str, value: &Value) -> Result<(), MemoryError>;
    async fn get(&self, key: &str) -> Result<Option<Value>, MemoryError>;
    async fn search(&self, query: &str) -> Result<Vec<Value>, MemoryError>;
}

#[derive(Default)]
pub struct InMemoryStore {
    inner: tokio::sync::RwLock<HashMap<String, Value>>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self { inner: tokio::sync::RwLock::new(HashMap::new()) }
    }
}

#[async_trait]
impl MemoryStore for InMemoryStore {
    async fn put(&self, key: &str, value: &Value) -> Result<(), MemoryError> {
        self.inner.write().await.insert(key.to_string(), value.clone());
        Ok(())
    }

    async fn get(&self, key: &str) -> Result<Option<Value>, MemoryError> {
        Ok(self.inner.read().await.get(key).cloned())
    }

    async fn search(&self, query: &str) -> Result<Vec<Value>, MemoryError> {
        let values = self
            .inner
            .read()
            .await
            .iter()
            .filter(|(k, v)| k.contains(query) || v.to_string().contains(query))
            .map(|(_, v)| v.clone())
            .collect();
        Ok(values)
    }
}

pub struct NullStore;

#[async_trait]
impl MemoryStore for NullStore {
    async fn put(&self, _key: &str, _value: &Value) -> Result<(), MemoryError> {
        Ok(())
    }

    async fn get(&self, _key: &str) -> Result<Option<Value>, MemoryError> {
        Ok(None)
    }

    async fn search(&self, _query: &str) -> Result<Vec<Value>, MemoryError> {
        Ok(vec![])
    }
}
