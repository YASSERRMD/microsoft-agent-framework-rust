use serde_json::Value;
use std::collections::HashMap;
use std::sync::RwLock;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("key not found: {0}")]
    NotFound(String),
    #[error("backend failure: {0}")]
    Backend(String),
    #[error("unsupported operation: {0}")]
    Unsupported(String),
}

/// Primary abstraction for storing and retrieving agent memory.
///
/// The trait intentionally mirrors the official framework contract
/// and is designed to work across in-memory, relational, cache, and
/// vector backends.
pub trait MemoryStore: Send + Sync + std::fmt::Debug {
    fn put(&self, key: &str, value: &Value) -> Result<(), MemoryError>;
    fn get(&self, key: &str) -> Result<Option<Value>, MemoryError>;
    fn search(&self, query: &str) -> Result<Vec<Value>, MemoryError>;
}

#[derive(Default, Debug)]
pub struct InMemoryStore {
    inner: RwLock<HashMap<String, Value>>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(HashMap::new()),
        }
    }
}

impl MemoryStore for InMemoryStore {
    fn put(&self, key: &str, value: &Value) -> Result<(), MemoryError> {
        self.inner
            .write()
            .map_err(|e| MemoryError::Backend(e.to_string()))?
            .insert(key.to_string(), value.clone());
        Ok(())
    }

    fn get(&self, key: &str) -> Result<Option<Value>, MemoryError> {
        Ok(self
            .inner
            .read()
            .map_err(|e| MemoryError::Backend(e.to_string()))?
            .get(key)
            .cloned())
    }

    fn search(&self, query: &str) -> Result<Vec<Value>, MemoryError> {
        let values = self
            .inner
            .read()
            .map_err(|e| MemoryError::Backend(e.to_string()))?
            .iter()
            .filter(|(k, v)| k.contains(query) || v.to_string().contains(query))
            .map(|(_, v)| v.clone())
            .collect();
        Ok(values)
    }
}

#[derive(Debug)]
pub struct NullStore;

impl MemoryStore for NullStore {
    fn put(&self, _key: &str, _value: &Value) -> Result<(), MemoryError> {
        Ok(())
    }

    fn get(&self, _key: &str) -> Result<Option<Value>, MemoryError> {
        Ok(None)
    }

    fn search(&self, _query: &str) -> Result<Vec<Value>, MemoryError> {
        Ok(vec![])
    }
}

#[derive(Debug, Clone)]
pub enum VectorBackend {
    Qdrant,
    Milvus,
    LocalHnsw,
}

#[derive(Debug)]
pub struct VectorStore {
    backend: VectorBackend,
    /// Minimal in-memory staging area until real vector DB integrations are wired in.
    buffer: RwLock<Vec<(String, Value)>>,
}

impl VectorStore {
    pub fn new(backend: VectorBackend) -> Self {
        Self {
            backend,
            buffer: RwLock::new(Vec::new()),
        }
    }
}

impl MemoryStore for VectorStore {
    fn put(&self, key: &str, value: &Value) -> Result<(), MemoryError> {
        self.buffer
            .write()
            .map_err(|e| MemoryError::Backend(e.to_string()))?
            .push((key.to_string(), value.clone()));
        Ok(())
    }

    fn get(&self, key: &str) -> Result<Option<Value>, MemoryError> {
        Ok(self
            .buffer
            .read()
            .map_err(|e| MemoryError::Backend(e.to_string()))?
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.clone()))
    }

    fn search(&self, query: &str) -> Result<Vec<Value>, MemoryError> {
        Ok(self
            .buffer
            .read()
            .map_err(|e| MemoryError::Backend(e.to_string()))?
            .iter()
            .filter(|(k, v)| k.contains(query) || v.to_string().contains(query))
            .map(|(_, v)| v.clone())
            .collect())
    }
}

#[derive(Debug)]
pub struct SqliteStore {
    connection_string: String,
    cache: RwLock<HashMap<String, Value>>,
}

impl SqliteStore {
    pub fn new<T: Into<String>>(connection_string: T) -> Self {
        Self {
            connection_string: connection_string.into(),
            cache: RwLock::new(HashMap::new()),
        }
    }
}

impl MemoryStore for SqliteStore {
    fn put(&self, key: &str, value: &Value) -> Result<(), MemoryError> {
        self.cache
            .write()
            .map_err(|e| MemoryError::Backend(e.to_string()))?
            .insert(key.to_string(), value.clone());
        Ok(())
    }

    fn get(&self, key: &str) -> Result<Option<Value>, MemoryError> {
        Ok(self
            .cache
            .read()
            .map_err(|e| MemoryError::Backend(e.to_string()))?
            .get(key)
            .cloned())
    }

    fn search(&self, query: &str) -> Result<Vec<Value>, MemoryError> {
        Ok(self
            .cache
            .read()
            .map_err(|e| MemoryError::Backend(e.to_string()))?
            .iter()
            .filter(|(k, v)| k.contains(query) || v.to_string().contains(query))
            .map(|(_, v)| v.clone())
            .collect())
    }
}

#[derive(Debug)]
pub struct PostgresStore {
    connection_string: String,
}

impl PostgresStore {
    pub fn new<T: Into<String>>(connection_string: T) -> Self {
        Self {
            connection_string: connection_string.into(),
        }
    }
}

impl MemoryStore for PostgresStore {
    fn put(&self, _key: &str, _value: &Value) -> Result<(), MemoryError> {
        Err(MemoryError::Unsupported(format!(
            "write not implemented for Postgres store ({})",
            self.connection_string
        )))
    }

    fn get(&self, _key: &str) -> Result<Option<Value>, MemoryError> {
        Err(MemoryError::Unsupported(format!(
            "read not implemented for Postgres store ({})",
            self.connection_string
        )))
    }

    fn search(&self, _query: &str) -> Result<Vec<Value>, MemoryError> {
        Err(MemoryError::Unsupported(format!(
            "search not implemented for Postgres store ({})",
            self.connection_string
        )))
    }
}

#[derive(Debug)]
pub struct RedisStore {
    connection_string: String,
    cache: RwLock<HashMap<String, Value>>,
}

impl RedisStore {
    pub fn new<T: Into<String>>(connection_string: T) -> Self {
        Self {
            connection_string: connection_string.into(),
            cache: RwLock::new(HashMap::new()),
        }
    }
}

impl MemoryStore for RedisStore {
    fn put(&self, key: &str, value: &Value) -> Result<(), MemoryError> {
        self.cache
            .write()
            .map_err(|e| MemoryError::Backend(e.to_string()))?
            .insert(key.to_string(), value.clone());
        Ok(())
    }

    fn get(&self, key: &str) -> Result<Option<Value>, MemoryError> {
        Ok(self
            .cache
            .read()
            .map_err(|e| MemoryError::Backend(e.to_string()))?
            .get(key)
            .cloned())
    }

    fn search(&self, query: &str) -> Result<Vec<Value>, MemoryError> {
        Ok(self
            .cache
            .read()
            .map_err(|e| MemoryError::Backend(e.to_string()))?
            .iter()
            .filter(|(k, v)| k.contains(query) || v.to_string().contains(query))
            .map(|(_, v)| v.clone())
            .collect())
    }
}
