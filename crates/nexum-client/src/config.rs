//! Client configuration.

use nexum_core::DbConfig;
use nexum_embed::EmbedderConfig;
use nexum_ingest::IngestConfig;
use serde::{Deserialize, Serialize};

/// Everything needed to open a database and do useful work with it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ClientConfig {
    #[serde(default)]
    pub db: DbConfig,
    #[serde(default)]
    pub embedder: EmbedderConfig,
    #[serde(default)]
    pub ingest: IngestConfig,
}

impl ClientConfig {
    /// Use a specific embedder.
    pub fn with_embedder(mut self, embedder: EmbedderConfig) -> Self {
        self.embedder = embedder;
        self
    }

    /// Use a specific ingestion configuration.
    pub fn with_ingest(mut self, ingest: IngestConfig) -> Self {
        self.ingest = ingest;
        self
    }

    /// Read from a JSON file.
    pub fn from_file(path: &std::path::Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        serde_json::from_slice(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

/// Whether a database may be created if it is missing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum OpenMode {
    /// Fail if it does not exist.
    #[default]
    Existing,
    /// Create it if needed.
    OrCreate,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_roundtrips_through_json() {
        let config = ClientConfig::default();
        let json = serde_json::to_string_pretty(&config).unwrap();
        assert_eq!(config, serde_json::from_str(&json).unwrap(), "{json}");
    }

    #[test]
    fn an_empty_object_yields_defaults() {
        let config: ClientConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(config, ClientConfig::default());
    }
}
