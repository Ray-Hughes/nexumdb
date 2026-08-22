//! Embedding providers.
//!
//! The engine does not care where a vector comes from — it records the model
//! name and dimension alongside it and moves on. This crate supplies the
//! providers and, critically, the model *identity* that gets written into
//! `PipelineRun` provenance: an embedding is only comparable to another
//! embedding from the same model at the same version.

pub mod hash;
pub mod ollama;
pub mod openai;

#[cfg(feature = "local")]
pub mod local;

mod provider;

pub use provider::{Embedder, EmbedderConfig, EmbeddingBatch, Usage, build};

/// The model name [`EmbedderConfig::Local`] defaults to.
pub fn local_default_model() -> &'static str {
    #[cfg(feature = "local")]
    {
        local::DEFAULT_MODEL
    }
    #[cfg(not(feature = "local"))]
    {
        "all-MiniLM-L6-v2"
    }
}

use thiserror::Error;

/// Failures an embedding provider can produce.
#[derive(Debug, Error)]
pub enum EmbedError {
    #[error("embedding provider `{provider}` is not configured: {reason}")]
    NotConfigured { provider: String, reason: String },

    #[error("embedding request to `{provider}` failed: {source}")]
    Transport {
        provider: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("embedding provider `{provider}` returned an error: {message}")]
    Api { provider: String, message: String },

    #[error("provider `{provider}` returned {got} vectors for {expected} inputs")]
    CountMismatch {
        provider: String,
        expected: usize,
        got: usize,
    },

    #[error("provider `{provider}` returned {got}-dimensional vectors, expected {expected}")]
    DimensionMismatch {
        provider: String,
        expected: usize,
        got: usize,
    },

    #[error("model `{0}` is not available")]
    UnknownModel(String),

    #[error("input at index {index} is too long: {tokens} tokens, limit is {limit}")]
    InputTooLong {
        index: usize,
        tokens: usize,
        limit: usize,
    },

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Other(String),
}

/// Result alias for embedding operations.
pub type Result<T> = std::result::Result<T, EmbedError>;
