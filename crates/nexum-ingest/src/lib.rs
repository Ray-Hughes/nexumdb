//! Ingestion: chunk, embed, extract, and write with versioning and provenance.

pub mod chunk;
pub mod dedup;
pub mod extract;
pub mod pipeline;
pub mod source;

pub use chunk::{ChunkerConfig, TextChunk};
pub use dedup::DedupConfig;
pub use extract::{EntityExtractor, Extraction, NullExtractor, RuleExtractor};
pub use pipeline::{IngestConfig, IngestOutcome, IngestReport, Ingestor};
pub use source::SourceDocument;

use thiserror::Error;

/// Failures the ingestion pipeline can produce.
#[derive(Debug, Error)]
pub enum IngestError {
    #[error("{0}")]
    Config(String),

    #[error("cannot ingest {path}: {reason}")]
    Unsupported { path: String, reason: String },

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Core(#[from] nexum_core::Error),

    #[error(transparent)]
    Embed(#[from] nexum_embed::EmbedError),
}

pub type Result<T> = std::result::Result<T, IngestError>;
