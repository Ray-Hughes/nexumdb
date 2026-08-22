//! Error types for the NexumDB core engine.

use crate::id::NodeId;

/// Result alias used throughout the core engine.
pub type Result<T> = std::result::Result<T, Error>;

/// Every failure mode the storage engine can surface.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("storage error: {0}")]
    Storage(String),

    #[error("codec error: {0}")]
    Codec(String),

    #[error("node {0} not found")]
    NodeNotFound(NodeId),

    #[error("node {id} is a {actual}, expected a {expected}")]
    NodeKindMismatch {
        id: NodeId,
        actual: &'static str,
        expected: &'static str,
    },

    #[error("edge {from} -[{edge_type}]-> {to} already exists")]
    DuplicateEdge {
        from: NodeId,
        to: NodeId,
        edge_type: &'static str,
    },

    #[error("vector dimension mismatch in namespace `{namespace}`: expected {expected}, got {got}")]
    DimensionMismatch {
        namespace: String,
        expected: usize,
        got: usize,
    },

    #[error("no vector index for namespace `{0}`")]
    UnknownNamespace(String),

    #[error("write-ahead log corrupted at offset {offset}: {reason}")]
    WalCorrupt { offset: u64, reason: String },

    #[error(
        "database at `{path}` was created by an incompatible format version {found} (this build supports {supported})"
    )]
    IncompatibleFormat {
        path: String,
        found: u32,
        supported: u32,
    },

    #[error("database is locked by another writer")]
    WriterLocked,

    #[error("invalid identifier `{0}`")]
    InvalidId(String),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

impl Error {
    pub(crate) fn codec<E: std::fmt::Display>(e: E) -> Self {
        Error::Codec(e.to_string())
    }
}

macro_rules! from_redb {
    ($($t:ty),* $(,)?) => {
        $(impl From<$t> for Error {
            fn from(e: $t) -> Self { Error::Storage(e.to_string()) }
        })*
    };
}

from_redb!(
    redb::Error,
    redb::DatabaseError,
    redb::TransactionError,
    redb::TableError,
    redb::StorageError,
    redb::CommitError,
    redb::CompactionError,
);
