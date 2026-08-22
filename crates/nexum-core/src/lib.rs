//! # NexumDB core engine
//!
//! A graph-native vector database: vectors are a property type on graph nodes,
//! so similarity search and traversal compose in one query engine rather than
//! being stitched together across two stores. Document versioning and
//! provenance are part of the data model, not bookkeeping layered on top.
//!
//! ## Layout
//!
//! - [`model`] — nodes, edges, and the versioning rules
//! - [`store`] — the key-value tables (redb)
//! - [`wal`] — the write-ahead log and audit trail
//! - [`vector`] — HNSW indexes, namespaced per `(model, dim)`
//! - [`query`] — the search / traverse / filter pipeline
//! - [`db`] — the handle that ties them together
//!
//! ```no_run
//! use nexum_core::{Db, DbConfig, Query, Predicate};
//! use nexum_core::model::EdgeType;
//!
//! let db = Db::create("./mydb", DbConfig::default())?;
//! let results = db.query(
//!     &Query::new()
//!         .search(vec![0.1; 384], 20)
//!         .traverse(vec![EdgeType::Mentions, EdgeType::RelatesTo], 2)
//!         .filter(Predicate::LatestVersion),
//! )?;
//! # Ok::<(), nexum_core::Error>(())
//! ```

pub mod codec;
pub mod db;
pub mod error;
pub mod id;
pub mod keys;
pub mod model;
pub mod property;
pub mod query;
pub mod store;
pub mod vector;
pub mod wal;

pub use db::{CompactionReport, Db, DbConfig, DbStats, Transaction};
pub use error::{Error, Result};
pub use id::{ContentHash, NodeId, Timestamp};
pub use model::{
    Chunk, Direction, Document, Edge, EdgeClass, EdgeType, EmbeddingMeta, Entity, Metadata, Node,
    NodeKind, PipelineRun, vector_namespace,
};
pub use property::{Properties, PropertyValue};
pub use query::{
    CompareOp, NodeDetail, Predicate, Query, QueryResult, QueryStats, ResultNode, Stage,
};
pub use store::{NamespaceInfo, ReadView};
pub use vector::{Metric, ScoredNode};

/// The engine version, as reported by `nexum stats` and the server's health
/// endpoint.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
