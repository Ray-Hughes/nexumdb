//! The NexumDB client library.
//!
//! One handle over an embedded database: ingest, search, traverse, inspect.
//! The CLI and the HTTP server are both thin wrappers over this type, which is
//! what keeps them from drifting apart — a capability the CLI has is one the
//! viewer has, because there is only one implementation of it.
//!
//! The spec proposed a text query language as a possible v1. This ships the
//! fluent pipeline API instead and leaves a DSL for later: the pipeline is the
//! thing a CLI and a UI can both build programmatically, and a text syntax
//! designed before the pipeline settles would have to be redesigned after.

pub mod config;

pub use config::{ClientConfig, OpenMode};

use nexum_core::model::Document;
use nexum_core::query::{NodeDetail, node_detail};
use nexum_core::{
    CompactionReport, Db, DbStats, Direction, EdgeType, Node, NodeId, NodeKind, Predicate, Query,
    QueryResult, ResultNode,
};
use nexum_embed::{Embedder, EmbedderConfig};
use nexum_ingest::{IngestConfig, IngestReport, Ingestor};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;

/// Failures the client surfaces.
#[derive(Debug, Error)]
pub enum ClientError {
    #[error(transparent)]
    Core(#[from] nexum_core::Error),

    #[error(transparent)]
    Embed(#[from] nexum_embed::EmbedError),

    #[error(transparent)]
    Ingest(#[from] nexum_ingest::IngestError),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, ClientError>;

/// An open NexumDB.
pub struct Nexum {
    db: Arc<Db>,
    embedder: Arc<dyn Embedder>,
    ingest_config: IngestConfig,
    embedder_config: EmbedderConfig,
}

impl Nexum {
    /// Create a new database at `path`.
    pub async fn create(path: impl AsRef<Path>, config: ClientConfig) -> Result<Self> {
        let db = Db::create(path, config.db.clone())?;
        Self::finish(db, config).await
    }

    /// Open an existing database.
    pub async fn open(path: impl AsRef<Path>, config: ClientConfig) -> Result<Self> {
        let db = Db::open(path)?;
        Self::finish(db, config).await
    }

    /// Open if present, create otherwise.
    pub async fn open_or_create(path: impl AsRef<Path>, config: ClientConfig) -> Result<Self> {
        let db = Db::open_or_create(path)?;
        Self::finish(db, config).await
    }

    async fn finish(db: Db, config: ClientConfig) -> Result<Self> {
        let embedder: Arc<dyn Embedder> = Arc::from(nexum_embed::build(&config.embedder).await?);
        Ok(Nexum {
            db: Arc::new(db),
            embedder,
            ingest_config: config.ingest,
            embedder_config: config.embedder,
        })
    }

    /// The underlying database, for callers that need the raw engine.
    pub fn db(&self) -> &Arc<Db> {
        &self.db
    }

    pub fn embedder(&self) -> &dyn Embedder {
        self.embedder.as_ref()
    }

    pub fn embedder_config(&self) -> &EmbedderConfig {
        &self.embedder_config
    }

    pub fn path(&self) -> &Path {
        self.db.path()
    }

    fn ingestor(&self) -> Ingestor {
        Ingestor::new(
            self.db.clone(),
            self.embedder.clone(),
            self.ingest_config.clone(),
        )
    }

    // ---- Ingestion -------------------------------------------------------

    /// Ingest raw text under a stable URI.
    pub async fn ingest_text(
        &self,
        source_uri: impl Into<String>,
        title: impl Into<String>,
        text: String,
    ) -> Result<IngestReport> {
        Ok(self.ingestor().ingest_text(source_uri, title, text).await?)
    }

    /// Ingest a file or directory tree.
    pub async fn ingest(
        &self,
        path: impl AsRef<Path>,
        recursive: bool,
    ) -> Result<Vec<IngestReport>> {
        Ok(self
            .ingestor()
            .ingest_path(path.as_ref(), recursive)
            .await?)
    }

    // ---- Retrieval -------------------------------------------------------

    /// Embed `query` and return the nearest chunks.
    ///
    /// Restricted to the current version of each document by default — the
    /// spec's "queries default to latest version only".
    pub async fn search(&self, query: &str, options: SearchOptions) -> Result<SearchResults> {
        let vector = self.embedder.embed_query(query).await?;
        self.search_vector(vector, options)
    }

    /// Search with a vector you already have.
    pub fn search_vector(&self, vector: Vec<f32>, options: SearchOptions) -> Result<SearchResults> {
        let mut pipeline = Query::new().search_with(
            vector,
            options.top_k,
            options.model.clone().or_else(|| {
                Some(nexum_core::vector_namespace(
                    self.embedder.model_id(),
                    self.embedder.dim(),
                ))
            }),
        );

        if let Some(predicate) = options.filter.clone() {
            pipeline = pipeline.filter(predicate);
        }
        if options.latest_only {
            pipeline = pipeline.latest();
        }
        if let Some(expand) = &options.expand {
            pipeline = pipeline.traverse_with(
                expand.edge_types.clone(),
                expand.max_hops,
                expand.direction,
                true,
            );
            if options.latest_only {
                pipeline = pipeline.latest();
            }
        }

        let result = self.db.query(&pipeline)?;
        Ok(SearchResults {
            query_model: self.embedder.model_id().to_string(),
            results: result.nodes,
            stats: result.stats,
        })
    }

    /// Run a query pipeline directly.
    pub fn query(&self, query: &Query) -> Result<QueryResult> {
        Ok(self.db.query(query)?)
    }

    /// Expand outward from a set of nodes.
    pub fn traverse(
        &self,
        start: impl IntoIterator<Item = NodeId>,
        edge_types: Vec<EdgeType>,
        max_hops: usize,
        direction: Direction,
    ) -> Result<QueryResult> {
        Ok(self.db.query(
            &Query::new()
                .seed(start)
                .traverse_with(edge_types, max_hops, direction, false),
        )?)
    }

    /// Fetch one node.
    pub fn get(&self, id: NodeId) -> Result<Option<Node>> {
        Ok(self.db.get_node(id)?)
    }

    /// Fetch a node with every edge around it.
    pub fn show(&self, id: NodeId) -> Result<Option<NodeDetail>> {
        Ok(node_detail(&self.db.read()?, id)?)
    }

    /// Immediate neighbours of a node.
    pub fn neighbors(
        &self,
        id: NodeId,
        edge_types: &[EdgeType],
        direction: Direction,
    ) -> Result<Vec<Node>> {
        Ok(self.db.neighbors(id, edge_types, direction)?)
    }

    /// Every version of a document, oldest first.
    pub fn history(&self, document_id: NodeId) -> Result<Vec<Document>> {
        Ok(self
            .db
            .history(document_id)?
            .into_iter()
            .filter_map(|n| n.as_document().ok().cloned())
            .collect())
    }

    /// Find a document by its source URI, newest version first.
    pub fn document_by_uri(&self, source_uri: &str) -> Result<Option<Document>> {
        let Some(id) = self.db.head_document(source_uri)? else {
            return Ok(None);
        };
        Ok(self
            .db
            .get_node(id)?
            .and_then(|n| n.as_document().ok().cloned()))
    }

    /// List documents, newest version only unless `include_superseded`.
    pub fn documents(&self, include_superseded: bool) -> Result<Vec<Document>> {
        let mut pipeline = Query::new().seed_kind(NodeKind::Document);
        if !include_superseded {
            pipeline = pipeline.latest();
        }
        Ok(self
            .db
            .query(&pipeline)?
            .nodes
            .into_iter()
            .filter_map(|n| n.node.as_document().ok().cloned())
            .collect())
    }

    /// Chunks belonging to a document, in reading order.
    pub fn chunks_of(&self, document_id: NodeId) -> Result<Vec<nexum_core::Chunk>> {
        let mut chunks: Vec<nexum_core::Chunk> = self
            .db
            .query(
                &Query::new()
                    .seed([document_id])
                    .traverse_with(vec![EdgeType::PartOf], 1, Direction::In, false)
                    .of_kind(NodeKind::Chunk),
            )?
            .nodes
            .into_iter()
            .filter_map(|n| n.node.as_chunk().ok().cloned())
            .collect();
        chunks.sort_by_key(|c| c.chunk_index);
        Ok(chunks)
    }

    // ---- Maintenance -----------------------------------------------------

    pub fn stats(&self) -> Result<DbStats> {
        Ok(self.db.stats()?)
    }

    pub fn flush(&self) -> Result<()> {
        Ok(self.db.flush()?)
    }

    /// Rebuild indexes, truncate the log, reclaim pages.
    pub fn compact(&mut self) -> Result<CompactionReport> {
        let db = Arc::get_mut(&mut self.db).ok_or_else(|| {
            ClientError::Other(
                "cannot compact while other handles to this database are open".into(),
            )
        })?;
        Ok(db.compact()?)
    }

    /// Stream every node and edge as JSON Lines.
    ///
    /// Includes superseded versions: an export that silently dropped history
    /// would not be a backup.
    pub fn export(&self, out: &mut dyn std::io::Write) -> Result<ExportSummary> {
        let view = self.db.read()?;
        let mut summary = ExportSummary::default();

        writeln!(
            out,
            "{}",
            serde_json::json!({
                "type": "header",
                "format_version": 1,
                "engine_version": nexum_core::VERSION,
                "exported_at": nexum_core::Timestamp::now().to_rfc3339(),
                "namespaces": view.namespaces()?,
            })
        )?;

        for kind in NodeKind::ALL {
            for id in view.ids_of_kind(kind)? {
                let Some(node) = view.get_node_including_deleted(id)? else {
                    continue;
                };
                let mut record = serde_json::to_value(&node).map_err(|e| {
                    ClientError::Other(format!("could not serialise node {id}: {e}"))
                })?;
                if let Some(object) = record.as_object_mut() {
                    object.insert("type".into(), serde_json::json!("node"));
                }
                writeln!(out, "{record}")?;
                summary.nodes += 1;
            }
        }

        view.scan_edges(|_, edge| {
            let mut record = serde_json::to_value(&edge).unwrap_or_default();
            if let Some(object) = record.as_object_mut() {
                object.insert("type".into(), serde_json::json!("edge"));
            }
            let _ = writeln!(out, "{record}");
            summary.edges += 1;
            Ok(())
        })?;

        // Vectors last and separately: they are the bulk of the bytes, and a
        // consumer that only wants the graph can stop reading before them.
        for (namespace, _) in view.namespaces()? {
            view.scan_vectors(&namespace, |id, vector| {
                let _ = writeln!(
                    out,
                    "{}",
                    serde_json::json!({
                        "type": "vector",
                        "namespace": namespace,
                        "node_id": id.to_string(),
                        "vector": vector,
                    })
                );
                summary.vectors += 1;
                Ok(())
            })?;
        }

        Ok(summary)
    }
}

/// What an export wrote.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ExportSummary {
    pub nodes: u64,
    pub edges: u64,
    pub vectors: u64,
}

/// Graph expansion applied after a search.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Expansion {
    #[serde(default)]
    pub edge_types: Vec<EdgeType>,
    #[serde(default = "default_hops")]
    pub max_hops: usize,
    #[serde(default)]
    pub direction: Direction,
}

fn default_hops() -> usize {
    1
}

impl Default for Expansion {
    fn default() -> Self {
        Expansion {
            edge_types: vec![EdgeType::Mentions, EdgeType::RelatesTo],
            max_hops: default_hops(),
            direction: Direction::Both,
        }
    }
}

/// Options for a search.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SearchOptions {
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Restrict to the current version of each document.
    #[serde(default = "default_latest_only")]
    pub latest_only: bool,
    /// Model or `model:dim` namespace to search. Defaults to the client's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Extra predicate applied to results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<Predicate>,
    /// Expand results across the graph.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expand: Option<Expansion>,
}

fn default_top_k() -> usize {
    10
}
fn default_latest_only() -> bool {
    true
}

impl Default for SearchOptions {
    fn default() -> Self {
        SearchOptions {
            top_k: default_top_k(),
            latest_only: default_latest_only(),
            model: None,
            filter: None,
            expand: None,
        }
    }
}

impl SearchOptions {
    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    pub fn include_superseded(mut self) -> Self {
        self.latest_only = false;
        self
    }

    pub fn filter(mut self, predicate: Predicate) -> Self {
        self.filter = Some(predicate);
        self
    }

    pub fn expand(mut self, expansion: Expansion) -> Self {
        self.expand = Some(expansion);
        self
    }
}

/// Search results, with the model that produced them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SearchResults {
    /// Recorded so a caller can tell which embedding space these scores live
    /// in — scores from different models are not comparable.
    pub query_model: String,
    pub results: Vec<ResultNode>,
    pub stats: nexum_core::QueryStats,
}

impl SearchResults {
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    pub fn len(&self) -> usize {
        self.results.len()
    }
}

/// Re-exports so callers need only depend on this crate.
pub mod prelude {
    pub use nexum_core::{
        Chunk, CompareOp, Direction, Document, Edge, EdgeType, Entity, Node, NodeId, NodeKind,
        PipelineRun, Predicate, PropertyValue, Query, QueryResult, ResultNode, Timestamp,
    };
    pub use nexum_embed::EmbedderConfig;
    pub use nexum_ingest::{ChunkerConfig, IngestConfig, IngestOutcome, IngestReport};

    pub use super::{ClientConfig, Expansion, Nexum, SearchOptions, SearchResults};
}
