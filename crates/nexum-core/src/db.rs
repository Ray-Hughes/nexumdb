//! The database handle.
//!
//! Ties the key-value store, the write-ahead log, and the vector indexes into
//! one object with a single write path. Concurrency is single-writer /
//! multi-reader, as the spec resolves: writes serialise on one lock, reads take
//! lock-free snapshots.
//!
//! Commit ordering is: append to the log and fsync, then apply to the tables,
//! then publish to the in-memory indexes. A crash between any two steps is
//! recovered by replaying the log past the watermark the tables recorded.

use crate::error::{Error, Result};
use crate::id::{NodeId, Timestamp};
use crate::model::{Direction, Edge, EdgeType, Node, NodeKind};
use crate::query::{NodeDetail, Query, QueryResult, node_detail, version_chain};
use crate::store::{NamespaceInfo, ReadView, Store};
use crate::vector::{HnswParams, Metric, VectorIndexSet};
use crate::wal::{Lsn, Wal, WalOp};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const STORE_FILE: &str = "nexum.redb";
const WAL_FILE: &str = "wal.log";
const CONFIG_FILE: &str = "config.json";
const LOCK_FILE: &str = "LOCK";

/// Meta key tracking how current the persisted HNSW graphs are.
///
/// Public so that recovery tooling — and the recovery tests — can inspect or
/// invalidate the snapshot without reaching into private state.
pub const META_INDEX_SNAPSHOT_LSN: &str = "index_snapshot_lsn";

/// Persistent database settings, written to `config.json` at creation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DbConfig {
    /// Distance metric for every vector index.
    #[serde(default)]
    pub metric: Metric,
    /// HNSW connectivity.
    #[serde(default = "default_m")]
    pub m: usize,
    #[serde(default = "default_ef_construction")]
    pub ef_construction: usize,
    #[serde(default = "default_ef_search")]
    pub ef_search: usize,
    /// Flush HNSW graphs to disk every N commits. Graphs can always be
    /// rebuilt from the vector table, so this trades startup time for commit
    /// throughput rather than durability.
    #[serde(default = "default_snapshot_interval")]
    pub index_snapshot_interval: u64,
    /// Default embedding model recorded for this database, if one was chosen
    /// at creation. Purely informational for the engine; the ingest pipeline
    /// reads it to pick a default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_embedding_model: Option<String>,
}

fn default_m() -> usize {
    16
}
fn default_ef_construction() -> usize {
    200
}
fn default_ef_search() -> usize {
    64
}
fn default_snapshot_interval() -> u64 {
    64
}

impl Default for DbConfig {
    fn default() -> Self {
        DbConfig {
            metric: Metric::default(),
            m: default_m(),
            ef_construction: default_ef_construction(),
            ef_search: default_ef_search(),
            index_snapshot_interval: default_snapshot_interval(),
            default_embedding_model: None,
        }
    }
}

impl DbConfig {
    fn hnsw_params(&self) -> HnswParams {
        HnswParams {
            m: self.m,
            m0: self.m * 2,
            ef_construction: self.ef_construction,
            ef_search: self.ef_search,
            metric: self.metric,
        }
    }
}

/// A summary of what a database contains.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DbStats {
    pub path: String,
    pub created_at: Timestamp,
    pub documents: u64,
    /// Documents with no superseding version.
    pub latest_documents: u64,
    pub chunks: u64,
    pub entities: u64,
    pub pipeline_runs: u64,
    pub edges: u64,
    pub edges_by_type: BTreeMap<String, u64>,
    pub tombstones: u64,
    pub namespaces: BTreeMap<String, NamespaceInfo>,
    pub store_bytes: u64,
    pub wal_bytes: u64,
    pub applied_lsn: Lsn,
}

/// An open database.
pub struct Db {
    path: PathBuf,
    config: DbConfig,
    store: Store,
    /// Serialises writers. The single-writer model lives here.
    wal: Mutex<Wal>,
    indexes: RwLock<VectorIndexSet>,
    commits_since_snapshot: Mutex<u64>,
    /// Held open for as long as the database is, to keep a second writer out.
    _lock: LockFile,
}

impl Db {
    /// Create a new database directory. Fails if one already exists there.
    pub fn create(path: impl AsRef<Path>, config: DbConfig) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if path.join(CONFIG_FILE).exists() {
            return Err(Error::InvalidArgument(format!(
                "a database already exists at {}",
                path.display()
            )));
        }
        std::fs::create_dir_all(&path)?;
        std::fs::write(
            path.join(CONFIG_FILE),
            serde_json::to_vec_pretty(&config).map_err(Error::codec)?,
        )?;
        Self::open_inner(path, config)
    }

    /// Open an existing database.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let config_path = path.join(CONFIG_FILE);
        if !config_path.exists() {
            return Err(Error::InvalidArgument(format!(
                "no NexumDB database at {} — run `nexum init` first",
                path.display()
            )));
        }
        let config: DbConfig =
            serde_json::from_slice(&std::fs::read(&config_path)?).map_err(Error::codec)?;
        Self::open_inner(path, config)
    }

    /// Open an existing database, or create one with default settings.
    pub fn open_or_create(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if path.join(CONFIG_FILE).exists() {
            Self::open(path)
        } else {
            Self::create(path, DbConfig::default())
        }
    }

    fn open_inner(path: PathBuf, config: DbConfig) -> Result<Self> {
        let lock = LockFile::acquire(path.join(LOCK_FILE))?;
        let store = Store::open(path.join(STORE_FILE))?;
        let mut wal = Wal::open(path.join(WAL_FILE))?;

        let applied = store.read()?.applied_lsn()?;
        let pending = wal.replay_from(applied)?;
        if !pending.is_empty() {
            tracing::info!(
                count = pending.len(),
                from_lsn = applied,
                "replaying write-ahead log"
            );
            let batch = store.write()?;
            let mut highest = applied;
            for record in &pending {
                apply_to_store(&batch, &record.op, record.lsn)?;
                highest = highest.max(record.lsn);
            }
            batch.set_applied_lsn(highest)?;
            // Any persisted graph predates this replay, so force a rebuild.
            batch.set_meta_u64(META_INDEX_SNAPSHOT_LSN, 0)?;
            batch.commit()?;
            wal.sync()?;
        }

        // A graph snapshot older than the applied LSN is missing writes.
        // Vectors are authoritative, so discard the stale graph and rebuild.
        let view = store.read()?;
        let applied = view.applied_lsn()?;
        let snapshot_lsn = view.meta_u64(META_INDEX_SNAPSHOT_LSN)?;
        let indexes = if snapshot_lsn == applied {
            VectorIndexSet::load(&store, config.hnsw_params())?
        } else {
            drop(view);
            tracing::info!(
                snapshot_lsn,
                applied,
                "vector index snapshot is stale; rebuilding from stored vectors"
            );
            rebuild_indexes(&store, config.hnsw_params())?
        };

        Ok(Db {
            path,
            config,
            store,
            wal: Mutex::new(wal),
            indexes: RwLock::new(indexes),
            commits_since_snapshot: Mutex::new(0),
            _lock: lock,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn config(&self) -> &DbConfig {
        &self.config
    }

    /// A read snapshot of the tables.
    pub fn read(&self) -> Result<ReadView> {
        self.store.read()
    }

    /// Begin a write transaction. Blocks until any other writer finishes.
    pub fn begin(&self) -> Transaction<'_> {
        Transaction {
            db: self,
            ops: Vec::new(),
        }
    }

    /// Fetch one node.
    pub fn get_node(&self, id: NodeId) -> Result<Option<Node>> {
        self.read()?.get_node(id)
    }

    /// Fetch a node with its edges.
    pub fn node_detail(&self, id: NodeId) -> Result<Option<NodeDetail>> {
        node_detail(&self.read()?, id)
    }

    /// Every version of a document, oldest first.
    pub fn history(&self, document_id: NodeId) -> Result<Vec<Node>> {
        version_chain(&self.read()?, document_id)
    }

    /// Neighbours of a node.
    pub fn neighbors(
        &self,
        id: NodeId,
        edge_types: &[EdgeType],
        direction: Direction,
    ) -> Result<Vec<Node>> {
        let view = self.read()?;
        let mut out = Vec::new();
        for neighbor in view.neighbors(id, edge_types, direction)? {
            if let Some(node) = view.get_node(neighbor.target)? {
                out.push(node);
            }
        }
        Ok(out)
    }

    /// Run a query pipeline.
    pub fn query(&self, query: &Query) -> Result<QueryResult> {
        let view = self.read()?;
        let indexes = self.indexes.read();
        query.execute(&view, &indexes)
    }

    /// Vector namespaces present in this database.
    pub fn namespaces(&self) -> Vec<String> {
        self.indexes
            .read()
            .namespaces()
            .into_iter()
            .map(str::to_string)
            .collect()
    }

    /// Resolve a model name to a full `model:dim` namespace.
    pub fn resolve_namespace(&self, model: Option<&str>) -> Result<String> {
        self.indexes.read().resolve(model)
    }

    /// The newest document ingested from a source URI.
    pub fn head_document(&self, source_uri: &str) -> Result<Option<NodeId>> {
        self.read()?.head_document(source_uri)
    }

    /// Summary counts for `nexum stats`.
    pub fn stats(&self) -> Result<DbStats> {
        let view = self.read()?;

        let mut edges_by_type: BTreeMap<String, u64> = BTreeMap::new();
        view.scan_edges(|_, edge| {
            *edges_by_type
                .entry(edge.edge_type.as_str().to_string())
                .or_insert(0) += 1;
            Ok(())
        })?;

        let document_ids = view.ids_of_kind(NodeKind::Document)?;
        let mut latest = 0u64;
        for id in &document_ids {
            if view
                .neighbors(*id, &[EdgeType::Supersedes], Direction::In)?
                .is_empty()
            {
                latest += 1;
            }
        }

        Ok(DbStats {
            path: self.path.display().to_string(),
            created_at: view.created_at()?,
            documents: document_ids.len() as u64,
            latest_documents: latest,
            chunks: view.count_of_kind(NodeKind::Chunk)?,
            entities: view.count_of_kind(NodeKind::Entity)?,
            pipeline_runs: view.count_of_kind(NodeKind::PipelineRun)?,
            edges: view.edge_count()?,
            edges_by_type,
            tombstones: view.tombstone_count()?,
            namespaces: view.namespaces()?,
            store_bytes: file_size(&self.path.join(STORE_FILE)),
            wal_bytes: file_size(&self.path.join(WAL_FILE)),
            applied_lsn: view.applied_lsn()?,
        })
    }

    /// Persist the vector graphs and sync the log.
    ///
    /// Called automatically on close and every `index_snapshot_interval`
    /// commits; call it directly before a process is going to be killed.
    pub fn flush(&self) -> Result<()> {
        let mut wal = self.wal.lock();
        wal.sync()?;
        let applied = self.store.read()?.applied_lsn()?;

        let batch = self.store.write()?;
        self.indexes.read().persist(&batch)?;
        batch.set_meta_u64(META_INDEX_SNAPSHOT_LSN, applied)?;
        batch.commit()?;
        *self.commits_since_snapshot.lock() = 0;
        Ok(())
    }

    /// Rebuild indexes that have accumulated tombstones, drop absorbed log
    /// records, and reclaim freed pages.
    pub fn compact(&mut self) -> Result<CompactionReport> {
        let rebuilt = self.indexes.write().compact()?;
        self.flush()?;

        let applied = self.store.read()?.applied_lsn()?;
        let wal_bytes_before = {
            let mut wal = self.wal.lock();
            let before = wal.size_bytes()?;
            wal.compact(applied)?;
            before
        };
        let wal_bytes_after = self.wal.lock().size_bytes()?;
        let store_before = file_size(&self.path.join(STORE_FILE));
        let reclaimed_pages = self.store.compact()?;
        let store_after = file_size(&self.path.join(STORE_FILE));

        Ok(CompactionReport {
            indexes_rebuilt: rebuilt,
            wal_bytes_reclaimed: wal_bytes_before.saturating_sub(wal_bytes_after),
            store_bytes_reclaimed: store_before.saturating_sub(store_after),
            store_pages_freed: reclaimed_pages,
        })
    }

    /// Apply a batch of operations atomically.
    ///
    /// The public write path: log first, then tables, then memory.
    fn commit(&self, ops: Vec<WalOp>) -> Result<()> {
        if ops.is_empty() {
            return Ok(());
        }

        let mut wal = self.wal.lock();

        // 1. Record intent durably.
        let mut highest = 0;
        for op in &ops {
            highest = wal.append(op.clone())?;
        }
        wal.sync()?;

        // 2. Apply to the tables.
        let batch = self.store.write()?;
        for (offset, op) in ops.iter().enumerate() {
            let lsn = highest - (ops.len() - 1 - offset) as u64;
            apply_to_store(&batch, op, lsn)?;
        }
        batch.set_applied_lsn(highest)?;

        // 3. Publish to the in-memory indexes. Held across the table commit so
        //    a reader cannot observe a vector in one and not the other.
        let mut indexes = self.indexes.write();
        let mut snapshot_taken = false;
        for op in &ops {
            apply_to_indexes(&mut indexes, op)?;
        }

        let mut since = self.commits_since_snapshot.lock();
        *since += 1;
        if *since >= self.config.index_snapshot_interval {
            indexes.persist(&batch)?;
            batch.set_meta_u64(META_INDEX_SNAPSHOT_LSN, highest)?;
            snapshot_taken = true;
        }

        match batch.commit() {
            Ok(()) => {
                if snapshot_taken {
                    *since = 0;
                }
                Ok(())
            }
            Err(e) => {
                // The tables rejected the batch, so the in-memory indexes are
                // now ahead of durable state. Rebuild rather than serve reads
                // that disagree with what is on disk.
                drop(since);
                tracing::error!(error = %e, "commit failed; rebuilding vector indexes from store");
                *indexes = rebuild_indexes(&self.store, self.config.hnsw_params())?;
                Err(e)
            }
        }
    }
}

impl std::fmt::Debug for Db {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Db")
            .field("path", &self.path)
            .field("namespaces", &self.indexes.read().namespaces())
            .finish_non_exhaustive()
    }
}

impl Drop for Db {
    fn drop(&mut self) {
        if let Err(e) = self.flush() {
            tracing::error!(error = %e, "failed to flush database on close");
        }
    }
}

/// What a compaction pass recovered.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompactionReport {
    pub indexes_rebuilt: Vec<String>,
    pub wal_bytes_reclaimed: u64,
    pub store_bytes_reclaimed: u64,
    pub store_pages_freed: bool,
}

/// A batch of pending mutations.
///
/// Nothing is written until [`Transaction::commit`]; dropping the transaction
/// discards it.
pub struct Transaction<'a> {
    db: &'a Db,
    ops: Vec<WalOp>,
}

impl Transaction<'_> {
    /// Insert or replace a node.
    pub fn put_node(&mut self, node: impl Into<Node>) -> &mut Self {
        self.ops.push(WalOp::PutNode(Box::new(node.into())));
        self
    }

    /// Append an edge. Duplicates are ignored at apply time.
    pub fn put_edge(&mut self, edge: Edge) -> &mut Self {
        self.ops.push(WalOp::PutEdge(Box::new(edge)));
        self
    }

    /// Append an edge, plus its inverse where the vocabulary defines one.
    ///
    /// `PRECEDES` and `FOLLOWS` describe the same adjacency read two ways;
    /// writing only one leaves the other silently unqueryable.
    pub fn put_edge_symmetric(&mut self, edge: Edge) -> &mut Self {
        if let Some(inverse) = edge.edge_type.inverse() {
            let mut reverse = Edge::new(edge.to, edge.from, inverse);
            reverse.created_at = edge.created_at;
            reverse.properties = edge.properties.clone();
            self.ops.push(WalOp::PutEdge(Box::new(reverse)));
        }
        self.ops.push(WalOp::PutEdge(Box::new(edge)));
        self
    }

    /// Attach a vector to a node in a namespace.
    pub fn put_vector(&mut self, namespace: &str, node_id: NodeId, vector: Vec<f32>) -> &mut Self {
        self.ops.push(WalOp::PutVector {
            namespace: namespace.to_string(),
            node_id,
            vector,
        });
        self
    }

    /// Tombstone a node.
    pub fn delete_node(&mut self, id: NodeId) -> &mut Self {
        self.ops.push(WalOp::DeleteNode(id));
        self
    }

    /// Number of pending operations.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Write everything atomically.
    pub fn commit(self) -> Result<()> {
        self.db.commit(self.ops)
    }
}

/// Apply one operation to the tables.
fn apply_to_store(batch: &crate::store::WriteBatch, op: &WalOp, lsn: Lsn) -> Result<()> {
    match op {
        WalOp::PutNode(node) => {
            batch.put_node(node)?;
            // Keep the source-URI head pointing at the newest version, so
            // re-ingestion can find what it supersedes without a scan.
            if let Node::Document(doc) = node.as_ref() {
                let replace = match batch.head_document(&doc.source_uri)? {
                    Some(current) => match batch.get_node(current)? {
                        Some(Node::Document(existing)) => doc.version >= existing.version,
                        _ => true,
                    },
                    None => true,
                };
                if replace {
                    batch.set_head_document(&doc.source_uri, doc.id)?;
                }
            }
        }
        WalOp::PutEdge(edge) => {
            batch.put_edge(edge)?;
        }
        WalOp::PutVector {
            namespace,
            node_id,
            vector,
        } => {
            batch.put_vector(namespace, *node_id, vector)?;
        }
        WalOp::DeleteNode(id) => {
            batch.delete_node(*id, lsn)?;
        }
        WalOp::Checkpoint => {}
    }
    Ok(())
}

/// Apply one operation to the in-memory vector indexes.
fn apply_to_indexes(indexes: &mut VectorIndexSet, op: &WalOp) -> Result<()> {
    match op {
        WalOp::PutVector {
            namespace,
            node_id,
            vector,
        } => indexes.insert(namespace, *node_id, vector),
        WalOp::DeleteNode(id) => {
            indexes.remove_everywhere(*id);
            Ok(())
        }
        WalOp::PutNode(_) | WalOp::PutEdge(_) | WalOp::Checkpoint => Ok(()),
    }
}

/// Rebuild every index from the vector table, ignoring persisted graphs.
fn rebuild_indexes(store: &Store, params: HnswParams) -> Result<VectorIndexSet> {
    let view = store.read()?;
    let mut set = VectorIndexSet::new(params);
    for (namespace, _) in view.namespaces()? {
        view.scan_vectors(&namespace, |id, vector| set.insert(&namespace, id, vector))?;
    }
    Ok(set)
}

fn file_size(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

/// An advisory lock enforcing the single-writer model.
///
/// A stale lock left by a crashed process is taken over rather than treated as
/// fatal — the alternative is a database that needs manual repair after every
/// hard kill. Concurrent access from a live process is still caught, because
/// the running process holds the file open.
struct LockFile {
    path: PathBuf,
}

impl LockFile {
    fn acquire(path: PathBuf) -> Result<Self> {
        use std::io::Write;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;
        writeln!(file, "{}", std::process::id())?;
        Ok(LockFile { path })
    }
}

impl Drop for LockFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}
