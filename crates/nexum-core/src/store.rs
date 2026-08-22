//! The key-value storage layer.
//!
//! Backed by redb: a pure-Rust embedded store with MVCC and a
//! single-writer/multi-reader model, which is exactly the concurrency model
//! this database commits to. The spec suggested RocksDB; redb was chosen
//! instead so that Windows and Linux builds need no C++ toolchain. The tables
//! below are the layout the spec describes — a node store, an adjacency index,
//! and a separate vector store — and everything reaches them through this
//! module, so swapping the backend means reimplementing this file and nothing
//! else.

use crate::codec;
use crate::error::{Error, Result};
use crate::id::NodeId;
use crate::keys::{self, StoredDirection};
use crate::model::{Direction, Edge, EdgeType, Node, NodeKind};
use crate::wal::Lsn;
use redb::{
    Database, ReadTransaction, ReadableDatabase, ReadableTable, ReadableTableMetadata,
    TableDefinition, WriteTransaction,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

/// `node_id -> serialised Node`
const NODES: TableDefinition<&[u8], &[u8]> = TableDefinition::new("nodes");
/// `kind || node_id -> ()` — lets "all documents" be one range scan.
const KIND_INDEX: TableDefinition<&[u8], ()> = TableDefinition::new("kind_index");
/// `seq -> serialised Edge` — the append-only edge log.
const EDGE_LOG: TableDefinition<u64, &[u8]> = TableDefinition::new("edge_log");
/// `node || dir || edge_type || other -> edge seq`
const ADJACENCY: TableDefinition<&[u8], u64> = TableDefinition::new("adjacency");
/// `namespace || 0 || node_id -> raw f32 little-endian`
const VECTORS: TableDefinition<&[u8], &[u8]> = TableDefinition::new("vectors");
/// `namespace -> serialised NamespaceInfo`
const NAMESPACES: TableDefinition<&str, &[u8]> = TableDefinition::new("namespaces");
/// `namespace -> serialised HNSW graph`
const HNSW: TableDefinition<&str, &[u8]> = TableDefinition::new("hnsw");
/// `node_id -> lsn at which it was tombstoned`
const TOMBSTONES: TableDefinition<&[u8], u64> = TableDefinition::new("tombstones");
/// `source_uri -> newest Document node id`
const URI_HEAD: TableDefinition<&str, &[u8]> = TableDefinition::new("uri_head");
/// Engine-level scalars: format version, applied LSN, edge sequence.
const META: TableDefinition<&str, &[u8]> = TableDefinition::new("meta");

const META_FORMAT_VERSION: &str = "format_version";
const META_APPLIED_LSN: &str = "applied_lsn";
const META_EDGE_SEQ: &str = "edge_seq";
const META_CREATED_AT: &str = "created_at";

/// What a vector index namespace holds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NamespaceInfo {
    pub model: String,
    pub dim: usize,
    pub count: u64,
}

/// A neighbour reached across one edge.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Neighbor {
    pub target: NodeId,
    pub edge_type: EdgeType,
    pub direction: Direction,
    pub edge_seq: u64,
}

/// Handle to the on-disk tables.
pub struct Store {
    db: Database,
}

impl Store {
    /// Open or create the store at `path`, verifying the format version.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let db = Database::create(path)?;
        let store = Store { db };
        store.bootstrap(path)?;
        Ok(store)
    }

    /// Create every table on first open and stamp the format version, or
    /// refuse a database written by an incompatible build.
    fn bootstrap(&self, path: &Path) -> Result<()> {
        let txn = self.db.begin_write()?;
        {
            // Opening a table creates it if absent.
            txn.open_table(NODES)?;
            txn.open_table(KIND_INDEX)?;
            txn.open_table(EDGE_LOG)?;
            txn.open_table(ADJACENCY)?;
            txn.open_table(VECTORS)?;
            txn.open_table(NAMESPACES)?;
            txn.open_table(HNSW)?;
            txn.open_table(TOMBSTONES)?;
            txn.open_table(URI_HEAD)?;

            let mut meta = txn.open_table(META)?;
            let existing = meta
                .get(META_FORMAT_VERSION)?
                .map(|v| u32::from_le_bytes(v.value().try_into().unwrap_or([0; 4])));

            match existing {
                Some(found) if found != codec::FORMAT_VERSION => {
                    return Err(Error::IncompatibleFormat {
                        path: path.display().to_string(),
                        found,
                        supported: codec::FORMAT_VERSION,
                    });
                }
                Some(_) => {}
                None => {
                    meta.insert(
                        META_FORMAT_VERSION,
                        &codec::FORMAT_VERSION.to_le_bytes()[..],
                    )?;
                    meta.insert(
                        META_CREATED_AT,
                        &crate::id::Timestamp::now().as_millis().to_le_bytes()[..],
                    )?;
                }
            }
        }
        txn.commit()?;
        Ok(())
    }

    /// Begin a read snapshot. Readers never block the writer.
    pub fn read(&self) -> Result<ReadView> {
        Ok(ReadView {
            txn: self.db.begin_read()?,
        })
    }

    /// Begin the write transaction. Only one may be open at a time.
    pub fn write(&self) -> Result<WriteBatch> {
        Ok(WriteBatch {
            txn: self.db.begin_write()?,
        })
    }

    /// Reclaim space from freed pages.
    pub fn compact(&mut self) -> Result<bool> {
        Ok(self.db.compact()?)
    }
}

/// A consistent read snapshot of the store.
pub struct ReadView {
    txn: ReadTransaction,
}

impl ReadView {
    /// Fetch a node, or `None` if it is absent or tombstoned.
    pub fn get_node(&self, id: NodeId) -> Result<Option<Node>> {
        if self.is_deleted(id)? {
            return Ok(None);
        }
        self.get_node_including_deleted(id)
    }

    /// Fetch a node even if it has been tombstoned — used by history and
    /// audit views, which are allowed to see what retrieval hides.
    pub fn get_node_including_deleted(&self, id: NodeId) -> Result<Option<Node>> {
        let table = self.txn.open_table(NODES)?;
        match table.get(id.as_bytes().as_slice())? {
            Some(bytes) => Ok(Some(codec::decode(bytes.value())?)),
            None => Ok(None),
        }
    }

    pub fn is_deleted(&self, id: NodeId) -> Result<bool> {
        let table = self.txn.open_table(TOMBSTONES)?;
        Ok(table.get(id.as_bytes().as_slice())?.is_some())
    }

    /// Every live node ID of one kind, oldest first.
    pub fn ids_of_kind(&self, kind: NodeKind) -> Result<Vec<NodeId>> {
        let table = self.txn.open_table(KIND_INDEX)?;
        let tombstones = self.txn.open_table(TOMBSTONES)?;
        let (lo, hi) = keys::kind_index_range(kind);
        let mut out = Vec::new();
        for entry in table.range(lo.as_slice()..=hi.as_slice())? {
            let (key, _) = entry?;
            let id = keys::kind_index_node(key.value())?;
            if tombstones.get(id.as_bytes().as_slice())?.is_none() {
                out.push(id);
            }
        }
        Ok(out)
    }

    /// Number of live nodes of one kind.
    pub fn count_of_kind(&self, kind: NodeKind) -> Result<u64> {
        Ok(self.ids_of_kind(kind)?.len() as u64)
    }

    /// Neighbours of `node` across the given edge types.
    ///
    /// An empty `edge_types` means every edge type. Tombstoned targets are
    /// skipped, so traversal never surfaces deleted nodes.
    pub fn neighbors(
        &self,
        node: NodeId,
        edge_types: &[EdgeType],
        direction: Direction,
    ) -> Result<Vec<Neighbor>> {
        let table = self.txn.open_table(ADJACENCY)?;
        let tombstones = self.txn.open_table(TOMBSTONES)?;
        let types: &[EdgeType] = if edge_types.is_empty() {
            &EdgeType::ALL
        } else {
            edge_types
        };

        let mut out = Vec::new();
        for stored in StoredDirection::for_query(direction) {
            for edge_type in types {
                let (lo, hi) = keys::adjacency_range(node, *stored, *edge_type);
                for entry in table.range(lo.as_slice()..=hi.as_slice())? {
                    let (key, seq) = entry?;
                    let target = keys::adjacency_target(key.value())?;
                    if tombstones.get(target.as_bytes().as_slice())?.is_some() {
                        continue;
                    }
                    out.push(Neighbor {
                        target,
                        edge_type: *edge_type,
                        direction: match stored {
                            StoredDirection::Out => Direction::Out,
                            StoredDirection::In => Direction::In,
                        },
                        edge_seq: seq.value(),
                    });
                }
            }
        }
        Ok(out)
    }

    /// Full edge records incident to `node`.
    pub fn edges_of(
        &self,
        node: NodeId,
        edge_types: &[EdgeType],
        direction: Direction,
    ) -> Result<Vec<Edge>> {
        let log = self.txn.open_table(EDGE_LOG)?;
        let mut out = Vec::new();
        for neighbor in self.neighbors(node, edge_types, direction)? {
            if let Some(bytes) = log.get(neighbor.edge_seq)? {
                out.push(codec::decode(bytes.value())?);
            }
        }
        Ok(out)
    }

    /// Look up one edge by its sequence number in the log.
    pub fn get_edge(&self, seq: u64) -> Result<Option<Edge>> {
        let log = self.txn.open_table(EDGE_LOG)?;
        match log.get(seq)? {
            Some(bytes) => Ok(Some(codec::decode(bytes.value())?)),
            None => Ok(None),
        }
    }

    /// Whether a specific edge already exists.
    pub fn has_edge(&self, from: NodeId, to: NodeId, edge_type: EdgeType) -> Result<bool> {
        let table = self.txn.open_table(ADJACENCY)?;
        let key = keys::adjacency(from, StoredDirection::Out, edge_type, to);
        Ok(table.get(key.as_slice())?.is_some())
    }

    /// Total edges ever written, including those pointing at tombstoned nodes.
    pub fn edge_count(&self) -> Result<u64> {
        Ok(self.txn.open_table(EDGE_LOG)?.len()?)
    }

    /// Walk the whole edge log, oldest first.
    pub fn scan_edges(&self, mut f: impl FnMut(u64, Edge) -> Result<()>) -> Result<()> {
        let log = self.txn.open_table(EDGE_LOG)?;
        for entry in log.iter()? {
            let (seq, bytes) = entry?;
            f(seq.value(), codec::decode(bytes.value())?)?;
        }
        Ok(())
    }

    /// One node's vector in a namespace.
    pub fn get_vector(&self, namespace: &str, node: NodeId) -> Result<Option<Vec<f32>>> {
        let table = self.txn.open_table(VECTORS)?;
        let key = keys::vector(namespace, node);
        match table.get(key.as_slice())? {
            Some(bytes) => Ok(Some(codec::decode_vector(bytes.value())?)),
            None => Ok(None),
        }
    }

    /// Stream every vector in a namespace without materialising them all.
    pub fn scan_vectors(
        &self,
        namespace: &str,
        mut f: impl FnMut(NodeId, &[f32]) -> Result<()>,
    ) -> Result<()> {
        let table = self.txn.open_table(VECTORS)?;
        let (lo, hi) = keys::vector_range(namespace);
        for entry in table.range(lo.as_slice()..=hi.as_slice())? {
            let (key, bytes) = entry?;
            let id = keys::vector_node(key.value())?;
            let vector = codec::decode_vector(bytes.value())?;
            f(id, &vector)?;
        }
        Ok(())
    }

    /// All registered vector namespaces, keyed by `model:dim`.
    pub fn namespaces(&self) -> Result<BTreeMap<String, NamespaceInfo>> {
        let table = self.txn.open_table(NAMESPACES)?;
        let mut out = BTreeMap::new();
        for entry in table.iter()? {
            let (key, bytes) = entry?;
            out.insert(key.value().to_string(), codec::decode(bytes.value())?);
        }
        Ok(out)
    }

    /// The persisted HNSW graph for a namespace, if one has been saved.
    pub fn get_hnsw(&self, namespace: &str) -> Result<Option<Vec<u8>>> {
        let table = self.txn.open_table(HNSW)?;
        Ok(table.get(namespace)?.map(|v| v.value().to_vec()))
    }

    /// The newest document ID ingested from `source_uri`.
    pub fn head_document(&self, source_uri: &str) -> Result<Option<NodeId>> {
        let table = self.txn.open_table(URI_HEAD)?;
        match table.get(source_uri)? {
            Some(bytes) => {
                let mut id = [0u8; 16];
                if bytes.value().len() != 16 {
                    return Err(Error::Storage("malformed uri head entry".into()));
                }
                id.copy_from_slice(bytes.value());
                Ok(Some(NodeId::from_bytes(id)))
            }
            None => Ok(None),
        }
    }

    /// Every `source_uri` that has at least one document.
    pub fn source_uris(&self) -> Result<Vec<String>> {
        let table = self.txn.open_table(URI_HEAD)?;
        let mut out = Vec::new();
        for entry in table.iter()? {
            out.push(entry?.0.value().to_string());
        }
        Ok(out)
    }

    /// Highest WAL LSN the tables have absorbed.
    pub fn applied_lsn(&self) -> Result<Lsn> {
        self.read_u64(META_APPLIED_LSN)
    }

    /// Next free edge sequence number.
    pub fn edge_seq(&self) -> Result<u64> {
        self.read_u64(META_EDGE_SEQ)
    }

    /// When the database was created.
    pub fn created_at(&self) -> Result<crate::id::Timestamp> {
        Ok(crate::id::Timestamp::from_millis(
            self.read_u64(META_CREATED_AT)? as i64,
        ))
    }

    fn read_u64(&self, key: &str) -> Result<u64> {
        let table = self.txn.open_table(META)?;
        Ok(table
            .get(key)?
            .map(|v| u64::from_le_bytes(v.value().try_into().unwrap_or([0; 8])))
            .unwrap_or(0))
    }

    /// Number of tombstoned nodes.
    pub fn tombstone_count(&self) -> Result<u64> {
        Ok(self.txn.open_table(TOMBSTONES)?.len()?)
    }
}

/// A pending batch of mutations. Nothing is visible until [`WriteBatch::commit`].
pub struct WriteBatch {
    txn: WriteTransaction,
}

impl WriteBatch {
    /// Insert or replace a node, keeping the kind index in step.
    ///
    /// Replacement is used for in-place field updates that do not constitute a
    /// new version — attaching an embedding to an existing chunk, or pointing
    /// an entity at its canonical twin. Document content itself is never
    /// replaced; re-ingestion mints a new node.
    pub fn put_node(&self, node: &Node) -> Result<()> {
        let id = node.id();
        let encoded = codec::encode(node)?;
        {
            let mut nodes = self.txn.open_table(NODES)?;
            nodes.insert(id.as_bytes().as_slice(), encoded.as_slice())?;
        }
        {
            let mut index = self.txn.open_table(KIND_INDEX)?;
            index.insert(keys::kind_index(node.kind(), id).as_slice(), ())?;
        }
        Ok(())
    }

    /// Tombstone a node. The record and its edges stay on disk so history can
    /// still be reconstructed; reads simply stop returning it.
    pub fn delete_node(&self, id: NodeId, lsn: Lsn) -> Result<()> {
        let mut tombstones = self.txn.open_table(TOMBSTONES)?;
        tombstones.insert(id.as_bytes().as_slice(), lsn)?;
        Ok(())
    }

    /// Lift a tombstone.
    pub fn undelete_node(&self, id: NodeId) -> Result<()> {
        let mut tombstones = self.txn.open_table(TOMBSTONES)?;
        tombstones.remove(id.as_bytes().as_slice())?;
        Ok(())
    }

    /// Append an edge and index it in both directions.
    ///
    /// Duplicate edges are silently ignored rather than appended twice — edges
    /// are immutable, so re-writing one is a no-op by definition. Returns the
    /// sequence number, or `None` if the edge already existed.
    pub fn put_edge(&self, edge: &Edge) -> Result<Option<u64>> {
        {
            let adjacency = self.txn.open_table(ADJACENCY)?;
            let key = keys::adjacency(
                edge.from,
                StoredDirection::Out,
                edge.edge_type,
                edge.to,
            );
            if adjacency.get(key.as_slice())?.is_some() {
                return Ok(None);
            }
        }

        let seq = self.next_edge_seq()?;
        let encoded = codec::encode(edge)?;
        {
            let mut log = self.txn.open_table(EDGE_LOG)?;
            log.insert(seq, encoded.as_slice())?;
        }
        {
            let mut adjacency = self.txn.open_table(ADJACENCY)?;
            adjacency.insert(
                keys::adjacency(edge.from, StoredDirection::Out, edge.edge_type, edge.to)
                    .as_slice(),
                seq,
            )?;
            adjacency.insert(
                keys::adjacency(edge.to, StoredDirection::In, edge.edge_type, edge.from)
                    .as_slice(),
                seq,
            )?;
        }
        Ok(Some(seq))
    }

    fn next_edge_seq(&self) -> Result<u64> {
        let mut meta = self.txn.open_table(META)?;
        let current = meta
            .get(META_EDGE_SEQ)?
            .map(|v| u64::from_le_bytes(v.value().try_into().unwrap_or([0; 8])))
            .unwrap_or(0);
        meta.insert(META_EDGE_SEQ, &(current + 1).to_le_bytes()[..])?;
        Ok(current)
    }

    /// Store a vector and register its namespace.
    pub fn put_vector(&self, namespace: &str, node: NodeId, vector: &[f32]) -> Result<()> {
        let key = keys::vector(namespace, node);
        let is_new = {
            let mut vectors = self.txn.open_table(VECTORS)?;
            let previous = vectors.insert(key.as_slice(), codec::encode_vector(vector).as_slice())?;
            previous.is_none()
        };

        if is_new {
            let (model, dim) = split_namespace(namespace, vector.len());
            let mut namespaces = self.txn.open_table(NAMESPACES)?;
            let mut info = match namespaces.get(namespace)? {
                Some(bytes) => codec::decode::<NamespaceInfo>(bytes.value())?,
                None => NamespaceInfo {
                    model,
                    dim,
                    count: 0,
                },
            };
            if info.dim != vector.len() {
                return Err(Error::DimensionMismatch {
                    namespace: namespace.to_string(),
                    expected: info.dim,
                    got: vector.len(),
                });
            }
            info.count += 1;
            namespaces.insert(namespace, codec::encode(&info)?.as_slice())?;
        }
        Ok(())
    }

    /// Remove a vector and decrement its namespace count.
    pub fn delete_vector(&self, namespace: &str, node: NodeId) -> Result<()> {
        let key = keys::vector(namespace, node);
        let existed = {
            let mut vectors = self.txn.open_table(VECTORS)?;
            vectors.remove(key.as_slice())?.is_some()
        };
        if existed {
            let mut namespaces = self.txn.open_table(NAMESPACES)?;
            let existing = namespaces
                .get(namespace)?
                .map(|bytes| codec::decode::<NamespaceInfo>(bytes.value()))
                .transpose()?;
            if let Some(mut info) = existing {
                info.count = info.count.saturating_sub(1);
                namespaces.insert(namespace, codec::encode(&info)?.as_slice())?;
            }
        }
        Ok(())
    }

    /// Persist a serialised HNSW graph.
    pub fn put_hnsw(&self, namespace: &str, graph: &[u8]) -> Result<()> {
        let mut table = self.txn.open_table(HNSW)?;
        table.insert(namespace, graph)?;
        Ok(())
    }

    /// Point a source URI at its newest document.
    pub fn set_head_document(&self, source_uri: &str, id: NodeId) -> Result<()> {
        let mut table = self.txn.open_table(URI_HEAD)?;
        table.insert(source_uri, id.as_bytes().as_slice())?;
        Ok(())
    }

    /// Record how far the tables have caught up with the write-ahead log.
    pub fn set_applied_lsn(&self, lsn: Lsn) -> Result<()> {
        let mut meta = self.txn.open_table(META)?;
        meta.insert(META_APPLIED_LSN, &lsn.to_le_bytes()[..])?;
        Ok(())
    }

    /// Read a node inside the open write transaction, so a batch can build on
    /// what it has already written.
    pub fn get_node(&self, id: NodeId) -> Result<Option<Node>> {
        let nodes = self.txn.open_table(NODES)?;
        match nodes.get(id.as_bytes().as_slice())? {
            Some(bytes) => Ok(Some(codec::decode(bytes.value())?)),
            None => Ok(None),
        }
    }

    /// Same, for the URI head pointer.
    pub fn head_document(&self, source_uri: &str) -> Result<Option<NodeId>> {
        let table = self.txn.open_table(URI_HEAD)?;
        match table.get(source_uri)? {
            Some(bytes) if bytes.value().len() == 16 => {
                let mut id = [0u8; 16];
                id.copy_from_slice(bytes.value());
                Ok(Some(NodeId::from_bytes(id)))
            }
            Some(_) => Err(Error::Storage("malformed uri head entry".into())),
            None => Ok(None),
        }
    }

    /// Make every mutation in this batch visible, atomically.
    pub fn commit(self) -> Result<()> {
        self.txn.commit()?;
        Ok(())
    }

    /// Throw the batch away.
    pub fn abort(self) -> Result<()> {
        self.txn.abort()?;
        Ok(())
    }
}

/// Split a `model:dim` namespace back into its parts, falling back to the
/// observed vector length when the suffix is missing or unparseable.
fn split_namespace(namespace: &str, observed_dim: usize) -> (String, usize) {
    match namespace.rsplit_once(':') {
        Some((model, dim)) => match dim.parse::<usize>() {
            Ok(dim) => (model.to_string(), dim),
            Err(_) => (namespace.to_string(), observed_dim),
        },
        None => (namespace.to_string(), observed_dim),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{ContentHash, Timestamp};
    use crate::model::Document;

    fn store() -> (tempfile::TempDir, Store) {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path().join("nexum.redb")).unwrap();
        (dir, store)
    }

    fn document(title: &str) -> Node {
        Node::Document(Document {
            id: NodeId::new(),
            title: title.to_string(),
            source_uri: format!("file:///{title}"),
            content_hash: ContentHash::of(title.as_bytes()),
            created_at: Timestamp::now(),
            version: 1,
            supersedes_id: None,
            run_id: None,
            metadata: Default::default(),
        })
    }

    #[test]
    fn nodes_roundtrip_and_index_by_kind() {
        let (_dir, store) = store();
        let a = document("a");
        let b = document("b");
        let batch = store.write().unwrap();
        batch.put_node(&a).unwrap();
        batch.put_node(&b).unwrap();
        batch.commit().unwrap();

        let view = store.read().unwrap();
        assert_eq!(view.get_node(a.id()).unwrap().unwrap(), a);
        assert_eq!(view.count_of_kind(NodeKind::Document).unwrap(), 2);
        assert_eq!(view.count_of_kind(NodeKind::Chunk).unwrap(), 0);
    }

    #[test]
    fn tombstoned_nodes_disappear_from_reads_but_stay_on_disk() {
        let (_dir, store) = store();
        let a = document("a");
        let batch = store.write().unwrap();
        batch.put_node(&a).unwrap();
        batch.commit().unwrap();

        let batch = store.write().unwrap();
        batch.delete_node(a.id(), 7).unwrap();
        batch.commit().unwrap();

        let view = store.read().unwrap();
        assert!(view.get_node(a.id()).unwrap().is_none());
        assert!(view.get_node_including_deleted(a.id()).unwrap().is_some());
        assert_eq!(view.count_of_kind(NodeKind::Document).unwrap(), 0);
    }

    #[test]
    fn edges_are_traversable_in_both_directions() {
        let (_dir, store) = store();
        let a = document("a");
        let b = document("b");
        let edge = Edge::new(a.id(), b.id(), EdgeType::Supersedes);

        let batch = store.write().unwrap();
        batch.put_node(&a).unwrap();
        batch.put_node(&b).unwrap();
        assert_eq!(batch.put_edge(&edge).unwrap(), Some(0));
        batch.commit().unwrap();

        let view = store.read().unwrap();
        let out = view.neighbors(a.id(), &[], Direction::Out).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].target, b.id());

        let inbound = view.neighbors(b.id(), &[], Direction::In).unwrap();
        assert_eq!(inbound.len(), 1);
        assert_eq!(inbound[0].target, a.id());

        // The wrong direction finds nothing.
        assert!(view.neighbors(a.id(), &[], Direction::In).unwrap().is_empty());
        // Filtering by an unrelated edge type finds nothing.
        assert!(
            view.neighbors(a.id(), &[EdgeType::Mentions], Direction::Out)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn duplicate_edges_are_not_appended_twice() {
        let (_dir, store) = store();
        let a = document("a");
        let b = document("b");
        let edge = Edge::new(a.id(), b.id(), EdgeType::PartOf);

        let batch = store.write().unwrap();
        batch.put_node(&a).unwrap();
        batch.put_node(&b).unwrap();
        batch.put_edge(&edge).unwrap();
        batch.commit().unwrap();

        let batch = store.write().unwrap();
        assert_eq!(batch.put_edge(&edge).unwrap(), None);
        batch.commit().unwrap();

        assert_eq!(store.read().unwrap().edge_count().unwrap(), 1);
    }

    #[test]
    fn traversal_skips_tombstoned_targets() {
        let (_dir, store) = store();
        let a = document("a");
        let b = document("b");
        let batch = store.write().unwrap();
        batch.put_node(&a).unwrap();
        batch.put_node(&b).unwrap();
        batch
            .put_edge(&Edge::new(a.id(), b.id(), EdgeType::PartOf))
            .unwrap();
        batch.delete_node(b.id(), 1).unwrap();
        batch.commit().unwrap();

        let view = store.read().unwrap();
        assert!(view.neighbors(a.id(), &[], Direction::Out).unwrap().is_empty());
    }

    #[test]
    fn vectors_roundtrip_and_track_namespace_counts() {
        let (_dir, store) = store();
        let a = NodeId::new();
        let b = NodeId::new();

        let batch = store.write().unwrap();
        batch.put_vector("m:3", a, &[1.0, 2.0, 3.0]).unwrap();
        batch.put_vector("m:3", b, &[4.0, 5.0, 6.0]).unwrap();
        batch.commit().unwrap();

        let view = store.read().unwrap();
        assert_eq!(view.get_vector("m:3", a).unwrap().unwrap(), vec![1.0, 2.0, 3.0]);
        let namespaces = view.namespaces().unwrap();
        assert_eq!(namespaces["m:3"].count, 2);
        assert_eq!(namespaces["m:3"].dim, 3);
        assert_eq!(namespaces["m:3"].model, "m");

        let mut seen = 0;
        view.scan_vectors("m:3", |_, v| {
            seen += 1;
            assert_eq!(v.len(), 3);
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, 2);
    }

    #[test]
    fn rewriting_a_vector_does_not_double_count() {
        let (_dir, store) = store();
        let a = NodeId::new();
        let batch = store.write().unwrap();
        batch.put_vector("m:2", a, &[1.0, 2.0]).unwrap();
        batch.put_vector("m:2", a, &[3.0, 4.0]).unwrap();
        batch.commit().unwrap();

        let view = store.read().unwrap();
        assert_eq!(view.namespaces().unwrap()["m:2"].count, 1);
        assert_eq!(view.get_vector("m:2", a).unwrap().unwrap(), vec![3.0, 4.0]);
    }

    #[test]
    fn wrong_dimension_for_a_namespace_is_rejected() {
        let (_dir, store) = store();
        let batch = store.write().unwrap();
        batch.put_vector("m:3", NodeId::new(), &[1.0, 2.0, 3.0]).unwrap();
        let err = batch.put_vector("m:3", NodeId::new(), &[1.0, 2.0]).unwrap_err();
        assert!(matches!(err, Error::DimensionMismatch { .. }), "got {err}");
    }

    #[test]
    fn aborted_batches_leave_nothing_behind() {
        let (_dir, store) = store();
        let a = document("a");
        let batch = store.write().unwrap();
        batch.put_node(&a).unwrap();
        batch.abort().unwrap();
        assert!(store.read().unwrap().get_node(a.id()).unwrap().is_none());
    }

    #[test]
    fn reopening_an_incompatible_format_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nexum.redb");
        {
            let store = Store::open(&path).unwrap();
            let batch = store.write().unwrap();
            {
                let mut meta = batch.txn.open_table(META).unwrap();
                meta.insert(META_FORMAT_VERSION, &999u32.to_le_bytes()[..]).unwrap();
            }
            batch.commit().unwrap();
        }
        let err = match Store::open(&path) {
            Ok(_) => panic!("expected an incompatible-format error"),
            Err(e) => e,
        };
        assert!(matches!(err, Error::IncompatibleFormat { found: 999, .. }), "got {err}");
    }
}
