//! The NexumDB data model: node types, edge types, and their properties.
//!
//! E        self.properties.get(key).and_then(PropertyValue::as_f64)PropertyValue::as_str)erything here is append-only. Documents are never mutated in place — a
//! re-ingest mints a new `Document` whose `supersedes_id` points at the prior
//! version, and edges are immutable once written.

use crate::id::{ContentHash, NodeId, Timestamp};
use crate::property::PropertyValue;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

use crate::error::{Error, Result};

/// Free-form per-node metadata carried alongside the typed properties.
pub type Metadata = BTreeMap<String, PropertyValue>;

/// Discriminant for the four node types.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    Document,
    Chunk,
    Entity,
    PipelineRun,
}

impl NodeKind {
    pub const ALL: [NodeKind; 4] = [
        NodeKind::Document,
        NodeKind::Chunk,
        NodeKind::Entity,
        NodeKind::PipelineRun,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            NodeKind::Document => "Document",
            NodeKind::Chunk => "Chunk",
            NodeKind::Entity => "Entity",
            NodeKind::PipelineRun => "PipelineRun",
        }
    }
}

impl fmt::Display for NodeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for NodeKind {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().replace(['_', '-'], "").as_str() {
            "document" | "doc" => Ok(NodeKind::Document),
            "chunk" => Ok(NodeKind::Chunk),
            "entity" => Ok(NodeKind::Entity),
            "pipelinerun" | "run" => Ok(NodeKind::PipelineRun),
            _ => Err(Error::InvalidArgument(format!("unknown node kind `{s}`"))),
        }
    }
}

/// The edge vocabulary, grouped as structural / semantic / provenance.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EdgeType {
    // Structural
    /// Chunk -> Document
    PartOf,
    /// Chunk -> Chunk, reading order
    Precedes,
    /// Chunk -> Chunk, reverse of `Precedes`
    Follows,
    /// Document -> Document, newer supersedes older
    Supersedes,

    // Semantic
    /// Chunk -> Entity
    Mentions,
    /// Entity -> Entity, carries a `relation_type` property
    RelatesTo,
    /// Chunk -> Chunk, cached precomputed top-k neighbours
    SimilarTo,

    // Provenance
    /// Chunk -> Document
    DerivedFrom,
    /// Chunk -> PipelineRun
    EmbeddedBy,
    /// Entity -> PipelineRun
    ExtractedBy,
}

/// Which family an edge type belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeClass {
    Structural,
    Semantic,
    Provenance,
}

impl EdgeType {
    pub const ALL: [EdgeType; 10] = [
        EdgeType::PartOf,
        EdgeType::Precedes,
        EdgeType::Follows,
        EdgeType::Supersedes,
        EdgeType::Mentions,
        EdgeType::RelatesTo,
        EdgeType::SimilarTo,
        EdgeType::DerivedFrom,
        EdgeType::EmbeddedBy,
        EdgeType::ExtractedBy,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            EdgeType::PartOf => "PART_OF",
            EdgeType::Precedes => "PRECEDES",
            EdgeType::Follows => "FOLLOWS",
            EdgeType::Supersedes => "SUPERSEDES",
            EdgeType::Mentions => "MENTIONS",
            EdgeType::RelatesTo => "RELATES_TO",
            EdgeType::SimilarTo => "SIMILAR_TO",
            EdgeType::DerivedFrom => "DERIVED_FROM",
            EdgeType::EmbeddedBy => "EMBEDDED_BY",
            EdgeType::ExtractedBy => "EXTRACTED_BY",
        }
    }

    pub const fn class(self) -> EdgeClass {
        match self {
            EdgeType::PartOf | EdgeType::Precedes | EdgeType::Follows | EdgeType::Supersedes => {
                EdgeClass::Structural
            }
            EdgeType::Mentions | EdgeType::RelatesTo | EdgeType::SimilarTo => EdgeClass::Semantic,
            EdgeType::DerivedFrom | EdgeType::EmbeddedBy | EdgeType::ExtractedBy => {
                EdgeClass::Provenance
            }
        }
    }

    /// Stable on-disk discriminant. Never renumber these — adjacency keys
    /// embed them, so a changed value silently orphans every existing edge.
    pub const fn code(self) -> u8 {
        match self {
            EdgeType::PartOf => 1,
            EdgeType::Precedes => 2,
            EdgeType::Follows => 3,
            EdgeType::Supersedes => 4,
            EdgeType::Mentions => 5,
            EdgeType::RelatesTo => 6,
            EdgeType::SimilarTo => 7,
            EdgeType::DerivedFrom => 8,
            EdgeType::EmbeddedBy => 9,
            EdgeType::ExtractedBy => 10,
        }
    }

    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(EdgeType::PartOf),
            2 => Some(EdgeType::Precedes),
            3 => Some(EdgeType::Follows),
            4 => Some(EdgeType::Supersedes),
            5 => Some(EdgeType::Mentions),
            6 => Some(EdgeType::RelatesTo),
            7 => Some(EdgeType::SimilarTo),
            8 => Some(EdgeType::DerivedFrom),
            9 => Some(EdgeType::EmbeddedBy),
            10 => Some(EdgeType::ExtractedBy),
            _ => None,
        }
    }

    /// The edge type that describes the same relationship read backwards,
    /// where one exists. Used to keep `PRECEDES`/`FOLLOWS` in sync.
    pub const fn inverse(self) -> Option<Self> {
        match self {
            EdgeType::Precedes => Some(EdgeType::Follows),
            EdgeType::Follows => Some(EdgeType::Precedes),
            _ => None,
        }
    }
}

impl fmt::Display for EdgeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for EdgeType {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        let norm = s.to_ascii_uppercase().replace('-', "_");
        EdgeType::ALL
            .into_iter()
            .find(|e| e.as_str() == norm)
            .ok_or_else(|| Error::InvalidArgument(format!("unknown edge type `{s}`")))
    }
}

/// Traversal direction across the adjacency index.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    /// Follow edges away from the node (node is the `from` side).
    #[default]
    Out,
    /// Follow edges into the node (node is the `to` side).
    In,
    /// Follow edges in both directions.
    Both,
}

impl Direction {
    pub const fn code(self) -> u8 {
        match self {
            Direction::Out => 0,
            Direction::In => 1,
            Direction::Both => 2,
        }
    }
}

impl FromStr for Direction {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "out" | "outgoing" => Ok(Direction::Out),
            "in" | "incoming" => Ok(Direction::In),
            "both" | "any" => Ok(Direction::Both),
            _ => Err(Error::InvalidArgument(format!("unknown direction `{s}`"))),
        }
    }
}

/// One embedding attached to a node, tagged with the run that produced it.
///
/// A node accumulates one of these per embedding model it has been run
/// through, which is what lets an embedding-model upgrade add vectors without
/// invalidating the old ones.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct EmbeddingMeta {
    /// Model identifier, e.g. `all-MiniLM-L6-v2` or `text-embedding-3-small`.
    pub model: String,
    /// Vector dimensionality.
    pub dim: usize,
    /// The `PipelineRun` that produced this vector.
    pub run_id: Option<NodeId>,
    pub embedded_at: Timestamp,
}

impl EmbeddingMeta {
    /// The vector index this embedding lives in. Indexes are namespaced by
    /// `(model, dim)` so several models can coexist in one database.
    pub fn namespace(&self) -> String {
        vector_namespace(&self.model, self.dim)
    }
}

/// Build the `(model, dim)` index namespace key.
pub fn vector_namespace(model: &str, dim: usize) -> String {
    format!("{model}:{dim}")
}

/// A source file or artifact. Immutable; re-ingestion mints a new version.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Document {
    pub id: NodeId,
    pub title: String,
    pub source_uri: String,
    pub content_hash: ContentHash,
    pub created_at: Timestamp,
    /// 1-based, incremented on each re-ingest of the same `source_uri`.
    pub version: u32,
    /// Prior version of this document, if any.
    pub supersedes_id: Option<NodeId>,
    /// The run that ingested this document.
    pub run_id: Option<NodeId>,
    #[serde(default)]
    pub metadata: Metadata,
}

/// A segment of a document, and the unit that retrieval returns.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Chunk {
    pub id: NodeId,
    pub document_id: NodeId,
    pub text: String,
    pub chunk_index: u32,
    pub token_count: u32,
    /// One entry per embedding model this chunk has been run through.
    #[serde(default)]
    pub embeddings: Vec<EmbeddingMeta>,
    pub created_at: Timestamp,
    #[serde(default)]
    pub metadata: Metadata,
}

impl Chunk {
    /// Look up this chunk's embedding for a given model, if it has one.
    pub fn embedding_for(&self, model: &str) -> Option<&EmbeddingMeta> {
        self.embeddings.iter().find(|e| e.model == model)
    }
}

/// An extracted named entity or concept.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Entity {
    pub id: NodeId,
    pub name: String,
    /// person / org / concept / location / …
    pub entity_type: String,
    /// Points at the surviving entity when this one was merged away by dedup.
    pub canonical_id: Option<NodeId>,
    #[serde(default)]
    pub embeddings: Vec<EmbeddingMeta>,
    pub created_at: Timestamp,
    #[serde(default)]
    pub metadata: Metadata,
}

impl Entity {
    /// True when this entity is an alias that dedup folded into another.
    pub fn is_alias(&self) -> bool {
        self.canonical_id.is_some_and(|c| c != self.id)
    }
}

/// Metadata about one ingestion/embedding run — the provenance anchor.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct PipelineRun {
    pub id: NodeId,
    pub pipeline_version: String,
    pub embedding_model: String,
    pub run_at: Timestamp,
    /// Hash over the full effective config, so identical runs are detectable.
    pub config_hash: ContentHash,
    pub chunker: String,
    #[serde(default)]
    pub metadata: Metadata,
}

/// Any node in the graph.
///
/// Serialises as `{"kind": "Chunk", ...}` in JSON, which is what the HTTP API
/// and viewer want, but as an externally tagged enum on disk. serde's
/// internally-tagged representation needs `deserialize_any`, which bincode
/// cannot provide, so the two encodings are selected explicitly below.
#[derive(Clone, PartialEq, Debug)]
pub enum Node {
    Document(Document),
    Chunk(Chunk),
    Entity(Entity),
    PipelineRun(PipelineRun),
}

/// JSON shape: internally tagged on `kind`.
#[derive(Serialize)]
#[serde(tag = "kind")]
enum NodeJsonRef<'a> {
    Document(&'a Document),
    Chunk(&'a Chunk),
    Entity(&'a Entity),
    PipelineRun(&'a PipelineRun),
}

#[derive(Deserialize)]
#[serde(tag = "kind")]
enum NodeJson {
    Document(Document),
    Chunk(Chunk),
    Entity(Entity),
    PipelineRun(PipelineRun),
}

/// On-disk shape: externally tagged, so no `deserialize_any` is needed.
#[derive(Serialize)]
enum NodeBinaryRef<'a> {
    Document(&'a Document),
    Chunk(&'a Chunk),
    Entity(&'a Entity),
    PipelineRun(&'a PipelineRun),
}

#[derive(Deserialize)]
enum NodeBinary {
    Document(Document),
    Chunk(Chunk),
    Entity(Entity),
    PipelineRun(PipelineRun),
}

macro_rules! project {
    ($node:expr, $target:ident) => {
        match $node {
            Node::Document(v) => $target::Document(v),
            Node::Chunk(v) => $target::Chunk(v),
            Node::Entity(v) => $target::Entity(v),
            Node::PipelineRun(v) => $target::PipelineRun(v),
        }
    };
}

macro_rules! lift {
    ($value:expr, $source:ident) => {
        match $value {
            $source::Document(v) => Node::Document(v),
            $source::Chunk(v) => Node::Chunk(v),
            $source::Entity(v) => Node::Entity(v),
            $source::PipelineRun(v) => Node::PipelineRun(v),
        }
    };
}

impl Serialize for Node {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        if serializer.is_human_readable() {
            project!(self, NodeJsonRef).serialize(serializer)
        } else {
            project!(self, NodeBinaryRef).serialize(serializer)
        }
    }
}

impl<'de> Deserialize<'de> for Node {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        if deserializer.is_human_readable() {
            Ok(lift!(NodeJson::deserialize(deserializer)?, NodeJson))
        } else {
            Ok(lift!(NodeBinary::deserialize(deserializer)?, NodeBinary))
        }
    }
}

impl Node {
    pub fn id(&self) -> NodeId {
        match self {
            Node::Document(d) => d.id,
            Node::Chunk(c) => c.id,
            Node::Entity(e) => e.id,
            Node::PipelineRun(r) => r.id,
        }
    }

    pub fn kind(&self) -> NodeKind {
        match self {
            Node::Document(_) => NodeKind::Document,
            Node::Chunk(_) => NodeKind::Chunk,
            Node::Entity(_) => NodeKind::Entity,
            Node::PipelineRun(_) => NodeKind::PipelineRun,
        }
    }

    pub fn created_at(&self) -> Timestamp {
        match self {
            Node::Document(d) => d.created_at,
            Node::Chunk(c) => c.created_at,
            Node::Entity(e) => e.created_at,
            Node::PipelineRun(r) => r.run_at,
        }
    }

    /// Short human label used by `nexum show` and the viewer's graph nodes.
    pub fn label(&self) -> String {
        match self {
            Node::Document(d) => d.title.clone(),
            Node::Chunk(c) => {
                let preview: String = c.text.chars().take(60).collect();
                if c.text.chars().nth(60).is_some() {
                    format!("{preview}…")
                } else {
                    preview
                }
            }
            Node::Entity(e) => e.name.clone(),
            Node::PipelineRun(r) => format!("run {} ({})", r.pipeline_version, r.embedding_model),
        }
    }

    pub fn metadata(&self) -> &Metadata {
        match self {
            Node::Document(d) => &d.metadata,
            Node::Chunk(c) => &c.metadata,
            Node::Entity(e) => &e.metadata,
            Node::PipelineRun(r) => &r.metadata,
        }
    }

    /// Embeddings attached to this node. Only chunks and entities carry them.
    pub fn embeddings(&self) -> &[EmbeddingMeta] {
        match self {
            Node::Chunk(c) => &c.embeddings,
            Node::Entity(e) => &e.embeddings,
            Node::Document(_) | Node::PipelineRun(_) => &[],
        }
    }

    pub fn as_document(&self) -> Result<&Document> {
        match self {
            Node::Document(d) => Ok(d),
            other => Err(Error::NodeKindMismatch {
                id: other.id(),
                actual: other.kind().as_str(),
                expected: "Document",
            }),
        }
    }

    pub fn as_chunk(&self) -> Result<&Chunk> {
        match self {
            Node::Chunk(c) => Ok(c),
            other => Err(Error::NodeKindMismatch {
                id: other.id(),
                actual: other.kind().as_str(),
                expected: "Chunk",
            }),
        }
    }

    pub fn as_entity(&self) -> Result<&Entity> {
        match self {
            Node::Entity(e) => Ok(e),
            other => Err(Error::NodeKindMismatch {
                id: other.id(),
                actual: other.kind().as_str(),
                expected: "Entity",
            }),
        }
    }

    pub fn as_pipeline_run(&self) -> Result<&PipelineRun> {
        match self {
            Node::PipelineRun(r) => Ok(r),
            other => Err(Error::NodeKindMismatch {
                id: other.id(),
                actual: other.kind().as_str(),
                expected: "PipelineRun",
            }),
        }
    }
}

impl From<Document> for Node {
    fn from(d: Document) -> Self {
        Node::Document(d)
    }
}
impl From<Chunk> for Node {
    fn from(c: Chunk) -> Self {
        Node::Chunk(c)
    }
}
impl From<Entity> for Node {
    fn from(e: Entity) -> Self {
        Node::Entity(e)
    }
}
impl From<PipelineRun> for Node {
    fn from(r: PipelineRun) -> Self {
        Node::PipelineRun(r)
    }
}

/// A directed, immutable edge.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
    pub edge_type: EdgeType,
    pub created_at: Timestamp,
    /// Edge-local properties, e.g. `relation_type` on `RELATES_TO` or
    /// `score` on `SIMILAR_TO`.
    #[serde(default)]
    pub properties: Metadata,
}

impl Edge {
    pub fn new(from: NodeId, to: NodeId, edge_type: EdgeType) -> Self {
        Edge {
            from,
            to,
            edge_type,
            created_at: Timestamp::now(),
            properties: Metadata::new(),
        }
    }

    /// Attach a property, builder-style.
    pub fn with(mut self, key: &str, value: impl Into<PropertyValue>) -> Self {
        self.properties.insert(key.to_string(), value.into());
        self
    }

    /// Read a property as a string.
    pub fn property_str(&self, key: &str) -> Option<&str> {
        self.properties.get(key).and_then(|v| v.as_str())
    }

    /// Read a property as a float.
    pub fn property_f64(&self, key: &str) -> Option<f64> {
        self.properties.get(key).and_then(|v| v.as_f64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_type_codes_are_unique_and_roundtrip() {
        let mut seen = std::collections::HashSet::new();
        for e in EdgeType::ALL {
            assert!(seen.insert(e.code()), "duplicate code for {e}");
            assert_eq!(Some(e), EdgeType::from_code(e.code()));
            assert_eq!(e, e.as_str().parse::<EdgeType>().unwrap());
        }
    }

    #[test]
    fn node_kinds_roundtrip_through_strings() {
        for k in NodeKind::ALL {
            assert_eq!(k, k.as_str().parse::<NodeKind>().unwrap());
        }
    }

    #[test]
    fn precedes_and_follows_are_inverses() {
        assert_eq!(EdgeType::Precedes.inverse(), Some(EdgeType::Follows));
        assert_eq!(EdgeType::Follows.inverse(), Some(EdgeType::Precedes));
        assert_eq!(EdgeType::PartOf.inverse(), None);
    }

    fn every_node_variant() -> Vec<Node> {
        let id = NodeId::new();
        let metadata = Metadata::from([("k".to_string(), PropertyValue::Text("v".into()))]);
        vec![
            Node::Document(Document {
                id,
                title: "T".into(),
                source_uri: "file:///t".into(),
                content_hash: ContentHash::of(b"t"),
                created_at: Timestamp::now(),
                version: 2,
                supersedes_id: Some(NodeId::new()),
                run_id: Some(NodeId::new()),
                metadata: metadata.clone(),
            }),
            Node::Chunk(Chunk {
                id,
                document_id: NodeId::new(),
                text: "body".into(),
                chunk_index: 3,
                token_count: 9,
                embeddings: vec![EmbeddingMeta {
                    model: "m".into(),
                    dim: 4,
                    run_id: Some(NodeId::new()),
                    embedded_at: Timestamp::now(),
                }],
                created_at: Timestamp::now(),
                metadata: metadata.clone(),
            }),
            Node::Entity(Entity {
                id,
                name: "Ada".into(),
                entity_type: "person".into(),
                canonical_id: None,
                embeddings: Vec::new(),
                created_at: Timestamp::now(),
                metadata: metadata.clone(),
            }),
            Node::PipelineRun(PipelineRun {
                id,
                pipeline_version: "1".into(),
                embedding_model: "m".into(),
                run_at: Timestamp::now(),
                config_hash: ContentHash::of(b"c"),
                chunker: "fixed".into(),
                metadata,
            }),
        ]
    }

    /// Guards the whole family of "works in JSON, silently breaks on disk"
    /// bugs: internally tagged enums and `skip_serializing_if` both compile
    /// fine and both make records undecodable by a non self-describing format.
    #[test]
    fn every_node_variant_roundtrips_through_the_on_disk_encoding() {
        for node in every_node_variant() {
            let bytes = crate::codec::encode(&node).unwrap();
            let decoded: Node = crate::codec::decode(&bytes).unwrap();
            assert_eq!(node, decoded, "{} failed binary roundtrip", node.kind());
        }
    }

    #[test]
    fn every_node_variant_roundtrips_through_json() {
        for node in every_node_variant() {
            let json = serde_json::to_string(&node).unwrap();
            let decoded: Node = serde_json::from_str(&json).unwrap();
            assert_eq!(node, decoded, "{} failed json roundtrip", node.kind());
        }
    }

    #[test]
    fn json_tags_nodes_with_a_kind_field() {
        for node in every_node_variant() {
            let json: serde_json::Value = serde_json::to_value(&node).unwrap();
            assert_eq!(
                json.get("kind").and_then(|k| k.as_str()),
                Some(node.kind().as_str()),
                "expected a flat `kind` discriminant, got {json}"
            );
            // The payload must be flattened alongside the tag, not nested.
            assert!(
                json.get("id").is_some(),
                "id should sit beside kind: {json}"
            );
        }
    }

    #[test]
    fn edges_roundtrip_through_both_encodings() {
        let edge = Edge::new(NodeId::new(), NodeId::new(), EdgeType::RelatesTo)
            .with("relation_type", "cites")
            .with("score", 0.75);
        let binary: Edge = crate::codec::decode(&crate::codec::encode(&edge).unwrap()).unwrap();
        assert_eq!(edge, binary);
        let json: Edge = serde_json::from_str(&serde_json::to_string(&edge).unwrap()).unwrap();
        assert_eq!(edge, json);
        assert_eq!(json.property_str("relation_type"), Some("cites"));
        assert_eq!(json.property_f64("score"), Some(0.75));
    }

    #[test]
    fn namespace_separates_models_and_dims() {
        assert_eq!(vector_namespace("m", 384), "m:384");
        assert_ne!(vector_namespace("m", 384), vector_namespace("m", 768));
    }
}
