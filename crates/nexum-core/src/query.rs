//! The query engine.
//!
//! A query is a pipeline over three primitives, exactly as the spec frames it:
//! vector search, graph traversal, and property filtering. Each stage takes the
//! previous stage's node set and produces the next, so
//! `search(q, 20) -> traverse([MENTIONS, RELATES_TO], 2) -> filter(latest)`
//! is one pipeline rather than two round trips against two stores.
//!
//! Results carry how they were reached — similarity score, hop count, and the
//! edge that led to them — which is what makes "expand via graph" in the viewer
//! explainable rather than magic.

use crate::error::{Error, Result};
use crate::id::{NodeId, Timestamp};
use crate::model::{Direction, Edge, EdgeType, Node, NodeKind};
use crate::property::PropertyValue;
use crate::store::ReadView;
use crate::vector::VectorIndexSet;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// How a property is compared in a filter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompareOp {
    Eq,
    Ne,
    Lt,
    Lte,
    Gt,
    Gte,
    /// Substring match on text, or membership for a list-valued property.
    Contains,
    StartsWith,
    /// Property equals any of the given values.
    In,
}

/// A predicate over a node.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Predicate {
    /// Matches everything.
    All,
    Kind {
        kinds: Vec<NodeKind>,
    },
    /// Documents with no superseding version, and chunks belonging to one.
    ///
    /// This is the default retrieval filter — the spec's "queries default to
    /// latest version only".
    LatestVersion,
    /// The complement: only nodes that have been superseded.
    Superseded,
    CreatedAfter {
        at: Timestamp,
    },
    CreatedBefore {
        at: Timestamp,
    },
    /// Compare a metadata key, or one of the built-in fields exposed by
    /// [`field_value`] (`title`, `source_uri`, `version`, `text`, `name`,
    /// `entity_type`, `chunk_index`, `token_count`, `document_id`,
    /// `embedding_model`).
    Property {
        key: String,
        op: CompareOp,
        value: PropertyValue,
    },
    /// Node carries an embedding from the named model.
    HasEmbedding {
        model: String,
    },
    /// Chunk belongs to this document.
    InDocument {
        document_id: NodeId,
    },
    And {
        of: Vec<Predicate>,
    },
    Or {
        of: Vec<Predicate>,
    },
    Not {
        of: Box<Predicate>,
    },
}

impl Predicate {
    /// Convenience constructor for the common single-kind case.
    pub fn kind(kind: NodeKind) -> Self {
        Predicate::Kind { kinds: vec![kind] }
    }

    pub fn and(predicates: impl IntoIterator<Item = Predicate>) -> Self {
        Predicate::And {
            of: predicates.into_iter().collect(),
        }
    }

    pub fn or(predicates: impl IntoIterator<Item = Predicate>) -> Self {
        Predicate::Or {
            of: predicates.into_iter().collect(),
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn not(predicate: Predicate) -> Self {
        Predicate::Not {
            of: Box::new(predicate),
        }
    }

    /// Equality against a metadata key or built-in field.
    pub fn eq(key: &str, value: impl Into<PropertyValue>) -> Self {
        Predicate::Property {
            key: key.to_string(),
            op: CompareOp::Eq,
            value: value.into(),
        }
    }

    /// Evaluate against one node.
    pub fn matches(&self, node: &Node, ctx: &FilterContext<'_>) -> Result<bool> {
        Ok(match self {
            Predicate::All => true,
            Predicate::Kind { kinds } => kinds.contains(&node.kind()),
            Predicate::LatestVersion => ctx.is_latest(node)?,
            Predicate::Superseded => !ctx.is_latest(node)?,
            Predicate::CreatedAfter { at } => node.created_at() > *at,
            Predicate::CreatedBefore { at } => node.created_at() < *at,
            Predicate::HasEmbedding { model } => node
                .embeddings()
                .iter()
                .any(|e| e.model == *model || e.namespace() == *model),
            Predicate::InDocument { document_id } => match node {
                Node::Chunk(chunk) => chunk.document_id == *document_id,
                Node::Document(doc) => doc.id == *document_id,
                _ => false,
            },
            Predicate::Property { key, op, value } => match field_value(node, key) {
                Some(actual) => compare(&actual, op, value),
                // A missing property matches only an explicit `!= x`; every
                // other comparison against absent data is false rather than
                // vacuously true.
                None => matches!(op, CompareOp::Ne),
            },
            Predicate::And { of } => {
                for predicate in of {
                    if !predicate.matches(node, ctx)? {
                        return Ok(false);
                    }
                }
                true
            }
            Predicate::Or { of } => {
                for predicate in of {
                    if predicate.matches(node, ctx)? {
                        return Ok(true);
                    }
                }
                false
            }
            Predicate::Not { of } => !of.matches(node, ctx)?,
        })
    }
}

/// Read a built-in field or metadata key off a node as a comparable value.
///
/// Built-in fields shadow metadata keys of the same name, so a document's
/// `title` always means the real title.
pub fn field_value(node: &Node, key: &str) -> Option<PropertyValue> {
    let builtin = match (node, key) {
        (_, "id") => Some(PropertyValue::Text(node.id().to_string())),
        (_, "kind") => Some(PropertyValue::Text(node.kind().as_str().to_string())),
        (_, "created_at") => Some(PropertyValue::Int(node.created_at().as_millis())),

        (Node::Document(d), "title") => Some(PropertyValue::Text(d.title.clone())),
        (Node::Document(d), "source_uri") => Some(PropertyValue::Text(d.source_uri.clone())),
        (Node::Document(d), "content_hash") => Some(PropertyValue::Text(d.content_hash.0.clone())),
        (Node::Document(d), "version") => Some(PropertyValue::Int(d.version as i64)),

        (Node::Chunk(c), "text") => Some(PropertyValue::Text(c.text.clone())),
        (Node::Chunk(c), "document_id") => Some(PropertyValue::Text(c.document_id.to_string())),
        (Node::Chunk(c), "chunk_index") => Some(PropertyValue::Int(c.chunk_index as i64)),
        (Node::Chunk(c), "token_count") => Some(PropertyValue::Int(c.token_count as i64)),
        (Node::Chunk(c), "embedding_model") => Some(PropertyValue::List(
            c.embeddings
                .iter()
                .map(|e| PropertyValue::Text(e.model.clone()))
                .collect(),
        )),

        (Node::Entity(e), "name") => Some(PropertyValue::Text(e.name.clone())),
        (Node::Entity(e), "entity_type") => Some(PropertyValue::Text(e.entity_type.clone())),
        (Node::Entity(e), "canonical_id") => {
            e.canonical_id.map(|id| PropertyValue::Text(id.to_string()))
        }

        (Node::PipelineRun(r), "pipeline_version") => {
            Some(PropertyValue::Text(r.pipeline_version.clone()))
        }
        (Node::PipelineRun(r), "embedding_model") => {
            Some(PropertyValue::Text(r.embedding_model.clone()))
        }
        (Node::PipelineRun(r), "chunker") => Some(PropertyValue::Text(r.chunker.clone())),
        _ => None,
    };
    builtin.or_else(|| node.metadata().get(key).cloned())
}

/// Compare an actual value against an expected one.
fn compare(actual: &PropertyValue, op: &CompareOp, expected: &PropertyValue) -> bool {
    match op {
        CompareOp::Eq => actual == expected,
        CompareOp::Ne => actual != expected,
        CompareOp::Lt | CompareOp::Lte | CompareOp::Gt | CompareOp::Gte => {
            let ordering = match (actual.as_f64(), expected.as_f64()) {
                (Some(a), Some(b)) => a.partial_cmp(&b),
                // Fall back to lexicographic ordering for text.
                _ => match (actual.as_str(), expected.as_str()) {
                    (Some(a), Some(b)) => Some(a.cmp(b)),
                    _ => None,
                },
            };
            match ordering {
                Some(std::cmp::Ordering::Less) => matches!(op, CompareOp::Lt | CompareOp::Lte),
                Some(std::cmp::Ordering::Equal) => matches!(op, CompareOp::Lte | CompareOp::Gte),
                Some(std::cmp::Ordering::Greater) => matches!(op, CompareOp::Gt | CompareOp::Gte),
                None => false,
            }
        }
        CompareOp::Contains => match (actual, expected) {
            (PropertyValue::Text(haystack), PropertyValue::Text(needle)) => {
                haystack.to_lowercase().contains(&needle.to_lowercase())
            }
            (PropertyValue::List(items), needle) => items.contains(needle),
            _ => false,
        },
        CompareOp::StartsWith => match (actual.as_str(), expected.as_str()) {
            (Some(haystack), Some(prefix)) => {
                haystack.to_lowercase().starts_with(&prefix.to_lowercase())
            }
            _ => false,
        },
        CompareOp::In => match expected {
            PropertyValue::List(options) => options.contains(actual),
            _ => false,
        },
    }
}

/// Everything predicate evaluation needs beyond the node itself.
///
/// Version checks require looking at neighbouring documents, so they get a
/// memoised view rather than re-walking the adjacency index per node.
pub struct FilterContext<'a> {
    view: &'a ReadView,
    latest_cache: std::cell::RefCell<HashMap<NodeId, bool>>,
}

impl<'a> FilterContext<'a> {
    pub fn new(view: &'a ReadView) -> Self {
        FilterContext {
            view,
            latest_cache: std::cell::RefCell::new(HashMap::new()),
        }
    }

    /// Whether a node belongs to the current version of its document.
    ///
    /// A document is current when nothing supersedes it. A chunk inherits the
    /// answer from its document. Everything else is always current — entities
    /// and pipeline runs are not versioned.
    pub fn is_latest(&self, node: &Node) -> Result<bool> {
        match node {
            Node::Document(doc) => self.document_is_latest(doc.id),
            Node::Chunk(chunk) => self.document_is_latest(chunk.document_id),
            Node::Entity(_) | Node::PipelineRun(_) => Ok(true),
        }
    }

    /// True when no other document supersedes this one.
    pub fn document_is_latest(&self, document_id: NodeId) -> Result<bool> {
        if let Some(cached) = self.latest_cache.borrow().get(&document_id) {
            return Ok(*cached);
        }
        // The superseding document points *at* this one, so an inbound
        // SUPERSEDES edge is exactly the evidence that a newer version exists.
        let superseded = !self
            .view
            .neighbors(document_id, &[EdgeType::Supersedes], Direction::In)?
            .is_empty();
        let latest = !superseded;
        self.latest_cache.borrow_mut().insert(document_id, latest);
        Ok(latest)
    }

    pub fn view(&self) -> &ReadView {
        self.view
    }
}

/// One stage of a query pipeline.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "stage", rename_all = "snake_case")]
pub enum Stage {
    /// Start from an explicit node set.
    Seed { ids: Vec<NodeId> },
    /// Start from every node of a kind.
    SeedKind { kind: NodeKind },
    /// Approximate nearest-neighbour search.
    Search {
        vector: Vec<f32>,
        /// Model or full `model:dim` namespace. Omitted when the database has
        /// exactly one.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        top_k: usize,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        ef: Option<usize>,
    },
    /// Expand across edges.
    Traverse {
        #[serde(default)]
        edge_types: Vec<EdgeType>,
        #[serde(default)]
        direction: Direction,
        max_hops: usize,
        /// Cap on nodes visited, so a hub node cannot blow up a query.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        limit: Option<usize>,
        /// Keep the nodes the traversal started from in the output.
        #[serde(default)]
        include_start: bool,
    },
    /// Keep only nodes matching a predicate.
    Filter { predicate: Predicate },
    /// Truncate, preserving order.
    Limit { n: usize },
}

/// A node in a query result, with how it was reached.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResultNode {
    pub node: Node,
    /// Similarity in 0..=1 when the node came from a vector search.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
    /// Raw metric distance behind `score`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distance: Option<f32>,
    /// Hops from the traversal's starting set, when a traversal produced it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hops: Option<usize>,
    /// The node this one was reached from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub via: Option<NodeId>,
    /// The edge type it was reached across.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub via_edge: Option<EdgeType>,
}

impl ResultNode {
    pub fn new(node: Node) -> Self {
        ResultNode {
            node,
            score: None,
            distance: None,
            hops: None,
            via: None,
            via_edge: None,
        }
    }

    pub fn id(&self) -> NodeId {
        self.node.id()
    }
}

/// A query result set, plus what it cost to produce.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QueryResult {
    pub nodes: Vec<ResultNode>,
    pub stats: QueryStats,
}

impl QueryResult {
    pub fn ids(&self) -> Vec<NodeId> {
        self.nodes.iter().map(ResultNode::id).collect()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

/// What each stage of a query did — the basis for the viewer's "expand via
/// graph showed you N more chunks" explanation.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct QueryStats {
    pub stages: Vec<StageStats>,
    pub nodes_visited: usize,
    pub edges_traversed: usize,
}

/// Per-stage accounting.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StageStats {
    pub stage: String,
    pub in_count: usize,
    pub out_count: usize,
}

/// A query pipeline.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Query {
    pub stages: Vec<Stage>,
}

impl Query {
    pub fn new() -> Self {
        Query::default()
    }

    /// Start from explicit nodes.
    pub fn seed(mut self, ids: impl IntoIterator<Item = NodeId>) -> Self {
        self.stages.push(Stage::Seed {
            ids: ids.into_iter().collect(),
        });
        self
    }

    /// Start from every node of a kind.
    pub fn seed_kind(mut self, kind: NodeKind) -> Self {
        self.stages.push(Stage::SeedKind { kind });
        self
    }

    /// Vector search.
    pub fn search(mut self, vector: Vec<f32>, top_k: usize) -> Self {
        self.stages.push(Stage::Search {
            vector,
            model: None,
            top_k,
            ef: None,
        });
        self
    }

    /// Vector search against a named model.
    pub fn search_with(mut self, vector: Vec<f32>, top_k: usize, model: Option<String>) -> Self {
        self.stages.push(Stage::Search {
            vector,
            model,
            top_k,
            ef: None,
        });
        self
    }

    /// Expand across edges.
    pub fn traverse(mut self, edge_types: Vec<EdgeType>, max_hops: usize) -> Self {
        self.stages.push(Stage::Traverse {
            edge_types,
            direction: Direction::Both,
            max_hops,
            limit: None,
            include_start: false,
        });
        self
    }

    /// Expand across edges in one direction, optionally keeping the start set.
    pub fn traverse_with(
        mut self,
        edge_types: Vec<EdgeType>,
        max_hops: usize,
        direction: Direction,
        include_start: bool,
    ) -> Self {
        self.stages.push(Stage::Traverse {
            edge_types,
            direction,
            max_hops,
            limit: None,
            include_start,
        });
        self
    }

    /// Filter by predicate.
    pub fn filter(mut self, predicate: Predicate) -> Self {
        self.stages.push(Stage::Filter { predicate });
        self
    }

    /// Restrict to the current version of every document. The usual last step.
    pub fn latest(self) -> Self {
        self.filter(Predicate::LatestVersion)
    }

    /// Restrict to one node kind.
    pub fn of_kind(self, kind: NodeKind) -> Self {
        self.filter(Predicate::kind(kind))
    }

    pub fn limit(mut self, n: usize) -> Self {
        self.stages.push(Stage::Limit { n });
        self
    }

    /// Run the pipeline.
    pub fn execute(&self, view: &ReadView, indexes: &VectorIndexSet) -> Result<QueryResult> {
        Engine::new(view, indexes).run(self)
    }
}

/// Pipeline executor.
pub struct Engine<'a> {
    view: &'a ReadView,
    indexes: &'a VectorIndexSet,
}

impl<'a> Engine<'a> {
    pub fn new(view: &'a ReadView, indexes: &'a VectorIndexSet) -> Self {
        Engine { view, indexes }
    }

    /// Run every stage in order, threading the node set through.
    pub fn run(&self, query: &Query) -> Result<QueryResult> {
        let ctx = FilterContext::new(self.view);
        let mut current: Vec<ResultNode> = Vec::new();
        let mut stats = QueryStats::default();

        for stage in &query.stages {
            let in_count = current.len();
            current = self.run_stage(stage, current, &ctx, &mut stats)?;
            stats.stages.push(StageStats {
                stage: stage_name(stage).to_string(),
                in_count,
                out_count: current.len(),
            });
        }

        Ok(QueryResult {
            nodes: current,
            stats,
        })
    }

    fn run_stage(
        &self,
        stage: &Stage,
        input: Vec<ResultNode>,
        ctx: &FilterContext<'_>,
        stats: &mut QueryStats,
    ) -> Result<Vec<ResultNode>> {
        match stage {
            Stage::Seed { ids } => {
                let mut out = Vec::with_capacity(ids.len());
                for id in ids {
                    if let Some(node) = self.view.get_node(*id)? {
                        out.push(ResultNode::new(node));
                    }
                }
                stats.nodes_visited += out.len();
                Ok(out)
            }

            Stage::SeedKind { kind } => {
                let ids = self.view.ids_of_kind(*kind)?;
                let mut out = Vec::with_capacity(ids.len());
                for id in ids {
                    if let Some(node) = self.view.get_node(id)? {
                        out.push(ResultNode::new(node));
                    }
                }
                stats.nodes_visited += out.len();
                Ok(out)
            }

            Stage::Search {
                vector,
                model,
                top_k,
                ef,
            } => {
                let namespace = self.indexes.resolve(model.as_deref())?;
                let metric = self
                    .indexes
                    .get(&namespace)
                    .map(|index| index.metric())
                    .unwrap_or_default();

                // A search after an earlier stage is scoped to that stage's
                // output — this is what makes "search within these documents"
                // a pipeline rather than a special case.
                let scope: Option<HashSet<NodeId>> = if input.is_empty() {
                    None
                } else {
                    Some(input.iter().map(ResultNode::id).collect())
                };
                let filter = scope
                    .as_ref()
                    .map(|scope| move |id: NodeId| scope.contains(&id));
                let filter_ref: Option<&dyn Fn(NodeId) -> bool> = match &filter {
                    Some(f) => Some(f),
                    None => None,
                };

                let hits = self
                    .indexes
                    .search(&namespace, vector, *top_k, *ef, filter_ref)?;

                let mut out = Vec::with_capacity(hits.len());
                for hit in hits {
                    if let Some(node) = self.view.get_node(hit.id)? {
                        out.push(ResultNode {
                            node,
                            score: Some(metric.similarity(hit.distance)),
                            distance: Some(hit.distance),
                            hops: None,
                            via: None,
                            via_edge: None,
                        });
                    }
                }
                stats.nodes_visited += out.len();
                Ok(out)
            }

            Stage::Traverse {
                edge_types,
                direction,
                max_hops,
                limit,
                include_start,
            } => self.traverse(
                input,
                edge_types,
                *direction,
                *max_hops,
                *limit,
                *include_start,
                stats,
            ),

            Stage::Filter { predicate } => {
                let mut out = Vec::with_capacity(input.len());
                for item in input {
                    if predicate.matches(&item.node, ctx)? {
                        out.push(item);
                    }
                }
                Ok(out)
            }

            Stage::Limit { n } => {
                let mut out = input;
                out.truncate(*n);
                Ok(out)
            }
        }
    }

    /// Breadth-first expansion, recording the shortest path to each node.
    ///
    /// BFS rather than DFS so `hops` is genuinely the shortest distance, which
    /// is what the hop-depth control in the graph explorer means.
    #[allow(clippy::too_many_arguments)]
    fn traverse(
        &self,
        input: Vec<ResultNode>,
        edge_types: &[EdgeType],
        direction: Direction,
        max_hops: usize,
        limit: Option<usize>,
        include_start: bool,
        stats: &mut QueryStats,
    ) -> Result<Vec<ResultNode>> {
        let mut seen: HashSet<NodeId> = input.iter().map(ResultNode::id).collect();
        let mut queue: VecDeque<(NodeId, usize)> =
            input.iter().map(|item| (item.id(), 0usize)).collect();

        let mut out: Vec<ResultNode> = if include_start { input } else { Vec::new() };
        let cap = limit.unwrap_or(usize::MAX);

        while let Some((id, hops)) = queue.pop_front() {
            if hops >= max_hops || out.len() >= cap {
                continue;
            }
            for neighbor in self.view.neighbors(id, edge_types, direction)? {
                stats.edges_traversed += 1;
                if !seen.insert(neighbor.target) {
                    continue;
                }
                let Some(node) = self.view.get_node(neighbor.target)? else {
                    continue;
                };
                stats.nodes_visited += 1;
                out.push(ResultNode {
                    node,
                    score: None,
                    distance: None,
                    hops: Some(hops + 1),
                    via: Some(id),
                    via_edge: Some(neighbor.edge_type),
                });
                if out.len() >= cap {
                    break;
                }
                queue.push_back((neighbor.target, hops + 1));
            }
        }
        Ok(out)
    }
}

fn stage_name(stage: &Stage) -> &'static str {
    match stage {
        Stage::Seed { .. } => "seed",
        Stage::SeedKind { .. } => "seed_kind",
        Stage::Search { .. } => "search",
        Stage::Traverse { .. } => "traverse",
        Stage::Filter { .. } => "filter",
        Stage::Limit { .. } => "limit",
    }
}

/// Every version of a document, oldest first.
///
/// Walks the supersession chain rather than trusting `version` numbers, so a
/// chain assembled out of order still reads correctly.
pub fn version_chain(view: &ReadView, document_id: NodeId) -> Result<Vec<Node>> {
    // Walk backwards to the original, then forwards collecting each version.
    let mut oldest = document_id;
    let mut guard = HashSet::new();
    loop {
        if !guard.insert(oldest) {
            return Err(Error::Storage(format!(
                "supersession cycle detected at {oldest}"
            )));
        }
        let older = view.neighbors(oldest, &[EdgeType::Supersedes], Direction::Out)?;
        match older.first() {
            Some(next) => oldest = next.target,
            None => break,
        }
    }

    let mut chain = Vec::new();
    let mut current = Some(oldest);
    let mut guard = HashSet::new();
    while let Some(id) = current {
        if !guard.insert(id) {
            break;
        }
        if let Some(node) = view.get_node_including_deleted(id)? {
            chain.push(node);
        }
        let newer = view.neighbors(id, &[EdgeType::Supersedes], Direction::In)?;
        current = newer.first().map(|n| n.target);
    }
    Ok(chain)
}

/// A node together with the edges around it — what `nexum show` prints and
/// what the viewer's inspector renders.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NodeDetail {
    pub node: Node,
    pub outgoing: Vec<EdgeWithTarget>,
    pub incoming: Vec<EdgeWithTarget>,
}

/// An edge paired with the label of the node at its far end.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EdgeWithTarget {
    pub edge: Edge,
    pub other_id: NodeId,
    pub other_kind: Option<NodeKind>,
    pub other_label: Option<String>,
}

/// Fetch a node with its full edge neighbourhood.
pub fn node_detail(view: &ReadView, id: NodeId) -> Result<Option<NodeDetail>> {
    let Some(node) = view.get_node_including_deleted(id)? else {
        return Ok(None);
    };

    let mut outgoing = Vec::new();
    let mut incoming = Vec::new();
    for (direction, bucket) in [
        (Direction::Out, &mut outgoing),
        (Direction::In, &mut incoming),
    ] {
        for neighbor in view.neighbors(id, &[], direction)? {
            let Some(edge) = view.get_edge(neighbor.edge_seq)? else {
                continue;
            };
            let other = view.get_node_including_deleted(neighbor.target)?;
            bucket.push(EdgeWithTarget {
                edge,
                other_id: neighbor.target,
                other_kind: other.as_ref().map(Node::kind),
                other_label: other.as_ref().map(Node::label),
            });
        }
    }

    Ok(Some(NodeDetail {
        node,
        outgoing,
        incoming,
    }))
}
