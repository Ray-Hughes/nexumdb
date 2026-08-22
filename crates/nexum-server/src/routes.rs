//! HTTP routes.
//!
//! Every read endpoint returns the same shapes the CLI prints, because both
//! are rendering the same client-library types. Endpoints that can return a
//! lot of rows are paginated with an explicit total, so the viewer can show
//! "1-50 of 12,431" rather than guessing.

use crate::ServerState;
use crate::error::{ApiError, ApiResult};
use crate::projection::{self, ProjectionMethod, ProjectionParams};
use axum::extract::{Path, Query as AxumQuery, State};
use axum::routing::{get, post};
use axum::{Json, Router};
use nexum_client::{Expansion, SearchOptions, SearchResults};
use nexum_core::model::Document;
use nexum_core::{
    Chunk, Direction, EdgeType, Node, NodeDetail, NodeId, NodeKind, Predicate, Query, QueryResult,
};
use nexum_ingest::IngestReport;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

pub fn routes() -> Router<ServerState> {
    Router::new()
        .route("/health", get(health))
        .route("/api/stats", get(stats))
        .route("/api/config", get(config))
        .route("/api/documents", get(list_documents))
        .route("/api/documents/{id}", get(get_document))
        .route("/api/documents/{id}/history", get(document_history))
        .route("/api/documents/{id}/chunks", get(document_chunks))
        .route("/api/nodes/{id}", get(get_node))
        .route("/api/nodes/{id}/neighbors", get(node_neighbors))
        .route("/api/graph/{id}", get(graph))
        .route("/api/search", post(search))
        .route("/api/traverse", post(traverse))
        .route("/api/query", post(run_query))
        .route("/api/ingest", post(ingest))
        .route("/api/projection", get(projection_route))
}

// ---- Health and metadata -------------------------------------------------

#[derive(Serialize)]
struct Health {
    status: &'static str,
    engine_version: &'static str,
    database: String,
}

async fn health(State(state): State<ServerState>) -> Json<Health> {
    Json(Health {
        status: "ok",
        engine_version: nexum_core::VERSION,
        database: state.nexum.path().display().to_string(),
    })
}

async fn stats(State(state): State<ServerState>) -> ApiResult<Json<nexum_core::DbStats>> {
    Ok(Json(state.nexum.stats()?))
}

#[derive(Serialize)]
struct ConfigResponse {
    embedding_model: String,
    embedding_dimensions: usize,
    embedder: String,
    namespaces: Vec<String>,
    engine_version: &'static str,
}

async fn config(State(state): State<ServerState>) -> Json<ConfigResponse> {
    Json(ConfigResponse {
        embedding_model: state.nexum.embedder().model_id().to_string(),
        embedding_dimensions: state.nexum.embedder().dim(),
        embedder: state.nexum.embedder().describe(),
        namespaces: state.nexum.db().namespaces(),
        engine_version: nexum_core::VERSION,
    })
}

// ---- Pagination ----------------------------------------------------------

/// A page of results, with enough context for a UI to render a pager.
#[derive(Serialize)]
struct Page<T> {
    items: Vec<T>,
    total: usize,
    offset: usize,
    limit: usize,
}

#[derive(Deserialize)]
struct PageParams {
    #[serde(default)]
    offset: usize,
    #[serde(default = "default_limit")]
    limit: usize,
}

fn default_limit() -> usize {
    50
}

impl PageParams {
    /// Cap the page size so one request cannot ask for the whole database.
    fn apply<T>(&self, items: Vec<T>) -> Page<T> {
        let total = items.len();
        let limit = self.limit.clamp(1, 1_000);
        let page = items.into_iter().skip(self.offset).take(limit).collect();
        Page {
            items: page,
            total,
            offset: self.offset,
            limit,
        }
    }
}

// ---- Documents -----------------------------------------------------------

/// Query parameters are deserialised from a urlencoded string, where every
/// value is text and `#[serde(flatten)]` cannot work — it needs a
/// self-describing format. The page fields are therefore spelled out.
#[derive(Deserialize)]
struct ListDocumentsParams {
    #[serde(default)]
    include_superseded: bool,
    #[serde(default)]
    offset: usize,
    #[serde(default = "default_limit")]
    limit: usize,
}

impl ListDocumentsParams {
    fn page(&self) -> PageParams {
        PageParams {
            offset: self.offset,
            limit: self.limit,
        }
    }
}

/// A document plus the counts the collection browser shows.
#[derive(Serialize)]
struct DocumentSummary {
    #[serde(flatten)]
    document: Document,
    chunk_count: usize,
    is_latest: bool,
}

async fn list_documents(
    State(state): State<ServerState>,
    AxumQuery(params): AxumQuery<ListDocumentsParams>,
) -> ApiResult<Json<Page<DocumentSummary>>> {
    let latest_ids: Vec<NodeId> = state
        .nexum
        .documents(false)?
        .into_iter()
        .map(|d| d.id)
        .collect();

    let mut summaries = Vec::new();
    for document in state.nexum.documents(params.include_superseded)? {
        let chunk_count = state.nexum.chunks_of(document.id)?.len();
        summaries.push(DocumentSummary {
            is_latest: latest_ids.contains(&document.id),
            chunk_count,
            document,
        });
    }
    // Newest first: that is the order a collection browser wants.
    summaries.sort_by_key(|s| std::cmp::Reverse(s.document.created_at));

    Ok(Json(params.page().apply(summaries)))
}

async fn get_document(
    State(state): State<ServerState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Document>> {
    let id = parse_id(&id)?;
    state
        .nexum
        .get(id)?
        .and_then(|n| n.as_document().ok().cloned())
        .map(Json)
        .ok_or_else(|| ApiError::NotFound(format!("no document {id}")))
}

async fn document_history(
    State(state): State<ServerState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Vec<Document>>> {
    let id = parse_id(&id)?;
    let history = state.nexum.history(id)?;
    if history.is_empty() {
        return Err(ApiError::NotFound(format!("no document {id}")));
    }
    Ok(Json(history))
}

async fn document_chunks(
    State(state): State<ServerState>,
    Path(id): Path<String>,
    AxumQuery(params): AxumQuery<PageParams>,
) -> ApiResult<Json<Page<Chunk>>> {
    let id = parse_id(&id)?;
    Ok(Json(params.apply(state.nexum.chunks_of(id)?)))
}

// ---- Nodes ---------------------------------------------------------------

async fn get_node(
    State(state): State<ServerState>,
    Path(id): Path<String>,
) -> ApiResult<Json<NodeDetail>> {
    let id = parse_id(&id)?;
    state
        .nexum
        .show(id)?
        .map(Json)
        .ok_or_else(|| ApiError::NotFound(format!("no node {id}")))
}

#[derive(Deserialize)]
struct NeighborParams {
    /// Comma-separated edge types. Empty means every type.
    #[serde(default)]
    edges: Option<String>,
    #[serde(default)]
    direction: Option<String>,
}

async fn node_neighbors(
    State(state): State<ServerState>,
    Path(id): Path<String>,
    AxumQuery(params): AxumQuery<NeighborParams>,
) -> ApiResult<Json<Vec<Node>>> {
    let id = parse_id(&id)?;
    let edge_types = parse_edge_types(params.edges.as_deref())?;
    let direction = parse_direction(params.direction.as_deref())?;
    Ok(Json(state.nexum.neighbors(id, &edge_types, direction)?))
}

// ---- Search and traversal ------------------------------------------------

#[derive(Deserialize)]
struct SearchRequest {
    /// Text to embed and search for. Mutually exclusive with `vector`.
    #[serde(default)]
    query: Option<String>,
    /// A pre-computed query vector.
    #[serde(default)]
    vector: Option<Vec<f32>>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    latest_only: Option<bool>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    filter: Option<Predicate>,
    /// Expand results across the graph — the viewer's "expand via graph".
    #[serde(default)]
    expand: Option<Expansion>,
}

async fn search(
    State(state): State<ServerState>,
    Json(request): Json<SearchRequest>,
) -> ApiResult<Json<SearchResults>> {
    let mut options = SearchOptions {
        top_k: request.top_k.unwrap_or(10).clamp(1, 1_000),
        latest_only: request.latest_only.unwrap_or(true),
        model: request.model,
        filter: request.filter,
        expand: request.expand,
    };
    // Guard the expansion too: an unbounded hop count on a hub node is the
    // easiest way to make this endpoint hang.
    if let Some(expansion) = &mut options.expand {
        expansion.max_hops = expansion.max_hops.clamp(1, 6);
    }

    match (request.query, request.vector) {
        (Some(text), None) => Ok(Json(state.nexum.search(&text, options).await?)),
        (None, Some(vector)) => Ok(Json(state.nexum.search_vector(vector, options)?)),
        (Some(_), Some(_)) => Err(ApiError::BadRequest(
            "pass either `query` or `vector`, not both".into(),
        )),
        (None, None) => Err(ApiError::BadRequest(
            "pass `query` (text to embed) or `vector` (a pre-computed embedding)".into(),
        )),
    }
}

#[derive(Deserialize)]
struct TraverseRequest {
    start_ids: Vec<String>,
    #[serde(default)]
    edge_types: Vec<EdgeType>,
    #[serde(default = "default_hops")]
    max_hops: usize,
    #[serde(default)]
    direction: Direction,
}

fn default_hops() -> usize {
    2
}

async fn traverse(
    State(state): State<ServerState>,
    Json(request): Json<TraverseRequest>,
) -> ApiResult<Json<QueryResult>> {
    let ids: Vec<NodeId> = request
        .start_ids
        .iter()
        .map(|s| parse_id(s))
        .collect::<ApiResult<_>>()?;
    if ids.is_empty() {
        return Err(ApiError::BadRequest("start_ids must not be empty".into()));
    }
    Ok(Json(state.nexum.traverse(
        ids,
        request.edge_types,
        request.max_hops.clamp(1, 6),
        request.direction,
    )?))
}

async fn run_query(
    State(state): State<ServerState>,
    Json(query): Json<Query>,
) -> ApiResult<Json<QueryResult>> {
    if query.stages.is_empty() {
        return Err(ApiError::BadRequest(
            "a query needs at least one stage".into(),
        ));
    }
    Ok(Json(state.nexum.query(&query)?))
}

// ---- Ingestion -----------------------------------------------------------

#[derive(Deserialize)]
#[serde(untagged)]
enum IngestRequest {
    /// Ingest text supplied in the request.
    Text {
        source_uri: String,
        title: String,
        text: String,
    },
    /// Ingest a path on the server's filesystem.
    Path {
        path: String,
        #[serde(default)]
        recursive: bool,
    },
}

async fn ingest(
    State(state): State<ServerState>,
    Json(request): Json<IngestRequest>,
) -> ApiResult<Json<Vec<IngestReport>>> {
    match request {
        IngestRequest::Text {
            source_uri,
            title,
            text,
        } => Ok(Json(vec![
            state.nexum.ingest_text(source_uri, title, text).await?,
        ])),
        IngestRequest::Path { path, recursive } => Ok(Json(
            state
                .nexum
                .ingest(std::path::Path::new(&path), recursive)
                .await?,
        )),
    }
}

// ---- Graph explorer ------------------------------------------------------

/// A node as the graph view needs it: identity, a label, and nothing heavy.
#[derive(Serialize)]
struct GraphNode {
    id: NodeId,
    kind: NodeKind,
    label: String,
    /// Distance from the node the view is centred on.
    hops: usize,
}

#[derive(Serialize)]
struct GraphLink {
    source: NodeId,
    target: NodeId,
    edge_type: EdgeType,
    #[serde(rename = "class")]
    edge_class: nexum_core::EdgeClass,
}

#[derive(Serialize)]
struct GraphResponse {
    center: NodeId,
    nodes: Vec<GraphNode>,
    links: Vec<GraphLink>,
    /// True when the result hit `limit` and more neighbours exist.
    truncated: bool,
}

#[derive(Deserialize)]
struct GraphParams {
    #[serde(default = "default_graph_hops")]
    hops: usize,
    #[serde(default)]
    edges: Option<String>,
    #[serde(default = "default_graph_limit")]
    limit: usize,
}

fn default_graph_hops() -> usize {
    1
}
fn default_graph_limit() -> usize {
    250
}

/// The neighbourhood around one node, shaped for a force-directed layout.
async fn graph(
    State(state): State<ServerState>,
    Path(id): Path<String>,
    AxumQuery(params): AxumQuery<GraphParams>,
) -> ApiResult<Json<GraphResponse>> {
    let center = parse_id(&id)?;
    let edge_types = parse_edge_types(params.edges.as_deref())?;
    let hops = params.hops.clamp(1, 5);
    // A hub node can have thousands of neighbours; rendering them all would
    // lock the browser, so the cap is enforced here and reported honestly.
    let limit = params.limit.clamp(1, 2_000);

    let Some(root) = state.nexum.get(center)? else {
        return Err(ApiError::NotFound(format!("no node {center}")));
    };

    let expanded = state
        .nexum
        .db()
        .query(&Query::new().seed([center]).traverse_with(
            edge_types.clone(),
            hops,
            Direction::Both,
            true,
        ))?;

    let truncated = expanded.len() > limit;
    let mut nodes = vec![GraphNode {
        id: center,
        kind: root.kind(),
        label: root.label(),
        hops: 0,
    }];
    let mut included: Vec<NodeId> = vec![center];

    for item in expanded.nodes.iter().take(limit) {
        if item.id() == center {
            continue;
        }
        included.push(item.id());
        nodes.push(GraphNode {
            id: item.id(),
            kind: item.node.kind(),
            label: item.node.label(),
            hops: item.hops.unwrap_or(0),
        });
    }

    // Include every edge between nodes that made it into the view, not just
    // the ones the traversal walked — otherwise the layout looks like a tree
    // when the data is a graph.
    let view = state.nexum.db().read()?;
    let mut links = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for node_id in &included {
        for neighbor in view.neighbors(*node_id, &edge_types, Direction::Out)? {
            if !included.contains(&neighbor.target) {
                continue;
            }
            if !seen.insert((*node_id, neighbor.target, neighbor.edge_type)) {
                continue;
            }
            links.push(GraphLink {
                source: *node_id,
                target: neighbor.target,
                edge_type: neighbor.edge_type,
                edge_class: neighbor.edge_type.class(),
            });
        }
    }

    Ok(Json(GraphResponse {
        center,
        nodes,
        links,
        truncated,
    }))
}

// ---- Embedding projection ------------------------------------------------

#[derive(Deserialize)]
struct ProjectionParamsQuery {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    method: Option<ProjectionMethod>,
    #[serde(default = "default_projection_limit")]
    limit: usize,
    #[serde(default)]
    include_superseded: bool,
}

fn default_projection_limit() -> usize {
    2_000
}

/// A projected point with the metadata the scatter plot colours by.
#[derive(Serialize)]
struct ProjectionPoint {
    id: NodeId,
    x: f32,
    y: f32,
    document_id: NodeId,
    chunk_index: u32,
    preview: String,
}

#[derive(Serialize)]
struct ProjectionResponse {
    method: ProjectionMethod,
    namespace: String,
    points: Vec<ProjectionPoint>,
    #[serde(skip_serializing_if = "Option::is_none")]
    explained_variance: Option<f32>,
    dimensions: usize,
    truncated: bool,
}

async fn projection_route(
    State(state): State<ServerState>,
    AxumQuery(params): AxumQuery<ProjectionParamsQuery>,
) -> ApiResult<Json<ProjectionResponse>> {
    let namespace = state
        .nexum
        .db()
        .resolve_namespace(params.model.as_deref())?;

    let mut pipeline = Query::new().seed_kind(NodeKind::Chunk);
    if !params.include_superseded {
        pipeline = pipeline.latest();
    }
    let chunks = state.nexum.db().query(&pipeline)?;

    // Refinement is quadratic in point count, so the cap is a real constraint
    // rather than politeness. Report when it bites.
    let limit = params.limit.clamp(1, 20_000);
    let truncated = chunks.len() > limit;

    let view = state.nexum.db().read()?;
    let mut ids = Vec::new();
    let mut vectors = Vec::new();
    let mut meta = Vec::new();

    for item in chunks.nodes.iter().take(limit) {
        let Ok(chunk) = item.node.as_chunk() else {
            continue;
        };
        let Some(vector) = view.get_vector(&namespace, chunk.id)? else {
            continue;
        };
        ids.push(chunk.id);
        vectors.push(vector);
        meta.push((chunk.document_id, chunk.chunk_index, preview(&chunk.text)));
    }

    let method = params.method.unwrap_or_default();
    let projected = projection::project(&ids, &vectors, method, ProjectionParams::default());

    Ok(Json(ProjectionResponse {
        method: projected.method,
        namespace,
        explained_variance: projected.explained_variance,
        dimensions: projected.dimensions,
        truncated,
        points: projected
            .points
            .into_iter()
            .zip(meta)
            .map(
                |(point, (document_id, chunk_index, preview))| ProjectionPoint {
                    id: point.id,
                    x: point.x,
                    y: point.y,
                    document_id,
                    chunk_index,
                    preview,
                },
            )
            .collect(),
    }))
}

fn preview(text: &str) -> String {
    let preview: String = text.chars().take(120).collect();
    if text.chars().nth(120).is_some() {
        format!("{preview}…")
    } else {
        preview
    }
}

// ---- Parsing helpers -----------------------------------------------------

fn parse_id(raw: &str) -> ApiResult<NodeId> {
    NodeId::from_str(raw).map_err(|_| ApiError::BadRequest(format!("`{raw}` is not a node id")))
}

fn parse_edge_types(raw: Option<&str>) -> ApiResult<Vec<EdgeType>> {
    let Some(raw) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(Vec::new());
    };
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            EdgeType::from_str(s)
                .map_err(|_| ApiError::BadRequest(format!("`{s}` is not an edge type")))
        })
        .collect()
}

fn parse_direction(raw: Option<&str>) -> ApiResult<Direction> {
    match raw.map(str::trim).filter(|s| !s.is_empty()) {
        Some(raw) => Direction::from_str(raw)
            .map_err(|_| ApiError::BadRequest(format!("`{raw}` is not a direction"))),
        None => Ok(Direction::Both),
    }
}
