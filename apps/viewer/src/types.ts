/**
 * Types mirroring the engine's JSON.
 *
 * Hand-written rather than generated: the API surface is small and stable, and
 * a generator would add a build step to a project whose whole build story is
 * "cargo build, npm build". Anything that drifts shows up immediately in the
 * views, because nothing here is `any`.
 */

export type NodeKind = "Document" | "Chunk" | "Entity" | "PipelineRun";

export type EdgeType =
  | "PART_OF"
  | "PRECEDES"
  | "FOLLOWS"
  | "SUPERSEDES"
  | "MENTIONS"
  | "RELATES_TO"
  | "SIMILAR_TO"
  | "DERIVED_FROM"
  | "EMBEDDED_BY"
  | "EXTRACTED_BY";

export type EdgeClass = "structural" | "semantic" | "provenance";

export type Direction = "out" | "in" | "both";

/** Milliseconds since the Unix epoch. */
export type Timestamp = number;

export type Metadata = Record<string, unknown>;

export interface EmbeddingMeta {
  model: string;
  dim: number;
  run_id: string | null;
  embedded_at: Timestamp;
}

export interface DocumentNode {
  kind: "Document";
  id: string;
  title: string;
  source_uri: string;
  content_hash: string;
  created_at: Timestamp;
  version: number;
  supersedes_id: string | null;
  run_id: string | null;
  metadata: Metadata;
}

export interface ChunkNode {
  kind: "Chunk";
  id: string;
  document_id: string;
  text: string;
  chunk_index: number;
  token_count: number;
  embeddings: EmbeddingMeta[];
  created_at: Timestamp;
  metadata: Metadata;
}

export interface EntityNode {
  kind: "Entity";
  id: string;
  name: string;
  entity_type: string;
  canonical_id: string | null;
  embeddings: EmbeddingMeta[];
  created_at: Timestamp;
  metadata: Metadata;
}

export interface PipelineRunNode {
  kind: "PipelineRun";
  id: string;
  pipeline_version: string;
  embedding_model: string;
  run_at: Timestamp;
  config_hash: string;
  chunker: string;
  metadata: Metadata;
}

export type GraphNodeRecord =
  | DocumentNode
  | ChunkNode
  | EntityNode
  | PipelineRunNode;

export interface Edge {
  from: string;
  to: string;
  edge_type: EdgeType;
  created_at: Timestamp;
  properties: Metadata;
}

export interface EdgeWithTarget {
  edge: Edge;
  other_id: string;
  other_kind: NodeKind | null;
  other_label: string | null;
}

export interface NodeDetail {
  node: GraphNodeRecord;
  outgoing: EdgeWithTarget[];
  incoming: EdgeWithTarget[];
}

export interface ResultNode {
  node: GraphNodeRecord;
  /** Similarity in 0..1. Absent for nodes reached by traversal. */
  score?: number | null;
  distance?: number | null;
  /** Shortest-path distance from the seed set. Absent for direct hits. */
  hops?: number | null;
  via?: string | null;
  via_edge?: EdgeType | null;
}

export interface StageStats {
  stage: string;
  in_count: number;
  out_count: number;
}

export interface QueryStats {
  stages: StageStats[];
  nodes_visited: number;
  edges_traversed: number;
}

export interface SearchResults {
  query_model: string;
  results: ResultNode[];
  stats: QueryStats;
}

export interface QueryResult {
  nodes: ResultNode[];
  stats: QueryStats;
}

export interface NamespaceInfo {
  model: string;
  dim: number;
  count: number;
}

export interface DbStats {
  path: string;
  created_at: Timestamp;
  documents: number;
  latest_documents: number;
  chunks: number;
  entities: number;
  pipeline_runs: number;
  edges: number;
  edges_by_type: Record<string, number>;
  tombstones: number;
  namespaces: Record<string, NamespaceInfo>;
  store_bytes: number;
  wal_bytes: number;
  applied_lsn: number;
}

export interface ServerConfigInfo {
  embedding_model: string;
  embedding_dimensions: number;
  embedder: string;
  namespaces: string[];
  engine_version: string;
}

export interface DocumentSummary extends DocumentNode {
  chunk_count: number;
  is_latest: boolean;
}

export interface Page<T> {
  items: T[];
  total: number;
  offset: number;
  limit: number;
}

export interface GraphViewNode {
  id: string;
  kind: NodeKind;
  label: string;
  hops: number;
}

export interface GraphViewLink {
  source: string;
  target: string;
  edge_type: EdgeType;
  class: EdgeClass;
}

export interface GraphView {
  center: string;
  nodes: GraphViewNode[];
  links: GraphViewLink[];
  truncated: boolean;
}

export type ProjectionMethod = "pca" | "neighborhood";

export interface ProjectionPoint {
  id: string;
  x: number;
  y: number;
  document_id: string;
  chunk_index: number;
  preview: string;
}

export interface ProjectionResponse {
  method: ProjectionMethod;
  namespace: string;
  points: ProjectionPoint[];
  explained_variance?: number;
  dimensions: number;
  truncated: boolean;
}

export type IngestOutcomeKind =
  | "created"
  | "superseded"
  | "unchanged"
  | "skipped";

export interface IngestReport {
  source_uri: string;
  title: string;
  outcome: IngestOutcomeKind;
  version?: number;
  supersedes?: string;
  existing?: string;
  reason?: string;
  document_id?: string;
  run_id?: string;
  chunks: number;
  entities: number;
  edges: number;
  aliases: number;
  embedding_model?: string;
  duration_ms: number;
}

/** Where the app is connected. */
export interface ApiInfo {
  base_url: string;
  database: string;
  embedding_model: string;
  embedding_dimensions: number;
  engine_version: string;
}
