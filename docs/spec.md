# Specification

The original specification for this project is `nexumdb-spec.pdf`. Its content
is reproduced here so the repository is self-contained and the implementation
can be checked against it without the PDF to hand.

See `architecture.md` for where the implementation departs from this document
and why.

---

## 1. Problem statement

Existing RAG stacks bolt vector search onto flat key-value or table storage.
This loses structural relationships (chunk hierarchy, document versions),
semantic relationships (entities and how they relate), and provenance (what
pipeline/model produced an embedding, and when). Multi-hop queries require
stitching together a vector store and a separate graph store.

This project builds a single storage engine where vectors are a native property
type on graph nodes, so similarity search and graph traversal can be combined in
one query, with versioning and provenance as first-class citizens.

## 2. Core data model

**Node types**

- `Document` — a source file/artifact: `id`, `title`, `source_uri`,
  `content_hash`, `created_at`, `version`, `supersedes_id`
- `Chunk` — a segment of a document: `id`, `document_id`, `text`,
  `chunk_index`, `token_count`, `embedding`, `embedding_model`, `embedded_at`
- `Entity` — an extracted entity or concept: `id`, `name`, `type`,
  `canonical_id`, `embedding`
- `PipelineRun` — ingestion/embedding run metadata: `id`, `pipeline_version`,
  `embedding_model`, `run_at`, `config_hash`

**Edge types**

- Structural: `PART_OF`, `PRECEDES`/`FOLLOWS`, `SUPERSEDES`
- Semantic: `MENTIONS`, `RELATES_TO` (with `relation_type`), `SIMILAR_TO`
- Provenance: `DERIVED_FROM`, `EMBEDDED_BY`, `EXTRACTED_BY`

All edges carry `created_at` and are immutable once written.

**Versioning**

Documents are never mutated in place. A re-ingested document creates a new
`Document` node with `supersedes_id` pointing to the prior version. Queries
default to latest-version-only. Chunks of superseded documents remain queryable
for audit but are excluded from default retrieval.

## 3. Storage engine

Build on an embedded key-value layer rather than raw disk I/O. Node store keyed
by node id; adjacency store keyed by `(node_id, edge_type, direction)`; HNSW
vector index per embedding dimension/model, stored separately. Write path is an
append-only WAL applied to the stores, which also gives the audit trail.

**Query engine** — three composable primitives:

1. `search(embedding, top_k, filter?)` → ranked node ids via HNSW
2. `traverse(start_ids, edge_types[], max_hops, direction)` → reachable ids
3. `filter(node_ids, predicate)` → predicate filtering

A query is a pipeline of these, e.g. `search(query_vec, top_k=20) →
traverse(edge_types=[MENTIONS, RELATES_TO], max_hops=2) → filter(latest)`.

**Query language** — define a small declarative language (GQL-lite), or expose
the pipeline as a fluent API first and treat a text DSL as a v2 nice-to-have.
The fluent API is recommended as faster to ship.

## 4. Ingestion pipeline

1. **Chunking** — pluggable strategy (fixed-size, semantic, recursive), with
   strategy and params recorded in `PipelineRun`
2. **Embedding** — pluggable provider, model name/version recorded
3. **Entity extraction** — pluggable NER/relation extraction, writing `Entity`
   nodes and `MENTIONS`/`RELATES_TO` edges
4. **Dedup** — canonicalize entities by name+type similarity

## 5. Client library / API

```
client.ingest(document)                   -> Document
client.search(query_text, top_k, filters?) -> [Chunk]
client.traverse(node_ids, edge_types, max_hops) -> [Node]
client.get_history(document_id)           -> [Document versions]
client.get_neighbors(node_id, edge_type?) -> [Node]
```

## 6. CLI

Binary name `nexum`. `--json` on every read command.

```
nexum init <path>
nexum ingest <file-or-dir> [--chunker=X] [--embedder=X]
nexum search "<query>" [--top-k=10] [--json]
nexum traverse <node-id> --edges=MENTIONS,RELATES_TO --hops=2
nexum show <node-id>
nexum stats
nexum history <document-id>
nexum export <path> --format=jsonl
nexum serve --port=8080
```

## 7. Desktop viewer

**Stack** — Tauri + React, talking to the engine via the `nexum serve` HTTP API.
Use frontend-design conventions for visual polish rather than default
component-library styling.

**Views** — collection browser; chunk table; graph explorer (node-link,
expandable by edge type, hop-depth control); embedding projection (UMAP/t-SNE
2D scatter coloured by document or cluster); search sandbox (ranked results,
then "expand via graph"); version/history view (supersession timeline with
chunk-level diffs); provenance inspector (full `PipelineRun` lineage).

**Data access** — the viewer talks to the engine exclusively through the same
API as the CLI, never touching storage files directly. This keeps CLI, viewer,
and future integrations consistent and prevents corruption from concurrent
writers.

## 8. Suggested tech

Rust core engine; `hnswlib-rs` or `instant-distance` for vectors; RocksDB for
storage; protobuf or flatbuffers for serialization; gRPC or plain HTTP/JSON for
the server; Tauri + React + TypeScript for the viewer with `react-force-graph`
and a WebGL scatter.

## 9. Build phases

1. Engine core — node/adjacency/vector storage, WAL, ingest/search/traverse
2. CLI — the `nexum` binary, all commands except `serve`
3. Server + versioning — `nexum serve`, supersession, provenance
4. Viewer app — Tauri shell, collection browser, chunk table, search sandbox
5. Graph explorer + embedding projection
6. Entity extraction pipeline + dedup

## 10. Resolved design decisions

- **Multiple embedding models**: supported via per-`(model, dimension)` index
  namespacing. A chunk can hold several embeddings, each tagged by its own
  `EMBEDDED_BY` edge to the run that produced it. Locking to one model would
  force a full re-index on every model upgrade.
- **Concurrency**: single-writer/multi-reader for v1. Dramatically simpler to
  get correct, and nothing in batch ingestion needs concurrent writes.
- **Multi-tenancy**: out of scope for v1 — one database per project. If needed
  later, add it as a thin layer rather than designing storage around it.
