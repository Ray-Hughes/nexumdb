# NexumDB

A graph-native vector database for RAG, where vectors are a property type on
graph nodes rather than a separate store bolted alongside one.

Most RAG stacks put embeddings in one system and everything else — document
structure, entity relationships, which model produced which vector — somewhere
else, or nowhere. A query like *"find chunks near this, then follow their
citation edges, but only in current document versions"* becomes application code
stitching two databases together.

NexumDB is one engine where that is a single query.

```
search(query_vector, top_k=20)
  → traverse([MENTIONS, RELATES_TO], max_hops=2)
  → filter(latest_version)
```

Runs on macOS, Windows, and Linux as an embedded engine, a CLI, and a desktop
app.

## What makes it different

**Versioning is structural, not a column.** Re-ingesting a document mints a new
version and writes a `SUPERSEDES` edge from new to old. "Current" means *nothing
supersedes this* — derived from the graph, not trusted from a number. Old
versions stay indexed and queryable for audit; they are just excluded from
retrieval by default.

**Provenance is written at ingest, not reconstructed.** Every chunk points at
the `PipelineRun` that embedded it, carrying the model, the chunker, and a hash
over the full effective config. Two embeddings are comparable evidence only if
they share that hash — and you can check.

**Several embedding models coexist.** Vector indexes are namespaced by
`(model, dimension)`. Upgrading your embedding model adds an index; it does not
invalidate the old one or force a full re-index.

**Search and traversal compose.** Results carry how they were reached —
similarity score, shortest-path hop count, and the edge they arrived on — so
"expand via graph" is explainable rather than magic.

## Quick start

```bash
cargo build --release
./target/release/nexum init ./mydb
./target/release/nexum ingest ./docs
./target/release/nexum search "what changed in the Q3 report"
```

The default embedder runs locally with no API key — the first ingest downloads
a small sentence-transformer into a cache directory. `--embedder hash` skips
even that (lexical matching only, useful for testing), and `openai:` and
`ollama:` providers are available for real workloads.

### The desktop app

```bash
cd apps/viewer
npm install
npm run app          # development
npm run app:build    # produce an installer
```

The viewer starts the same HTTP server the CLI does, on a loopback port, and
reaches the engine only through it. It never touches storage files.

### As a library

```rust
use nexum_client::{ClientConfig, Nexum, SearchOptions};

let db = Nexum::open("./mydb", ClientConfig::default()).await?;
db.ingest("./docs", true).await?;

let hits = db.search("multi-hop retrieval", SearchOptions::default().top_k(10)).await?;
for hit in hits.results {
    println!("{:.3}  {}", hit.score.unwrap_or(0.0), hit.node.label());
}
```

## CLI

Every read command takes `--json`.

| Command | What it does |
| --- | --- |
| `nexum init <path>` | Create a database |
| `nexum ingest <path>` | Ingest a file or directory tree |
| `nexum search "<query>"` | Vector search, optionally `--expand` across the graph |
| `nexum traverse <node>` | Walk outward by edge type and hop count |
| `nexum show <node>` | A node's properties, edges, and provenance |
| `nexum docs` | List documents with versions and chunk counts |
| `nexum history <doc>` | The supersession chain |
| `nexum stats` | Counts, index sizes, embedding models |
| `nexum export <path>` | JSONL dump for backup or inspection |
| `nexum compact` | Rebuild indexes, truncate the log, reclaim space |
| `nexum serve` | HTTP API for the viewer and remote clients |

Node IDs accept a unique prefix or the short form the CLI prints.

## Data model

Four node types — `Document`, `Chunk`, `Entity`, `PipelineRun` — and ten edge
types in three families:

- **Structural**: `PART_OF`, `PRECEDES`/`FOLLOWS`, `SUPERSEDES`
- **Semantic**: `MENTIONS`, `RELATES_TO`, `SIMILAR_TO`
- **Provenance**: `DERIVED_FROM`, `EMBEDDED_BY`, `EXTRACTED_BY`

Edges are immutable once written, so history can always be reconstructed.

## Layout

```
crates/nexum-core     storage engine: model, tables, WAL, HNSW, query pipeline
crates/nexum-embed    embedding providers
crates/nexum-ingest   chunking, entity extraction, dedup, versioned writes
crates/nexum-client   the API the CLI and server both use
crates/nexum-server   HTTP API
crates/nexum-cli      the `nexum` binary
apps/viewer           Tauri + React desktop app
```

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — how it works, what departs
  from the spec and why, and the known limitations
- [`docs/spec.md`](docs/spec.md) — the specification this implements

## Status

All six build phases are implemented. Storage is redb rather than RocksDB and
serialization is bincode rather than protobuf — both chosen so Windows and Linux
builds need no C++ toolchain; see `docs/architecture.md` for the reasoning.

## Licence

Apache-2.0
