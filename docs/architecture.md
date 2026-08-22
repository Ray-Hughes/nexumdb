# NexumDB architecture

This documents what was built, and — more usefully — where it departs from the
specification and why. Every deviation below was a deliberate trade, not an
oversight.

## Shape

```
crates/
  nexum-core     storage engine: model, tables, WAL, HNSW, query pipeline
  nexum-embed    embedding providers (local ONNX, OpenAI, Ollama, hash)
  nexum-ingest   chunking, entity extraction, dedup, versioned writes
  nexum-client   the fluent API the CLI and server both use
  nexum-server   HTTP API (axum)
  nexum-cli      the `nexum` binary
apps/
  viewer         Tauri + React desktop app
```

The dependency direction is strictly downward. `nexum-core` knows nothing about
embeddings, HTTP, or the CLI; the CLI and the server are both thin wrappers over
`nexum-client`, which is why a capability in one exists in the other.

## Deviations from the spec

### redb instead of RocksDB

The spec recommends RocksDB. RocksDB is a C++ dependency: it needs a working
MSVC or clang toolchain on every build host, adds minutes to a cold compile, and
is a recurring source of Windows CI breakage. Since first-class Windows, macOS,
and Linux support was a requirement, that cost lands on every platform, every
build.

redb is a pure-Rust embedded key-value store with MVCC and a
single-writer/multi-reader model — which is exactly the concurrency model the
spec resolves on in section 10. The table layout is the one the spec describes:
a node store, an adjacency index keyed by `(node, direction, edge_type)`, and a
separate vector store.

Everything reaches storage through `store.rs`, so swapping the backend means
reimplementing one file. If benchmarks later show redb is the bottleneck,
RocksDB can be reintroduced behind the same interface.

### bincode instead of protobuf or flatbuffers

Both suggested formats need a schema compiler in the build. bincode is compact,
fast, and pure Rust with no codegen step. The wire format for clients is JSON
either way, so this is purely an on-disk choice.

It has one sharp edge, and it cost real debugging time: **bincode is not
self-describing**. Two serde features that work fine in JSON silently break
binary decoding — `#[serde(tag = "...")]` internally tagged enums, and
`skip_serializing_if`, which omits a field the decoder still expects to read.
`Node` and `PropertyValue` therefore carry explicit dual representations that
branch on `is_human_readable()`: natural JSON on the wire, tagged encoding on
disk. `model.rs` has tests pinning both, because this class of bug produces a
database that writes fine and cannot be read back.

### HNSW implemented in-tree

The spec offers `hnswlib-rs` / `instant-distance` or an implementation of the
Malkov & Yashunin paper. This implements the paper, because the database needs
three things no off-the-shelf crate offered together:

- **Incremental insertion** — documents arrive over time, not in one batch.
- **Tombstone deletion** — superseded versions leave the live set.
- **Filtered search** — the default query is "latest version only", and
  applying that filter *after* a top-k search silently returns too few rows.
  The implementation traverses through filtered-out nodes (they are often the
  only path to qualifying ones) and widens `ef` when a selective filter starves
  the result list.

Level assignment is derived from the node ID rather than an RNG, so rebuilding
an index from the same data produces the same graph. Recall against exhaustive
search is asserted in the test suite.

### Fluent API, no text query language

The spec proposes GQL-lite as a possible v1 and recommends the fluent API as
the faster path. This ships the fluent pipeline only. A pipeline is what a CLI
and a UI can both construct programmatically; a text syntax designed before the
pipeline stabilised would only have to be redesigned afterwards. `Query` is
serde-serialisable, so `POST /api/query` already accepts a pipeline as JSON —
a text DSL would compile to exactly that.

### PCA, not UMAP or t-SNE

The projection view ships PCA by power iteration, optionally followed by a
neighbour-embedding refinement that pulls each point toward its nearest
neighbours and pushes it off random others — the core of what t-SNE and UMAP
optimise, without the dependency.

It is *not* UMAP, so the API response names the method it actually used rather
than implying one that did not run. Both modes are deterministic: a scatter
plot that reshuffles on every refresh is worse than a simpler one that does not.

## Design decisions worth knowing

### Commit ordering

Writes go: **append to the log and fsync → apply to the tables → publish to the
in-memory index**. The tables record the highest log sequence number they have
absorbed, and opening the database replays anything past that watermark.

redb is already transactional, so the log is not strictly required for crash
safety. It earns its place as the durable audit trail the spec asks for, and
recovery falls out of it for free. A failed table commit rebuilds the in-memory
indexes rather than leaving them ahead of durable state.

HNSW graphs are flushed periodically rather than on every commit, because
serialising a whole graph per write is expensive. A stale snapshot is detected
on open and rebuilt from the vector table, which stays authoritative.

### Versioning

A document is current when nothing supersedes it. The `SUPERSEDES` edge points
from the new version to the old one, so "no inbound `SUPERSEDES`" *is* the
definition of current — derived from the graph rather than trusted from a
version number that could be wrong.

Superseded chunks stay indexed and queryable. They are excluded from default
retrieval by the `LatestVersion` predicate, not deleted, so audit and history
still work.

### Entity dedup is two layers

Exact dedup is free: an entity's node ID is derived from its normalised name
and type, so the same entity mentioned across fifty documents lands on one node
with no merge pass at all. That layer does the real work.

Fuzzy dedup handles the residue — "Ada Lovelace" versus "A. Lovelace" — via
Jaro-Winkler, and **links** with `canonical_id` rather than merging. A wrong
fold stays visible and reversible.

### The viewer never touches storage

The desktop app starts the same HTTP server `nexum serve` runs, on a loopback
port the OS assigns, and reaches the engine only through it. There is one API
surface to keep correct, and no path by which the viewer could become a second
writer against a single-writer database.

The corollary: while `nexum serve` or the viewer holds a database, a second
process cannot open it. That is the single-writer model working as designed,
and the error message says so.

## Known limitations

- **Entity extraction is rule-based.** It recognises named things and guesses
  their type from surface form. It is genuinely weaker than a trained NER
  model, reports honest confidence scores, and `EntityExtractor` is the seam
  where a spaCy service or an LLM extractor slots in.
- **Token counts are estimates.** Chunk sizes are in characters, which are
  exact and model-agnostic; real tokenisation depends on which tokenizer you
  ask.
- **No multi-tenancy, no auth.** Out of scope per the spec. The server binds
  loopback by default for that reason.
- **The CLI cannot talk to a remote server.** Read commands open the database
  directly, so they conflict with a running `serve`. Adding `--remote <url>`
  would close this and is the natural next step.
- **`SIMILAR_TO` edges are modelled but not populated.** The spec describes them
  as an optional precomputed cache; nothing writes them yet.
