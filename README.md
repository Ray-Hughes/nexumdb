# NexumDB

A graph-native vector database for RAG, where vectors are a property type on
graph nodes rather than a separate store bolted alongside one. Similarity
search and graph traversal compose in a single query, and versioning and
provenance are first-class rather than bookkeeping you bolt on later.

Ships as an embedded engine, a `nexum` CLI, and a desktop viewer — on macOS,
Windows, and Linux.

Status: in development. See `docs/architecture.md` for design notes and
`docs/spec.md` for the specification this implements.
