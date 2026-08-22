//! Command implementations.
//!
//! Each command renders the same data twice: once for a terminal and once as
//! JSON. The JSON is the client-library type serialised directly, so it stays
//! in step with the HTTP API without a second set of DTOs to keep aligned.

use crate::resolve;
use crate::style;
use anyhow::{Context, Result};
use nexum_client::{ClientConfig, Expansion, Nexum, SearchOptions};
use nexum_core::{Direction, EdgeType, Node, NodeKind};
use nexum_embed::EmbedderConfig;
use nexum_ingest::IngestOutcome;
use serde::Serialize;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

/// Print a value as pretty JSON.
fn emit_json<T: Serialize>(value: &T) -> Result<()> {
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer_pretty(&mut stdout, value)?;
    writeln!(stdout)?;
    Ok(())
}

// ---- init ----------------------------------------------------------------

pub async fn init(path: &Path, embedder_spec: &str, json: bool) -> Result<()> {
    let embedder = EmbedderConfig::parse(embedder_spec)
        .with_context(|| format!("invalid --embedder `{embedder_spec}`"))?;

    if path.join("config.json").exists() {
        anyhow::bail!("a database already exists at {}", path.display());
    }

    let config = ClientConfig::default().with_embedder(embedder.clone());
    let nexum = Nexum::create(path, config)
        .await
        .with_context(|| format!("could not create database at {}", path.display()))?;

    // Record the embedder alongside the database so later commands search the
    // same vector space they ingested into, without the user restating it.
    std::fs::write(
        path.join("embedder.json"),
        serde_json::to_vec_pretty(&embedder)?,
    )?;

    if json {
        return emit_json(&serde_json::json!({
            "path": nexum.path().display().to_string(),
            "embedder": embedder,
            "embedding_model": nexum.embedder().model_id(),
            "dimensions": nexum.embedder().dim(),
        }));
    }

    println!(
        "{} database at {}",
        style::green("Created"),
        style::bold(&nexum.path().display().to_string())
    );
    println!(
        "  {}  {}",
        style::dim("embedder"),
        nexum.embedder().describe()
    );
    println!();
    println!(
        "Next: {}",
        style::cyan(&format!("nexum ingest <path> --db {}", path.display()))
    );
    Ok(())
}

// ---- ingest --------------------------------------------------------------

pub async fn ingest(nexum: &Nexum, path: &Path, recursive: bool, json: bool) -> Result<()> {
    let reports = nexum
        .ingest(path, recursive)
        .await
        .with_context(|| format!("could not ingest {}", path.display()))?;
    nexum.flush()?;

    if json {
        return emit_json(&reports);
    }

    if reports.is_empty() {
        println!("{}", style::yellow("Nothing to ingest."));
        return Ok(());
    }

    let mut chunks = 0usize;
    let mut entities = 0usize;
    let mut written = 0usize;
    let mut unchanged = 0usize;
    let mut skipped = 0usize;

    for report in &reports {
        let (marker, detail) = match &report.outcome {
            IngestOutcome::Created { version } => {
                written += 1;
                (style::green("+"), format!("v{version}"))
            }
            IngestOutcome::Superseded { version, .. } => {
                written += 1;
                (
                    style::cyan("^"),
                    format!("v{version}, superseded v{}", version - 1),
                )
            }
            IngestOutcome::Unchanged { .. } => {
                unchanged += 1;
                (style::dim("="), "unchanged".to_string())
            }
            IngestOutcome::Skipped { reason } => {
                skipped += 1;
                (style::yellow("!"), reason.clone())
            }
        };
        chunks += report.chunks;
        entities += report.entities;

        println!(
            "{marker} {:<44} {}",
            style::truncate(&report.title, 44),
            style::dim(&detail)
        );
    }

    println!();
    println!(
        "{} {written} written, {unchanged} unchanged, {skipped} skipped — {chunks} chunks, {entities} entities",
        style::bold("Ingested")
    );
    if written > 0 {
        println!("  {}  {}", style::dim("model"), nexum.embedder().model_id());
    }
    Ok(())
}

// ---- search --------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub async fn search(
    nexum: &Nexum,
    query: &str,
    top_k: usize,
    expand: Option<&[String]>,
    hops: usize,
    include_superseded: bool,
    model: Option<String>,
    json: bool,
) -> Result<()> {
    let mut options = SearchOptions::default().top_k(top_k);
    options.model = model;
    if include_superseded {
        options = options.include_superseded();
    }
    if let Some(edges) = expand {
        options = options.expand(Expansion {
            edge_types: resolve::edge_types(Some(edges))?,
            max_hops: hops,
            direction: Direction::Both,
        });
    }

    let results = nexum.search(query, options).await?;

    if json {
        return emit_json(&results);
    }

    if results.is_empty() {
        println!("{}", style::yellow("No matches."));
        if !include_superseded {
            println!(
                "  {}",
                style::dim(
                    "only current document versions were searched; --include-superseded widens it"
                )
            );
        }
        return Ok(());
    }

    println!(
        "{} for {} {}",
        style::bold(&format!("{} results", results.len())),
        style::cyan(&format!("\"{query}\"")),
        style::dim(&format!("via {}", results.query_model))
    );
    println!();

    for (rank, item) in results.results.iter().enumerate() {
        let score = match item.score {
            Some(score) => format!("{score:.3}"),
            // Nodes reached by graph expansion have no similarity score;
            // showing 0.000 would read as "very poor match" rather than
            // "arrived a different way".
            None => match item.hops {
                Some(hops) => format!("+{hops}hop"),
                None => "—".to_string(),
            },
        };

        println!(
            "{:>3}. {} {} {}",
            rank + 1,
            style::bold(&score),
            style::kind(item.node.kind()),
            style::dim(&style::short_id(item.id()))
        );

        match &item.node {
            Node::Chunk(chunk) => {
                println!("     {}", style::truncate(&chunk.text, 100));
                println!(
                    "     {}",
                    style::dim(&format!(
                        "chunk {} of document {}",
                        chunk.chunk_index,
                        style::short_id(chunk.document_id)
                    ))
                );
            }
            other => println!("     {}", style::truncate(&other.label(), 100)),
        }

        if let (Some(via), Some(edge)) = (item.via, item.via_edge) {
            println!(
                "     {} {} {}",
                style::dim("reached via"),
                style::edge(edge),
                style::dim(&format!("from {}", style::short_id(via)))
            );
        }
        println!();
    }
    Ok(())
}

// ---- traverse ------------------------------------------------------------

pub fn traverse(
    nexum: &Nexum,
    node_ref: &str,
    edges: Option<&[String]>,
    hops: usize,
    direction: &str,
    json: bool,
) -> Result<()> {
    let start = resolve::node_id(nexum, node_ref)?;
    let edge_types = resolve::edge_types(edges)?;
    let direction = resolve::direction(direction)?;

    let result = nexum.traverse([start], edge_types, hops, direction)?;

    if json {
        return emit_json(&result);
    }

    if result.is_empty() {
        println!("{}", style::yellow("No nodes reachable."));
        return Ok(());
    }

    println!(
        "{} within {hops} hop{} of {}",
        style::bold(&format!("{} nodes", result.len())),
        if hops == 1 { "" } else { "s" },
        style::dim(&style::short_id(start))
    );
    println!();

    // Grouping by distance makes the shape of the neighbourhood legible in a
    // way a flat list does not.
    let max_hops = result
        .nodes
        .iter()
        .filter_map(|n| n.hops)
        .max()
        .unwrap_or(0);
    for hop in 1..=max_hops {
        let at_hop: Vec<_> = result
            .nodes
            .iter()
            .filter(|n| n.hops == Some(hop))
            .collect();
        if at_hop.is_empty() {
            continue;
        }
        println!(
            "{}",
            style::bold(&format!("{hop} hop{}", if hop == 1 { "" } else { "s" }))
        );
        for item in at_hop {
            println!(
                "  {} {} {} {}",
                style::dim(&style::short_id(item.id())),
                style::kind(item.node.kind()),
                item.via_edge.map(style::edge).unwrap_or_default(),
                style::truncate(&item.node.label(), 60)
            );
        }
        println!();
    }

    println!(
        "{}",
        style::dim(&format!(
            "visited {} nodes across {} edges",
            result.stats.nodes_visited, result.stats.edges_traversed
        ))
    );
    Ok(())
}

// ---- show ----------------------------------------------------------------

pub fn show(nexum: &Nexum, node_ref: &str, full: bool, json: bool) -> Result<()> {
    let id = resolve::node_id(nexum, node_ref)?;
    let detail = nexum
        .show(id)?
        .ok_or_else(|| anyhow::anyhow!("no node {id}"))?;

    if json {
        return emit_json(&detail);
    }

    println!(
        "{} {}",
        style::kind(detail.node.kind()),
        style::bold(&id.to_string())
    );
    println!();

    let field = |name: &str, value: &str| println!("  {:<16} {value}", style::dim(name));

    match &detail.node {
        Node::Document(doc) => {
            field("title", &doc.title);
            field("source", &doc.source_uri);
            field("version", &doc.version.to_string());
            field("created", &doc.created_at.to_rfc3339());
            field("content hash", &doc.content_hash.0[..16]);
            if let Some(prior) = doc.supersedes_id {
                field("supersedes", &prior.to_string());
            }
        }
        Node::Chunk(chunk) => {
            field("document", &chunk.document_id.to_string());
            field("index", &chunk.chunk_index.to_string());
            field("tokens (est.)", &chunk.token_count.to_string());
            for embedding in &chunk.embeddings {
                field(
                    "embedding",
                    &format!(
                        "{} ({}d) at {}",
                        embedding.model,
                        embedding.dim,
                        embedding.embedded_at.to_rfc3339()
                    ),
                );
            }
            println!();
            println!("  {}", style::dim("text"));
            let text = if full {
                chunk.text.clone()
            } else {
                style::truncate(&chunk.text, 500)
            };
            for line in text.lines() {
                println!("    {line}");
            }
        }
        Node::Entity(entity) => {
            field("name", &entity.name);
            field("type", &entity.entity_type);
            if let Some(canonical) = entity.canonical_id {
                field("alias of", &canonical.to_string());
            }
        }
        Node::PipelineRun(run) => {
            field("pipeline", &run.pipeline_version);
            field("model", &run.embedding_model);
            field("chunker", &run.chunker);
            field("run at", &run.run_at.to_rfc3339());
            field("config hash", &run.config_hash.0[..16]);
        }
    }

    for (label, edges) in [
        ("outgoing", &detail.outgoing),
        ("incoming", &detail.incoming),
    ] {
        if edges.is_empty() {
            continue;
        }
        println!();
        println!("{} ({})", style::bold(label), edges.len());
        for edge in edges {
            let properties = if edge.edge.properties.is_empty() {
                String::new()
            } else {
                let rendered = edge
                    .edge
                    .properties
                    .iter()
                    .map(|(k, v)| format!("{k}={}", v.to_display_string()))
                    .collect::<Vec<_>>()
                    .join(" ");
                format!("  {}", style::dim(&rendered))
            };
            println!(
                "  {} {} {} {}{properties}",
                style::edge(edge.edge.edge_type),
                style::dim(&style::short_id(edge.other_id)),
                edge.other_kind.map(style::kind).unwrap_or_default(),
                style::truncate(edge.other_label.as_deref().unwrap_or(""), 40),
            );
        }
    }

    if !detail.node.metadata().is_empty() {
        println!();
        println!("{}", style::bold("metadata"));
        for (key, value) in detail.node.metadata() {
            println!("  {:<16} {}", style::dim(key), value.to_display_string());
        }
    }
    Ok(())
}

// ---- docs ----------------------------------------------------------------

pub fn docs(nexum: &Nexum, all: bool, limit: usize, json: bool) -> Result<()> {
    let mut documents = nexum.documents(all)?;
    documents.sort_by_key(|d| std::cmp::Reverse(d.created_at));

    if json {
        return emit_json(&documents);
    }

    if documents.is_empty() {
        println!("{}", style::yellow("No documents yet."));
        println!("  {}", style::dim("ingest some with `nexum ingest <path>`"));
        return Ok(());
    }

    println!(
        "{:<10} {:<40} {:>4} {:>7}  {}",
        style::dim("ID"),
        style::dim("TITLE"),
        style::dim("VER"),
        style::dim("CHUNKS"),
        style::dim("SOURCE")
    );

    for document in documents.iter().take(limit) {
        let chunks = nexum.chunks_of(document.id)?.len();
        println!(
            "{:<10} {:<40} {:>4} {:>7}  {}",
            style::short_id(document.id),
            style::truncate(&document.title, 40),
            document.version,
            chunks,
            style::dim(&style::truncate(&document.source_uri, 50))
        );
    }

    if documents.len() > limit {
        println!();
        println!(
            "{}",
            style::dim(&format!(
                "showing {limit} of {} — raise with --limit",
                documents.len()
            ))
        );
    }
    Ok(())
}

// ---- stats ---------------------------------------------------------------

pub fn stats(nexum: &Nexum, json: bool) -> Result<()> {
    let stats = nexum.stats()?;

    if json {
        return emit_json(&stats);
    }

    println!("{}", style::bold(&stats.path));
    println!(
        "  {}  {}",
        style::dim("created"),
        stats.created_at.to_rfc3339()
    );
    println!();

    println!("{}", style::bold("nodes"));
    println!(
        "  {:<14} {:>8}  {}",
        style::kind(NodeKind::Document),
        stats.documents,
        style::dim(&format!("{} current", stats.latest_documents))
    );
    println!("  {:<14} {:>8}", style::kind(NodeKind::Chunk), stats.chunks);
    println!(
        "  {:<14} {:>8}",
        style::kind(NodeKind::Entity),
        stats.entities
    );
    println!(
        "  {:<14} {:>8}",
        style::kind(NodeKind::PipelineRun),
        stats.pipeline_runs
    );
    if stats.tombstones > 0 {
        println!("  {:<14} {:>8}", style::dim("deleted"), stats.tombstones);
    }

    println!();
    println!("{} ({})", style::bold("edges"), stats.edges);
    for edge_type in EdgeType::ALL {
        let count = stats
            .edges_by_type
            .get(edge_type.as_str())
            .copied()
            .unwrap_or(0);
        if count > 0 {
            println!("  {:<24} {:>8}", style::edge(edge_type), count);
        }
    }

    println!();
    println!("{}", style::bold("embeddings"));
    if stats.namespaces.is_empty() {
        println!("  {}", style::dim("none yet"));
    } else {
        for (namespace, info) in &stats.namespaces {
            println!(
                "  {:<32} {:>8} vectors  {}",
                namespace,
                info.count,
                style::dim(&format!("{}d", info.dim))
            );
        }
    }

    println!();
    println!("{}", style::bold("storage"));
    println!(
        "  {:<14} {}",
        style::dim("store"),
        style::bytes(stats.store_bytes)
    );
    println!(
        "  {:<14} {}",
        style::dim("log"),
        style::bytes(stats.wal_bytes)
    );
    println!("  {:<14} {}", style::dim("log position"), stats.applied_lsn);
    Ok(())
}

// ---- history -------------------------------------------------------------

pub fn history(nexum: &Nexum, document_ref: &str, json: bool) -> Result<()> {
    let id = resolve::node_id(nexum, document_ref)?;
    let versions = nexum.history(id)?;

    if versions.is_empty() {
        anyhow::bail!("{id} is not a document");
    }

    if json {
        return emit_json(&versions);
    }

    println!(
        "{} of {}",
        style::bold(&format!("{} versions", versions.len())),
        style::cyan(&versions[0].source_uri)
    );
    println!();

    for (position, document) in versions.iter().enumerate() {
        let is_current = position + 1 == versions.len();
        let marker = if is_current {
            style::green("●")
        } else {
            style::dim("○")
        };
        println!(
            "{marker} {} {}  {}",
            style::bold(&format!("v{}", document.version)),
            style::dim(&style::short_id(document.id)),
            document.created_at.to_rfc3339()
        );
        println!("    {:<12} {}", style::dim("title"), document.title);
        println!(
            "    {:<12} {}",
            style::dim("content"),
            &document.content_hash.0[..16]
        );

        let chunks = nexum.chunks_of(document.id)?;
        println!("    {:<12} {}", style::dim("chunks"), chunks.len());

        if is_current {
            println!("    {}", style::green("current version"));
        }
        println!();
    }
    Ok(())
}

// ---- export --------------------------------------------------------------

pub fn export(nexum: &Nexum, path: &Path, format: &str, json: bool) -> Result<()> {
    if !matches!(format.to_ascii_lowercase().as_str(), "jsonl" | "ndjson") {
        anyhow::bail!("unsupported format `{format}` — only jsonl is available");
    }

    let summary = if path == Path::new("-") {
        let mut stdout = std::io::stdout().lock();
        nexum.export(&mut stdout)?
    } else {
        let file = std::fs::File::create(path)
            .with_context(|| format!("could not write to {}", path.display()))?;
        let mut writer = std::io::BufWriter::new(file);
        let summary = nexum.export(&mut writer)?;
        writer.flush()?;
        summary
    };

    // Writing a progress report into the stream would corrupt it.
    if path == Path::new("-") {
        return Ok(());
    }

    if json {
        return emit_json(&serde_json::json!({
            "path": path.display().to_string(),
            "nodes": summary.nodes,
            "edges": summary.edges,
            "vectors": summary.vectors,
        }));
    }

    println!(
        "{} {} nodes, {} edges, {} vectors to {}",
        style::green("Exported"),
        summary.nodes,
        summary.edges,
        summary.vectors,
        style::bold(&path.display().to_string())
    );
    Ok(())
}

// ---- compact -------------------------------------------------------------

pub fn compact(nexum: &mut Nexum, json: bool) -> Result<()> {
    let report = nexum.compact()?;

    if json {
        return emit_json(&report);
    }

    println!("{}", style::green("Compacted"));
    println!(
        "  {:<20} {}",
        style::dim("log reclaimed"),
        style::bytes(report.wal_bytes_reclaimed)
    );
    println!(
        "  {:<20} {}",
        style::dim("store reclaimed"),
        style::bytes(report.store_bytes_reclaimed)
    );
    if report.indexes_rebuilt.is_empty() {
        println!("  {:<20} no rebuild needed", style::dim("indexes"));
    } else {
        println!(
            "  {:<20} {}",
            style::dim("indexes rebuilt"),
            report.indexes_rebuilt.join(", ")
        );
    }
    Ok(())
}

// ---- serve ---------------------------------------------------------------

pub async fn serve(nexum: Nexum, host: &str, port: u16, json: bool) -> Result<()> {
    let addr: std::net::SocketAddr = format!("{host}:{port}")
        .parse()
        .with_context(|| format!("`{host}:{port}` is not a valid address"))?;

    let path = nexum.path().display().to_string();
    let model = nexum.embedder().model_id().to_string();
    let nexum = Arc::new(nexum);

    let (bound, server) = nexum_server::bind(
        nexum,
        nexum_server::ServerConfig {
            addr,
            permissive_cors: true,
        },
    )
    .await
    .with_context(|| format!("could not bind {addr}"))?;

    if json {
        emit_json(&serde_json::json!({
            "listening": bound.to_string(),
            "database": path,
            "embedding_model": model,
        }))?;
    } else {
        println!("{} {}", style::green("Serving"), style::bold(&path));
        println!("  {:<12} http://{bound}", style::dim("api"));
        println!("  {:<12} http://{bound}/health", style::dim("health"));
        println!("  {:<12} {model}", style::dim("model"));
        println!();
        println!("{}", style::dim("Press Ctrl-C to stop."));
    }

    tokio::select! {
        result = server => result.context("server error")?,
        _ = tokio::signal::ctrl_c() => {
            if !json {
                println!("\n{}", style::dim("Stopped."));
            }
        }
    }
    Ok(())
}
