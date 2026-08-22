//! The `nexum` command-line interface.
//!
//! Every read command takes `--json`, so anything the CLI can show is also
//! something a script can consume. The human output is the same data rendered
//! for a terminal — there is no capability that exists in one mode and not the
//! other.

mod commands;
mod resolve;
mod style;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use nexum_client::{ClientConfig, Nexum};
use nexum_embed::EmbedderConfig;
use nexum_ingest::{ChunkerConfig, IngestConfig};
use std::path::PathBuf;

/// Default database location when `--db` is not given.
const DEFAULT_DB_DIR: &str = "nexum.db";

#[derive(Parser, Debug)]
#[command(
    name = "nexum",
    version,
    about = "A graph-native vector database for RAG",
    long_about = "NexumDB stores vectors as a property on graph nodes, so similarity search \
                  and graph traversal compose in one query, with document versioning and \
                  provenance as first-class citizens.",
    propagate_version = true
)]
struct Cli {
    /// Path to the database directory.
    ///
    /// Falls back to `$NEXUM_DB`, then `./nexum.db`.
    #[arg(long, short = 'd', global = true, env = "NEXUM_DB")]
    db: Option<PathBuf>,

    /// Emit JSON instead of human-readable output.
    #[arg(long, global = true)]
    json: bool,

    /// Increase log verbosity. Repeat for more.
    #[arg(long, short = 'v', global = true, action = clap::ArgAction::Count)]
    verbose: u8,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Create a new database.
    Init {
        /// Where to create it. Defaults to the global `--db` path.
        path: Option<PathBuf>,

        /// Embedding provider: `hash`, `local[:model]`, `openai[:model]`, `ollama:<model>`.
        #[arg(long, default_value = "local")]
        embedder: String,
    },

    /// Ingest a file or directory.
    Ingest {
        /// File or directory to ingest.
        path: PathBuf,

        /// Chunking strategy: `recursive[:size[:overlap]]`, `fixed:...`,
        /// `sentence:...`, or `whole`.
        #[arg(long)]
        chunker: Option<String>,

        /// Embedding provider override for this run.
        #[arg(long)]
        embedder: Option<String>,

        /// Do not descend into subdirectories.
        #[arg(long)]
        no_recursive: bool,

        /// Skip entity extraction.
        #[arg(long)]
        no_entities: bool,

        /// Re-ingest even when the content is unchanged.
        #[arg(long)]
        force: bool,
    },

    /// Search for chunks matching a query.
    Search {
        /// The query text.
        query: String,

        /// How many results to return.
        #[arg(long, short = 'k', default_value_t = 10)]
        top_k: usize,

        /// Also expand results across these edge types.
        #[arg(long, value_delimiter = ',')]
        expand: Option<Vec<String>>,

        /// Hops to expand when `--expand` is given.
        #[arg(long, default_value_t = 1)]
        hops: usize,

        /// Include chunks from superseded document versions.
        #[arg(long)]
        include_superseded: bool,

        /// Search a specific embedding model's index.
        #[arg(long)]
        model: Option<String>,
    },

    /// Walk the graph outward from a node.
    Traverse {
        /// Node to start from. A unique ID prefix is enough.
        node_id: String,

        /// Edge types to follow. Defaults to every type.
        #[arg(long, value_delimiter = ',')]
        edges: Option<Vec<String>>,

        /// Maximum hops.
        #[arg(long, default_value_t = 2)]
        hops: usize,

        /// Direction: `out`, `in`, or `both`.
        #[arg(long, default_value = "both")]
        direction: String,
    },

    /// Show a node's properties and edges.
    Show {
        /// Node to show. A unique ID prefix is enough.
        node_id: String,

        /// Print the chunk's full text rather than a preview.
        #[arg(long)]
        full: bool,
    },

    /// List documents.
    Docs {
        /// Include superseded versions.
        #[arg(long)]
        all: bool,

        /// Maximum rows.
        #[arg(long, default_value_t = 50)]
        limit: usize,
    },

    /// Show node and edge counts, index sizes, and embedding models.
    Stats,

    /// Show a document's version chain.
    History {
        /// Document to trace. A unique ID prefix is enough.
        document_id: String,
    },

    /// Dump the database for backup or inspection.
    Export {
        /// Where to write. Use `-` for stdout.
        path: PathBuf,

        /// Output format.
        #[arg(long, default_value = "jsonl")]
        format: String,
    },

    /// Rebuild indexes, truncate the log, and reclaim space.
    Compact,

    /// Serve the HTTP API for the viewer app and remote clients.
    Serve {
        /// Port to listen on.
        #[arg(long, short = 'p', default_value_t = 8080)]
        port: u16,

        /// Address to bind. Loopback by default — the API has no auth.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    init_tracing(cli.verbose);

    if let Err(e) = run(cli).await {
        // The chain matters: "no such file" alone does not say which file.
        eprintln!("{}: {e}", style::red("error"));
        for cause in e.chain().skip(1) {
            eprintln!("  {} {cause}", style::dim("caused by:"));
        }
        std::process::exit(1);
    }
}

fn init_tracing(verbose: u8) {
    let level = match verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    let filter = std::env::var("RUST_LOG").unwrap_or_else(|_| format!("nexum={level}"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .without_time()
        .try_init();
}

async fn run(cli: Cli) -> Result<()> {
    let db_path = cli
        .db
        .clone()
        .unwrap_or_else(|| PathBuf::from(DEFAULT_DB_DIR));

    match cli.command {
        Command::Init { path, embedder } => {
            let target = path.unwrap_or(db_path);
            commands::init(&target, &embedder, cli.json).await
        }

        Command::Ingest {
            path,
            chunker,
            embedder,
            no_recursive,
            no_entities,
            force,
        } => {
            let mut config = base_config(embedder.as_deref())?;
            if let Some(spec) = chunker {
                config.ingest.chunker = ChunkerConfig::parse(&spec).context("invalid --chunker")?;
            }
            config.ingest.extract_entities = !no_entities;
            config.ingest.skip_unchanged = !force;

            let nexum = open(&db_path, config).await?;
            commands::ingest(&nexum, &path, !no_recursive, cli.json).await
        }

        Command::Search {
            query,
            top_k,
            expand,
            hops,
            include_superseded,
            model,
        } => {
            let nexum = open(&db_path, base_config(None)?).await?;
            commands::search(
                &nexum,
                &query,
                top_k,
                expand.as_deref(),
                hops,
                include_superseded,
                model,
                cli.json,
            )
            .await
        }

        Command::Traverse {
            node_id,
            edges,
            hops,
            direction,
        } => {
            let nexum = open(&db_path, base_config(None)?).await?;
            commands::traverse(
                &nexum,
                &node_id,
                edges.as_deref(),
                hops,
                &direction,
                cli.json,
            )
        }

        Command::Show { node_id, full } => {
            let nexum = open(&db_path, base_config(None)?).await?;
            commands::show(&nexum, &node_id, full, cli.json)
        }

        Command::Docs { all, limit } => {
            let nexum = open(&db_path, base_config(None)?).await?;
            commands::docs(&nexum, all, limit, cli.json)
        }

        Command::Stats => {
            let nexum = open(&db_path, base_config(None)?).await?;
            commands::stats(&nexum, cli.json)
        }

        Command::History { document_id } => {
            let nexum = open(&db_path, base_config(None)?).await?;
            commands::history(&nexum, &document_id, cli.json)
        }

        Command::Export { path, format } => {
            let nexum = open(&db_path, base_config(None)?).await?;
            commands::export(&nexum, &path, &format, cli.json)
        }

        Command::Compact => {
            let mut nexum = open(&db_path, base_config(None)?).await?;
            commands::compact(&mut nexum, cli.json)
        }

        Command::Serve { port, host } => {
            let nexum = open(&db_path, base_config(None)?).await?;
            commands::serve(nexum, &host, port, cli.json).await
        }
    }
}

/// Build a client config, honouring an embedder override.
///
/// When no override is given the embedder comes from the database's own
/// config, so a database ingested with one model is not silently searched with
/// another.
fn base_config(embedder: Option<&str>) -> Result<ClientConfig> {
    let mut config = ClientConfig {
        ingest: IngestConfig::default(),
        ..Default::default()
    };
    if let Some(spec) = embedder {
        config.embedder =
            EmbedderConfig::parse(spec).with_context(|| format!("invalid --embedder `{spec}`"))?;
    }
    Ok(config)
}

/// Open an existing database, with a message that says what to do if it is not
/// there.
async fn open(path: &std::path::Path, mut config: ClientConfig) -> Result<Nexum> {
    if !path.join("config.json").exists() {
        anyhow::bail!(
            "no database at {} — create one with `nexum init {}`",
            path.display(),
            path.display()
        );
    }

    // The database records which embedding model it was built with. Reuse it
    // unless the caller explicitly asked for something else, so query vectors
    // land in the same space as the stored ones.
    if let Ok(saved) = std::fs::read(path.join("embedder.json"))
        && let Ok(saved) = serde_json::from_slice::<EmbedderConfig>(&saved)
        && config.embedder == EmbedderConfig::default()
    {
        config.embedder = saved;
    }

    Nexum::open(path, config)
        .await
        .with_context(|| format!("could not open database at {}", path.display()))
}
