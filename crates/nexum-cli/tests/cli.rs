//! Drives the built `nexum` binary the way a user or a script would.
//!
//! These run the real executable rather than calling library functions, so
//! they cover argument parsing, exit codes, and the `--json` contract that
//! scripts depend on.

use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// Path to the binary under test, as cargo places it beside the test runner.
fn binary() -> PathBuf {
    let mut path = std::env::current_exe().expect("test binary path");
    path.pop();
    if path.ends_with("deps") {
        path.pop();
    }
    path.join(format!("nexum{}", std::env::consts::EXE_SUFFIX))
}

struct Cli {
    dir: tempfile::TempDir,
}

impl Cli {
    fn new() -> Self {
        Cli {
            dir: tempfile::tempdir().unwrap(),
        }
    }

    fn db(&self) -> PathBuf {
        self.dir.path().join("db")
    }

    fn docs(&self) -> PathBuf {
        self.dir.path().join("docs")
    }

    fn run(&self, args: &[&str]) -> Output {
        Command::new(binary())
            .args(args)
            // NO_COLOR keeps escape codes out of the assertions.
            .env("NO_COLOR", "1")
            .output()
            .expect("nexum should be runnable")
    }

    /// Run against this fixture's database, expecting success.
    fn ok(&self, args: &[&str]) -> String {
        let db = self.db();
        let mut full = vec!["--db", db.to_str().unwrap()];
        full.extend_from_slice(args);
        let output = self.run(&full);
        assert!(
            output.status.success(),
            "`nexum {}` failed:\n{}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).to_string()
    }

    /// Run expecting a non-zero exit, returning stderr.
    fn fails(&self, args: &[&str]) -> String {
        let db = self.db();
        let mut full = vec!["--db", db.to_str().unwrap()];
        full.extend_from_slice(args);
        let output = self.run(&full);
        assert!(
            !output.status.success(),
            "`nexum {}` should have failed but printed:\n{}",
            args.join(" "),
            String::from_utf8_lossy(&output.stdout)
        );
        String::from_utf8_lossy(&output.stderr).to_string()
    }

    fn json(&self, args: &[&str]) -> Value {
        let mut full = args.to_vec();
        full.push("--json");
        let out = self.ok(&full);
        serde_json::from_str(&out).unwrap_or_else(|e| {
            panic!(
                "`nexum {} --json` did not emit JSON: {e}\n{out}",
                args.join(" ")
            )
        })
    }

    /// A database with three documents, ready to query.
    fn seeded() -> Self {
        let cli = Cli::new();
        std::fs::create_dir_all(cli.docs()).unwrap();
        write(
            &cli.docs().join("lovelace.md"),
            "# Ada Lovelace\n\nAda Lovelace wrote the first algorithm intended for a machine. \
             She worked with Charles Babbage on the Analytical Engine.",
        );
        write(
            &cli.docs().join("turing.md"),
            "# Alan Turing\n\nAlan Turing formalised computation with the Turing machine. \
             His work at Bletchley Park helped break the Enigma cipher.",
        );
        write(
            &cli.docs().join("sourdough.txt"),
            "Combine flour, water and salt. Let the dough rise overnight before baking.",
        );

        cli.ok_global(&["init", cli.db().to_str().unwrap(), "--embedder", "hash:128"]);
        cli.ok(&["ingest", cli.docs().to_str().unwrap()]);
        cli
    }

    /// Run without injecting `--db`, for commands that take a path directly.
    fn ok_global(&self, args: &[&str]) -> String {
        let output = self.run(args);
        assert!(
            output.status.success(),
            "`nexum {}` failed:\n{}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).to_string()
    }
}

fn write(path: &Path, content: &str) {
    std::fs::write(path, content).unwrap();
}

#[test]
fn help_and_version_work() {
    let cli = Cli::new();
    let help = cli.run(&["--help"]);
    assert!(help.status.success());
    let text = String::from_utf8_lossy(&help.stdout);
    // Every command in the spec must be discoverable from help.
    for command in [
        "init", "ingest", "search", "traverse", "show", "stats", "history", "export", "serve",
    ] {
        assert!(text.contains(command), "`{command}` missing from help");
    }

    assert!(cli.run(&["--version"]).status.success());
}

#[test]
fn init_creates_a_database() {
    let cli = Cli::new();
    let output = cli.ok_global(&["init", cli.db().to_str().unwrap(), "--embedder", "hash:128"]);
    assert!(output.contains("Created"));
    assert!(cli.db().join("config.json").exists());

    // A second init must not silently clobber the first.
    let stderr =
        String::from_utf8_lossy(&cli.run(&["init", cli.db().to_str().unwrap()]).stderr).to_string();
    assert!(stderr.contains("already exists"), "got: {stderr}");
}

#[test]
fn init_emits_json_when_asked() {
    let cli = Cli::new();
    let out = cli.ok_global(&[
        "init",
        cli.db().to_str().unwrap(),
        "--embedder",
        "hash:64",
        "--json",
    ]);
    let value: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(value["dimensions"], 64);
    assert_eq!(value["embedding_model"], "hash-bow-v1-64");
}

#[test]
fn commands_against_a_missing_database_explain_how_to_fix_it() {
    let cli = Cli::new();
    let stderr = cli.fails(&["stats"]);
    assert!(stderr.contains("nexum init"), "got: {stderr}");
}

#[test]
fn an_invalid_embedder_is_rejected_at_init() {
    let cli = Cli::new();
    let output = cli.run(&["init", cli.db().to_str().unwrap(), "--embedder", "nonsense"]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("hash, local, openai, or ollama"),
        "got: {stderr}"
    );
}

#[test]
fn ingest_reports_what_it_wrote() {
    let cli = Cli::seeded();
    let stats = cli.json(&["stats"]);
    assert_eq!(stats["documents"], 3);
    assert!(stats["chunks"].as_u64().unwrap() >= 3);
}

#[test]
fn reingesting_unchanged_files_is_a_no_op() {
    let cli = Cli::seeded();
    let output = cli.ok(&["ingest", cli.docs().to_str().unwrap()]);
    assert!(output.contains("3 unchanged"), "got: {output}");
    assert_eq!(cli.json(&["stats"])["documents"], 3);
}

#[test]
fn changed_content_supersedes_and_history_shows_the_chain() {
    let cli = Cli::seeded();
    write(
        &cli.docs().join("turing.md"),
        "# Alan Turing\n\nRevised entirely: Turing also studied morphogenesis in biology.",
    );
    let output = cli.ok(&["ingest", cli.docs().to_str().unwrap()]);
    assert!(output.contains("superseded"), "got: {output}");

    let docs = cli.json(&["docs"]);
    let turing = docs
        .as_array()
        .unwrap()
        .iter()
        .find(|d| d["title"].as_str().unwrap().contains("Turing"))
        .unwrap();
    assert_eq!(turing["version"], 2);

    let history = cli.json(&["history", turing["id"].as_str().unwrap()]);
    let versions: Vec<u64> = history
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d["version"].as_u64().unwrap())
        .collect();
    assert_eq!(versions, vec![1, 2], "history reads oldest first");

    // The default listing shows only current versions.
    assert_eq!(docs.as_array().unwrap().len(), 3);
    assert_eq!(cli.json(&["docs", "--all"]).as_array().unwrap().len(), 4);
}

#[test]
fn search_ranks_by_relevance() {
    let cli = Cli::seeded();
    let results = cli.json(&["search", "analytical engine algorithm", "--top-k", "3"]);
    let hits = results["results"].as_array().unwrap();
    assert!(!hits.is_empty());

    let top = hits[0]["node"]["text"].as_str().unwrap().to_lowercase();
    assert!(
        top.contains("algorithm") || top.contains("engine"),
        "expected the Lovelace chunk on top, got: {top}"
    );

    let scores: Vec<f64> = hits.iter().map(|h| h["score"].as_f64().unwrap()).collect();
    assert!(
        scores.windows(2).all(|w| w[0] >= w[1]),
        "scores must descend"
    );
}

#[test]
fn search_honours_top_k() {
    let cli = Cli::seeded();
    let results = cli.json(&["search", "the", "--top-k", "2"]);
    assert!(results["results"].as_array().unwrap().len() <= 2);
}

#[test]
fn search_can_expand_across_the_graph() {
    let cli = Cli::seeded();
    let results = cli.json(&[
        "search",
        "analytical engine",
        "--top-k",
        "2",
        "--expand",
        "MENTIONS",
        "--hops",
        "2",
    ]);
    let hits = results["results"].as_array().unwrap();
    assert!(
        hits.iter().any(|h| !h["hops"].is_null()),
        "expansion should add graph-reached nodes"
    );
    assert!(
        hits.iter().any(|h| h["node"]["kind"] == "Entity"),
        "MENTIONS should reach entities"
    );
}

#[test]
fn an_unknown_edge_type_lists_the_valid_ones() {
    let cli = Cli::seeded();
    let stderr = cli.fails(&["search", "x", "--expand", "NOT_REAL"]);
    assert!(
        stderr.contains("MENTIONS"),
        "should list valid types: {stderr}"
    );
}

#[test]
fn show_prints_a_node_and_its_edges() {
    let cli = Cli::seeded();
    let docs = cli.json(&["docs"]);
    let id = docs[0]["id"].as_str().unwrap();

    let output = cli.ok(&["show", id]);
    assert!(output.contains("Document"));
    assert!(
        output.contains("PART_OF"),
        "edges should be listed: {output}"
    );

    let detail = cli.json(&["show", id]);
    assert_eq!(detail["node"]["kind"], "Document");
    assert!(!detail["incoming"].as_array().unwrap().is_empty());
}

#[test]
fn nodes_can_be_referenced_by_their_printed_short_id() {
    let cli = Cli::seeded();
    let docs = cli.json(&["docs"]);
    let full = docs[0]["id"].as_str().unwrap();
    // The short form the CLI prints is the tail, because UUIDv7 prefixes are
    // timestamps and collide across a batch.
    let short: String = full.chars().filter(|c| *c != '-').collect::<String>()[24..].to_string();

    let detail = cli.json(&["show", &short]);
    assert_eq!(detail["node"]["id"], full);
}

#[test]
fn an_unknown_node_reference_is_an_error() {
    let cli = Cli::seeded();
    assert!(cli.fails(&["show", "ffffffffff"]).contains("no node"));
    assert!(cli.fails(&["show", "ab"]).contains("too short"));
    assert!(
        cli.fails(&["show", "not-hex-at-all"])
            .contains("not a node id")
    );
}

#[test]
fn traverse_walks_the_graph() {
    let cli = Cli::seeded();
    let docs = cli.json(&["docs"]);
    let id = docs[0]["id"].as_str().unwrap();

    let result = cli.json(&[
        "traverse",
        id,
        "--edges",
        "PART_OF",
        "--hops",
        "1",
        "--direction",
        "in",
    ]);
    let nodes = result["nodes"].as_array().unwrap();
    assert!(!nodes.is_empty());
    assert!(nodes.iter().all(|n| n["node"]["kind"] == "Chunk"));
    assert!(nodes.iter().all(|n| n["hops"] == 1));
}

#[test]
fn stats_reports_counts_and_edge_breakdown() {
    let cli = Cli::seeded();
    let stats = cli.json(&["stats"]);
    assert_eq!(stats["documents"], 3);
    assert_eq!(stats["latest_documents"], 3);
    assert!(stats["edges_by_type"]["PART_OF"].as_u64().unwrap() > 0);
    assert!(
        stats["namespaces"]["hash-bow-v1-128:128"]["count"]
            .as_u64()
            .unwrap()
            > 0
    );

    // The human rendering should mention the model too.
    let text = cli.ok(&["stats"]);
    assert!(text.contains("hash-bow-v1-128"));
}

#[test]
fn export_writes_a_readable_jsonl_dump() {
    let cli = Cli::seeded();
    let out = cli.dir.path().join("dump.jsonl");
    cli.ok(&["export", out.to_str().unwrap()]);

    let content = std::fs::read_to_string(&out).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert!(lines.len() > 5);

    // Every line must be valid JSON with a type tag.
    let mut kinds = std::collections::HashSet::new();
    for line in &lines {
        let value: Value =
            serde_json::from_str(line).unwrap_or_else(|e| panic!("bad JSONL line: {e}\n{line}"));
        kinds.insert(value["type"].as_str().unwrap_or("header").to_string());
    }
    assert!(kinds.contains("node"));
    assert!(kinds.contains("edge"));
    assert!(kinds.contains("vector"));

    // The header records what produced the dump.
    let header: Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(header["format_version"], 1);
    assert!(header["engine_version"].is_string());
}

#[test]
fn export_to_stdout_emits_only_the_dump() {
    let cli = Cli::seeded();
    let out = cli.ok(&["export", "-"]);
    // A status line mixed into the stream would make it unparseable.
    for line in out.lines() {
        serde_json::from_str::<Value>(line)
            .unwrap_or_else(|e| panic!("stdout export was polluted: {e}\n{line}"));
    }
}

#[test]
fn an_unsupported_export_format_is_rejected() {
    let cli = Cli::seeded();
    assert!(
        cli.fails(&["export", "out.parquet", "--format", "parquet"])
            .contains("jsonl")
    );
}

#[test]
fn compact_preserves_the_data() {
    let cli = Cli::seeded();
    let before = cli.json(&["stats"]);
    let report = cli.json(&["compact"]);
    assert!(report["wal_bytes_reclaimed"].is_number());

    let after = cli.json(&["stats"]);
    assert_eq!(before["documents"], after["documents"]);
    assert_eq!(before["chunks"], after["chunks"]);
    assert_eq!(before["edges"], after["edges"]);

    // Still searchable afterwards.
    assert!(
        !cli.json(&["search", "engine"])["results"]
            .as_array()
            .unwrap()
            .is_empty()
    );
}

#[test]
fn every_read_command_supports_json() {
    let cli = Cli::seeded();
    let docs = cli.json(&["docs"]);
    let id = docs[0]["id"].as_str().unwrap().to_string();

    // The spec requires --json on every read command; check each one parses.
    cli.json(&["stats"]);
    cli.json(&["docs"]);
    cli.json(&["search", "engine"]);
    cli.json(&["show", &id]);
    cli.json(&["traverse", &id, "--hops", "1"]);
    cli.json(&["history", &id]);
}

#[test]
fn json_output_carries_no_ansi_escapes() {
    let cli = Cli::seeded();
    // Even with colour forced on, --json must stay machine-readable.
    let output = Command::new(binary())
        .args(["--db", cli.db().to_str().unwrap(), "stats", "--json"])
        .env("CLICOLOR_FORCE", "1")
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(!text.contains('\x1b'), "escape codes leaked into JSON");
}

#[test]
fn the_database_path_can_come_from_the_environment() {
    let cli = Cli::seeded();
    let output = Command::new(binary())
        .args(["stats", "--json"])
        .env("NEXUM_DB", cli.db())
        .env("NO_COLOR", "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["documents"], 3);
}

#[test]
fn entity_extraction_can_be_disabled_per_run() {
    let cli = Cli::new();
    std::fs::create_dir_all(cli.docs()).unwrap();
    write(
        &cli.docs().join("a.md"),
        "Ada Lovelace worked with Charles Babbage on the Analytical Engine.",
    );
    cli.ok_global(&["init", cli.db().to_str().unwrap(), "--embedder", "hash:64"]);
    cli.ok(&["ingest", cli.docs().to_str().unwrap(), "--no-entities"]);
    assert_eq!(cli.json(&["stats"])["entities"], 0);
}

#[test]
fn the_chunker_is_selectable_and_recorded() {
    let cli = Cli::new();
    std::fs::create_dir_all(cli.docs()).unwrap();
    write(
        &cli.docs().join("a.txt"),
        &"Sentence number one. ".repeat(40),
    );
    cli.ok_global(&["init", cli.db().to_str().unwrap(), "--embedder", "hash:64"]);
    cli.ok(&[
        "ingest",
        cli.docs().to_str().unwrap(),
        "--chunker",
        "fixed:100:10",
    ]);

    let stats = cli.json(&["stats"]);
    assert!(
        stats["chunks"].as_u64().unwrap() > 5,
        "small chunks should split a lot"
    );

    let bad = cli.fails(&[
        "ingest",
        cli.docs().to_str().unwrap(),
        "--chunker",
        "fixed:10:10",
    ]);
    assert!(bad.contains("overlap"), "got: {bad}");
}
