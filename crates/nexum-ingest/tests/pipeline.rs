//! End-to-end ingestion: chunking, embedding, entities, versioning, provenance.

use nexum_core::{
    Db, DbConfig, Direction, EdgeType, Node, NodeKind, Predicate, PropertyValue, Query,
};
use nexum_embed::{Embedder, hash::HashEmbedder};
use nexum_ingest::{ChunkerConfig, IngestConfig, IngestOutcome, Ingestor};
use std::sync::Arc;

struct Fixture {
    _dir: tempfile::TempDir,
    db: Arc<Db>,
    ingestor: Ingestor,
}

impl Fixture {
    fn new() -> Self {
        Fixture::with_config(IngestConfig::default())
    }

    fn with_config(config: IngestConfig) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Db::create(dir.path().join("db"), DbConfig::default()).unwrap());
        let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::new(128));
        let ingestor = Ingestor::new(db.clone(), embedder, config);
        Fixture {
            _dir: dir,
            db,
            ingestor,
        }
    }
}

const ARTICLE: &str = "Ada Lovelace wrote the first algorithm intended for a machine. \
She collaborated with Charles Babbage on the Analytical Engine. \
The work was published in 1843 and is widely regarded as the first computer program. \
Babbage designed the machine but never completed a working version.";

#[tokio::test]
async fn ingesting_text_creates_a_document_with_chunks() {
    let f = Fixture::new();
    let report = f
        .ingestor
        .ingest_text("mem:///article", "Ada Lovelace", ARTICLE.into())
        .await
        .unwrap();

    assert!(matches!(
        report.outcome,
        IngestOutcome::Created { version: 1 }
    ));
    assert!(report.chunks >= 1);
    assert!(report.document_id.is_some());
    assert!(report.run_id.is_some());

    let stats = f.db.stats().unwrap();
    assert_eq!(stats.documents, 1);
    assert_eq!(stats.chunks as usize, report.chunks);
    assert_eq!(stats.pipeline_runs, 1);
    assert_eq!(
        stats.namespaces["hash-bow-v1-128:128"].count as usize,
        report.chunks
    );
}

#[tokio::test]
async fn every_chunk_is_wired_to_its_document_and_run() {
    let f = Fixture::with_config(IngestConfig {
        chunker: ChunkerConfig::Sentence {
            max_size: 80,
            overlap_sentences: 0,
        },
        ..Default::default()
    });
    let report = f
        .ingestor
        .ingest_text("mem:///a", "A", ARTICLE.into())
        .await
        .unwrap();
    assert!(report.chunks > 1, "expected several chunks");

    let doc_id = report.document_id.unwrap();
    let run_id = report.run_id.unwrap();
    let chunks =
        f.db.query(&Query::new().seed_kind(NodeKind::Chunk))
            .unwrap();

    for chunk in &chunks.nodes {
        let detail = f.db.node_detail(chunk.id()).unwrap().unwrap();
        let outgoing: Vec<(EdgeType, _)> = detail
            .outgoing
            .iter()
            .map(|e| (e.edge.edge_type, e.other_id))
            .collect();
        assert!(outgoing.contains(&(EdgeType::PartOf, doc_id)));
        assert!(outgoing.contains(&(EdgeType::DerivedFrom, doc_id)));
        assert!(
            outgoing.contains(&(EdgeType::EmbeddedBy, run_id)),
            "every chunk must trace back to the run that embedded it"
        );
    }
}

#[tokio::test]
async fn chunks_are_linked_in_reading_order() {
    let f = Fixture::with_config(IngestConfig {
        chunker: ChunkerConfig::Sentence {
            max_size: 80,
            overlap_sentences: 0,
        },
        ..Default::default()
    });
    f.ingestor
        .ingest_text("mem:///a", "A", ARTICLE.into())
        .await
        .unwrap();

    let chunks =
        f.db.query(&Query::new().seed_kind(NodeKind::Chunk))
            .unwrap();
    let mut ordered: Vec<&Node> = chunks.nodes.iter().map(|n| &n.node).collect();
    ordered.sort_by_key(|n| n.as_chunk().unwrap().chunk_index);

    // Walking PRECEDES from the first chunk must visit them all in order.
    let walk =
        f.db.query(&Query::new().seed([ordered[0].id()]).traverse_with(
            vec![EdgeType::Precedes],
            ordered.len(),
            Direction::Out,
            false,
        ))
        .unwrap();
    assert_eq!(walk.len(), ordered.len() - 1);

    // And FOLLOWS must work backwards from the last.
    let back =
        f.db.neighbors(
            ordered.last().unwrap().id(),
            &[EdgeType::Follows],
            Direction::Out,
        )
        .unwrap();
    assert_eq!(back.len(), 1);
}

#[tokio::test]
async fn search_finds_ingested_content() {
    let f = Fixture::new();
    f.ingestor
        .ingest_text("mem:///ada", "Ada", ARTICLE.into())
        .await
        .unwrap();
    f.ingestor
        .ingest_text(
            "mem:///bread",
            "Bread",
            "Combine flour, water, salt and yeast. Knead the dough and let it rise overnight."
                .into(),
        )
        .await
        .unwrap();

    let embedder = HashEmbedder::new(128);
    let query = embedder
        .embed_query("analytical engine algorithm")
        .await
        .unwrap();
    let results = f.db.query(&Query::new().search(query, 3).latest()).unwrap();

    assert!(!results.is_empty());
    let top = results.nodes[0].node.as_chunk().unwrap();
    assert!(
        top.text.to_lowercase().contains("engine") || top.text.to_lowercase().contains("algorithm"),
        "top hit should be the Ada article, got: {}",
        top.text
    );
}

#[tokio::test]
async fn reingesting_identical_content_is_a_no_op() {
    let f = Fixture::new();
    let first = f
        .ingestor
        .ingest_text("mem:///a", "A", ARTICLE.into())
        .await
        .unwrap();
    let second = f
        .ingestor
        .ingest_text("mem:///a", "A", ARTICLE.into())
        .await
        .unwrap();

    assert!(matches!(second.outcome, IngestOutcome::Unchanged { .. }));
    assert_eq!(second.chunks, 0);
    // Nothing new should have been written.
    let stats = f.db.stats().unwrap();
    assert_eq!(stats.documents, 1);
    assert_eq!(stats.chunks as usize, first.chunks);
    assert_eq!(
        stats.pipeline_runs, 1,
        "a skipped ingest must not mint a run"
    );
}

#[tokio::test]
async fn changed_content_creates_a_new_version() {
    let f = Fixture::new();
    let v1 = f
        .ingestor
        .ingest_text("mem:///a", "A", "The original text.".into())
        .await
        .unwrap();
    let v2 = f
        .ingestor
        .ingest_text(
            "mem:///a",
            "A revised",
            "The revised text, quite different.".into(),
        )
        .await
        .unwrap();

    match v2.outcome {
        IngestOutcome::Superseded {
            version,
            supersedes,
        } => {
            assert_eq!(version, 2);
            assert_eq!(supersedes, v1.document_id.unwrap());
        }
        other => panic!("expected supersession, got {other:?}"),
    }

    // Both versions exist; only the newest is latest.
    assert_eq!(f.db.stats().unwrap().documents, 2);
    let latest =
        f.db.query(&Query::new().seed_kind(NodeKind::Document).latest())
            .unwrap();
    assert_eq!(latest.ids(), vec![v2.document_id.unwrap()]);

    // History reads oldest first.
    let history = f.db.history(v2.document_id.unwrap()).unwrap();
    let versions: Vec<u32> = history
        .iter()
        .map(|n| n.as_document().unwrap().version)
        .collect();
    assert_eq!(versions, vec![1, 2]);
}

#[tokio::test]
async fn superseded_chunks_are_excluded_from_default_retrieval() {
    let f = Fixture::new();
    f.ingestor
        .ingest_text("mem:///a", "A", "Alpha beta gamma delta.".into())
        .await
        .unwrap();
    f.ingestor
        .ingest_text("mem:///a", "A", "Epsilon zeta eta theta.".into())
        .await
        .unwrap();

    let all =
        f.db.query(&Query::new().seed_kind(NodeKind::Chunk))
            .unwrap();
    let live =
        f.db.query(&Query::new().seed_kind(NodeKind::Chunk).latest())
            .unwrap();
    assert!(all.len() > live.len(), "old chunks should still exist");
    assert!(!live.is_empty());

    // But they remain reachable for audit.
    let superseded =
        f.db.query(
            &Query::new()
                .seed_kind(NodeKind::Chunk)
                .filter(Predicate::Superseded),
        )
        .unwrap();
    assert_eq!(superseded.len(), all.len() - live.len());
}

#[tokio::test]
async fn entities_are_extracted_and_linked_to_chunks() {
    let f = Fixture::new();
    let report = f
        .ingestor
        .ingest_text("mem:///ada", "Ada", ARTICLE.into())
        .await
        .unwrap();
    assert!(report.entities > 0, "expected some entities");

    let entities =
        f.db.query(&Query::new().seed_kind(NodeKind::Entity))
            .unwrap();
    let names: Vec<String> = entities
        .nodes
        .iter()
        .map(|n| n.node.as_entity().unwrap().name.clone())
        .collect();
    assert!(
        names.iter().any(|n| n.contains("Lovelace")),
        "expected Ada Lovelace in {names:?}"
    );

    // Each entity must trace back to the run that extracted it.
    for entity in &entities.nodes {
        let runs =
            f.db.neighbors(entity.id(), &[EdgeType::ExtractedBy], Direction::Out)
                .unwrap();
        assert_eq!(runs.len(), 1, "entity should point at exactly one run");
        assert_eq!(runs[0].kind(), NodeKind::PipelineRun);
    }

    // And chunks must mention them.
    let mentions =
        f.db.query(&Query::new().seed_kind(NodeKind::Chunk).traverse_with(
            vec![EdgeType::Mentions],
            1,
            Direction::Out,
            false,
        ))
        .unwrap();
    assert!(!mentions.is_empty());
}

#[tokio::test]
async fn the_same_entity_across_documents_is_one_node() {
    let f = Fixture::new();
    f.ingestor
        .ingest_text(
            "mem:///a",
            "A",
            "Ada Lovelace wrote about the engine.".into(),
        )
        .await
        .unwrap();
    f.ingestor
        .ingest_text(
            "mem:///b",
            "B",
            "Much later, Ada Lovelace was recognised.".into(),
        )
        .await
        .unwrap();

    let entities =
        f.db.query(&Query::new().seed_kind(NodeKind::Entity))
            .unwrap();
    let lovelace: Vec<_> = entities
        .nodes
        .iter()
        .filter(|n| n.node.as_entity().unwrap().name.contains("Lovelace"))
        .collect();
    assert_eq!(
        lovelace.len(),
        1,
        "content-addressed IDs should dedupe: {lovelace:?}"
    );

    // Mentions accumulate across runs rather than resetting.
    let mentions = lovelace[0]
        .node
        .metadata()
        .get("mentions")
        .and_then(PropertyValue::as_i64)
        .unwrap();
    assert!(
        mentions >= 2,
        "expected accumulated mentions, got {mentions}"
    );

    // The shared entity connects the two documents.
    let reached =
        f.db.query(&Query::new().seed([lovelace[0].id()]).traverse_with(
            vec![EdgeType::Mentions],
            1,
            Direction::In,
            false,
        ))
        .unwrap();
    assert!(reached.len() >= 2, "entity should bridge both documents");
}

#[tokio::test]
async fn entity_extraction_can_be_switched_off() {
    let f = Fixture::with_config(IngestConfig {
        extract_entities: false,
        ..Default::default()
    });
    let report = f
        .ingestor
        .ingest_text("mem:///a", "A", ARTICLE.into())
        .await
        .unwrap();
    assert_eq!(report.entities, 0);
    assert_eq!(f.db.stats().unwrap().entities, 0);
}

#[tokio::test]
async fn provenance_records_the_full_configuration() {
    let f = Fixture::with_config(IngestConfig {
        chunker: ChunkerConfig::Fixed {
            size: 120,
            overlap: 20,
        },
        ..Default::default()
    });
    let report = f
        .ingestor
        .ingest_text("mem:///a", "A", ARTICLE.into())
        .await
        .unwrap();

    let run = f.db.get_node(report.run_id.unwrap()).unwrap().unwrap();
    let run = run.as_pipeline_run().unwrap();
    assert_eq!(run.chunker, "fixed:120:20");
    assert_eq!(run.embedding_model, "hash-bow-v1-128");
    assert!(!run.config_hash.as_str().is_empty());
    assert_eq!(
        run.metadata.get("vector_namespace").unwrap().as_str(),
        Some("hash-bow-v1-128:128")
    );
}

#[tokio::test]
async fn identical_configs_hash_identically_and_different_ones_do_not() {
    let a = Fixture::with_config(IngestConfig {
        chunker: ChunkerConfig::Fixed {
            size: 100,
            overlap: 10,
        },
        ..Default::default()
    });
    let b = Fixture::with_config(IngestConfig {
        chunker: ChunkerConfig::Fixed {
            size: 100,
            overlap: 10,
        },
        ..Default::default()
    });
    let c = Fixture::with_config(IngestConfig {
        chunker: ChunkerConfig::Fixed {
            size: 500,
            overlap: 10,
        },
        ..Default::default()
    });

    let hash_of = |f: &Fixture, report: nexum_ingest::IngestReport| {
        f.db.get_node(report.run_id.unwrap())
            .unwrap()
            .unwrap()
            .as_pipeline_run()
            .unwrap()
            .config_hash
            .clone()
    };

    let ha = hash_of(
        &a,
        a.ingestor
            .ingest_text("m:///x", "x", ARTICLE.into())
            .await
            .unwrap(),
    );
    let hb = hash_of(
        &b,
        b.ingestor
            .ingest_text("m:///x", "x", ARTICLE.into())
            .await
            .unwrap(),
    );
    let hc = hash_of(
        &c,
        c.ingestor
            .ingest_text("m:///x", "x", ARTICLE.into())
            .await
            .unwrap(),
    );

    assert_eq!(ha, hb, "same config must hash the same");
    assert_ne!(ha, hc, "a different chunker must change the hash");
}

#[tokio::test]
async fn a_document_with_no_text_is_skipped() {
    let f = Fixture::new();
    let report = f
        .ingestor
        .ingest_text("mem:///empty", "Empty", "   \n\n  ".into())
        .await
        .unwrap();
    assert!(matches!(report.outcome, IngestOutcome::Skipped { .. }));
    assert_eq!(f.db.stats().unwrap().documents, 0);
}

#[tokio::test]
async fn ingesting_a_directory_walks_it() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.md"), "# First\n\nSome content here.").unwrap();
    std::fs::write(dir.path().join("b.txt"), "Other content entirely.").unwrap();
    std::fs::create_dir(dir.path().join("sub")).unwrap();
    std::fs::write(dir.path().join("sub/c.md"), "# Third\n\nNested content.").unwrap();
    std::fs::write(dir.path().join("skip.png"), "binary-ish").unwrap();

    let f = Fixture::new();
    let reports = f.ingestor.ingest_path(dir.path(), true).await.unwrap();
    assert_eq!(reports.len(), 3, "png should be excluded: {reports:?}");
    assert!(reports.iter().all(|r| r.outcome.wrote()));

    // Markdown headings become titles.
    let titles: Vec<&str> = reports.iter().map(|r| r.title.as_str()).collect();
    assert!(titles.contains(&"First"));
    assert!(titles.contains(&"Third"));
    assert_eq!(f.db.stats().unwrap().documents, 3);
}

#[tokio::test]
async fn a_binary_file_is_reported_as_skipped_not_fatal() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.txt");
    std::fs::write(&path, [0x00, 0x01, 0x02]).unwrap();

    let f = Fixture::new();
    let report = f.ingestor.ingest_file(&path).await.unwrap();
    assert!(matches!(report.outcome, IngestOutcome::Skipped { .. }));
}

#[tokio::test]
async fn the_multi_hop_query_from_the_spec_works_end_to_end() {
    let f = Fixture::new();
    f.ingestor
        .ingest_text("mem:///a", "A", ARTICLE.into())
        .await
        .unwrap();
    f.ingestor
        .ingest_text(
            "mem:///b",
            "B",
            "Charles Babbage also designed the Difference Engine. \
             His work influenced later computing pioneers considerably."
                .into(),
        )
        .await
        .unwrap();

    // search -> traverse -> filter, exactly as section 3.2 describes.
    let embedder = HashEmbedder::new(128);
    let query = embedder
        .embed_query("first algorithm machine")
        .await
        .unwrap();
    let result =
        f.db.query(
            &Query::new()
                .search(query, 5)
                .traverse(vec![EdgeType::Mentions, EdgeType::RelatesTo], 2)
                .latest(),
        )
        .unwrap();

    assert!(!result.is_empty(), "traversal should surface related nodes");
    // The pipeline should reach beyond what vector search alone returned.
    assert!(
        result.nodes.iter().any(|n| n.hops.is_some()),
        "results should include graph-expanded nodes"
    );
    assert_eq!(result.stats.stages.len(), 3);
}

#[tokio::test]
async fn re_embedding_with_a_second_model_adds_an_index_without_disturbing_the_first() {
    let dir = tempfile::tempdir().unwrap();
    let db = Arc::new(Db::create(dir.path().join("db"), DbConfig::default()).unwrap());

    let small: Arc<dyn Embedder> = Arc::new(HashEmbedder::new(64));
    let large: Arc<dyn Embedder> = Arc::new(HashEmbedder::new(256));

    let first = Ingestor::new(db.clone(), small, IngestConfig::default());
    first
        .ingest_text("mem:///a", "A", ARTICLE.into())
        .await
        .unwrap();

    // A different model on the same source: a new version, a new namespace.
    let second = Ingestor::new(
        db.clone(),
        large,
        IngestConfig {
            skip_unchanged: false,
            ..Default::default()
        },
    );
    second
        .ingest_text("mem:///a", "A", ARTICLE.into())
        .await
        .unwrap();

    let mut namespaces = db.namespaces();
    namespaces.sort();
    assert_eq!(
        namespaces,
        vec![
            "hash-bow-v1-256:256".to_string(),
            "hash-bow-v1-64:64".to_string()
        ]
    );

    // Both indexes answer, each in its own space.
    let q64 = HashEmbedder::new(64)
        .embed_query("analytical engine")
        .await
        .unwrap();
    let q256 = HashEmbedder::new(256)
        .embed_query("analytical engine")
        .await
        .unwrap();
    assert!(
        !db.query(&Query::new().search_with(q64, 3, Some("hash-bow-v1-64".into())))
            .unwrap()
            .is_empty()
    );
    assert!(
        !db.query(&Query::new().search_with(q256, 3, Some("hash-bow-v1-256".into())))
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn everything_survives_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db");

    let (doc_count, chunk_count) = {
        let db = Arc::new(Db::create(&path, DbConfig::default()).unwrap());
        let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::new(128));
        let ingestor = Ingestor::new(db.clone(), embedder, IngestConfig::default());
        ingestor
            .ingest_text("mem:///a", "A", ARTICLE.into())
            .await
            .unwrap();
        let stats = db.stats().unwrap();
        (stats.documents, stats.chunks)
    };

    let db = Db::open(&path).unwrap();
    let stats = db.stats().unwrap();
    assert_eq!(stats.documents, doc_count);
    assert_eq!(stats.chunks, chunk_count);

    let query = HashEmbedder::new(128)
        .embed_query("analytical engine")
        .await
        .unwrap();
    assert!(!db.query(&Query::new().search(query, 3)).unwrap().is_empty());
}
