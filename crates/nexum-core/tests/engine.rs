//! End-to-end tests for the engine: ingest, query, versioning, recovery.

use nexum_core::model::{Chunk, Document, EmbeddingMeta, Entity, PipelineRun};
use nexum_core::{
    ContentHash, Db, DbConfig, Direction, Edge, EdgeType, Metadata, Node, NodeId, NodeKind,
    Predicate, PropertyValue, Query, Timestamp, vector_namespace,
};

const MODEL: &str = "test-model";
const DIM: usize = 8;

fn namespace() -> String {
    vector_namespace(MODEL, DIM)
}

/// A deterministic vector derived from `seed`.
///
/// Spread across every dimension rather than one-hot, so distinct seeds are
/// genuinely distinguishable — a one-hot scheme collides as soon as two seeds
/// share a residue mod DIM.
fn vector(seed: usize) -> Vec<f32> {
    let mut state = (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..DIM)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / 8_388_608.0) - 1.0
        })
        .collect()
}

/// Stable seed for a chunk's text.
fn seed_of(text: &str) -> usize {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in text.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash as usize
}

struct Fixture {
    _dir: tempfile::TempDir,
    db: Db,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::create(dir.path().join("db"), DbConfig::default()).unwrap();
        Fixture { _dir: dir, db }
    }
}

/// Write a document with `chunk_count` embedded chunks, wired up with the
/// structural and provenance edges the ingest pipeline would create.
fn ingest(
    db: &Db,
    source_uri: &str,
    title: &str,
    body: &[&str],
    supersedes: Option<NodeId>,
) -> (NodeId, Vec<NodeId>) {
    let run = PipelineRun {
        id: NodeId::new(),
        pipeline_version: "test".into(),
        embedding_model: MODEL.into(),
        run_at: Timestamp::now(),
        config_hash: ContentHash::of(b"cfg"),
        chunker: "fixed".into(),
        metadata: Metadata::new(),
    };

    let version = supersedes.map_or(1, |prior| {
        db.get_node(prior)
            .unwrap()
            .and_then(|n| n.as_document().ok().map(|d| d.version + 1))
            .unwrap_or(2)
    });

    let doc = Document {
        id: NodeId::new(),
        title: title.into(),
        source_uri: source_uri.into(),
        content_hash: ContentHash::of(body.concat().as_bytes()),
        created_at: Timestamp::now(),
        version,
        supersedes_id: supersedes,
        run_id: Some(run.id),
        metadata: Metadata::from([("lang".into(), PropertyValue::Text("en".into()))]),
    };

    let mut tx = db.begin();
    tx.put_node(run.clone());
    tx.put_node(doc.clone());
    if let Some(prior) = supersedes {
        tx.put_edge(Edge::new(doc.id, prior, EdgeType::Supersedes));
    }

    let mut chunk_ids = Vec::new();
    let mut previous: Option<NodeId> = None;
    for (i, text) in body.iter().enumerate() {
        let chunk = Chunk {
            id: NodeId::new(),
            document_id: doc.id,
            text: (*text).to_string(),
            chunk_index: i as u32,
            token_count: text.split_whitespace().count() as u32,
            embeddings: vec![EmbeddingMeta {
                model: MODEL.into(),
                dim: DIM,
                run_id: Some(run.id),
                embedded_at: Timestamp::now(),
            }],
            created_at: Timestamp::now(),
            metadata: Metadata::new(),
        };
        chunk_ids.push(chunk.id);

        tx.put_node(chunk.clone());
        tx.put_edge(Edge::new(chunk.id, doc.id, EdgeType::PartOf));
        tx.put_edge(Edge::new(chunk.id, doc.id, EdgeType::DerivedFrom));
        tx.put_edge(Edge::new(chunk.id, run.id, EdgeType::EmbeddedBy));
        if let Some(prev) = previous {
            tx.put_edge_symmetric(Edge::new(prev, chunk.id, EdgeType::Precedes));
        }
        previous = Some(chunk.id);

        // Vectors are seeded off a hash of the text so identical text in two
        // versions lands in the same place, which the version tests rely on.
        tx.put_vector(&namespace(), chunk.id, vector(seed_of(text)));
    }
    tx.commit().unwrap();
    (doc.id, chunk_ids)
}

#[test]
fn create_then_open_roundtrips() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db");
    let doc_id = {
        let db = Db::create(&path, DbConfig::default()).unwrap();
        let (doc_id, _) = ingest(&db, "file:///a.txt", "A", &["hello world"], None);
        doc_id
    };

    let db = Db::open(&path).unwrap();
    let node = db
        .get_node(doc_id)
        .unwrap()
        .expect("document should survive");
    assert_eq!(node.as_document().unwrap().title, "A");
}

#[test]
fn opening_a_missing_database_explains_itself() {
    let dir = tempfile::tempdir().unwrap();
    let err = Db::open(dir.path().join("nope")).unwrap_err().to_string();
    assert!(err.contains("nexum init"), "got: {err}");
}

#[test]
fn creating_over_an_existing_database_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db");
    let _db = Db::create(&path, DbConfig::default()).unwrap();
    assert!(Db::create(&path, DbConfig::default()).is_err());
}

#[test]
fn search_returns_the_nearest_chunk() {
    let f = Fixture::new();
    let (_, chunks) = ingest(
        &f.db,
        "file:///a.txt",
        "A",
        &["alpha", "beta text here", "gamma"],
        None,
    );

    // Query with the exact vector of the second chunk.
    let result =
        f.db.query(&Query::new().search(vector(seed_of("beta text here")), 3))
            .unwrap();

    assert!(!result.is_empty());
    assert_eq!(result.nodes[0].id(), chunks[1]);
    assert!(result.nodes[0].score.unwrap() > 0.99);
    // Scores must be present and descending.
    let scores: Vec<f32> = result.nodes.iter().filter_map(|n| n.score).collect();
    assert_eq!(scores.len(), result.len());
    assert!(scores.windows(2).all(|w| w[0] >= w[1]));
}

#[test]
fn traversal_reaches_the_document_from_a_chunk() {
    let f = Fixture::new();
    let (doc, chunks) = ingest(&f.db, "file:///a.txt", "A", &["one", "two"], None);

    let result =
        f.db.query(
            &Query::new()
                .seed([chunks[0]])
                .traverse(vec![EdgeType::PartOf], 1),
        )
        .unwrap();

    assert_eq!(result.len(), 1);
    assert_eq!(result.nodes[0].id(), doc);
    assert_eq!(result.nodes[0].hops, Some(1));
    assert_eq!(result.nodes[0].via, Some(chunks[0]));
    assert_eq!(result.nodes[0].via_edge, Some(EdgeType::PartOf));
}

#[test]
fn traversal_hop_count_is_the_shortest_path() {
    let f = Fixture::new();
    let (_, chunks) = ingest(&f.db, "file:///a.txt", "A", &["a", "b", "c", "d"], None);

    let result =
        f.db.query(&Query::new().seed([chunks[0]]).traverse_with(
            vec![EdgeType::Precedes],
            3,
            Direction::Out,
            false,
        ))
        .unwrap();

    let hops: std::collections::HashMap<NodeId, usize> = result
        .nodes
        .iter()
        .map(|n| (n.id(), n.hops.unwrap()))
        .collect();
    assert_eq!(hops[&chunks[1]], 1);
    assert_eq!(hops[&chunks[2]], 2);
    assert_eq!(hops[&chunks[3]], 3);
}

#[test]
fn precedes_and_follows_are_both_written() {
    let f = Fixture::new();
    let (_, chunks) = ingest(&f.db, "file:///a.txt", "A", &["a", "b"], None);

    let forward =
        f.db.neighbors(chunks[0], &[EdgeType::Precedes], Direction::Out)
            .unwrap();
    assert_eq!(forward.len(), 1);
    assert_eq!(forward[0].id(), chunks[1]);

    let backward =
        f.db.neighbors(chunks[1], &[EdgeType::Follows], Direction::Out)
            .unwrap();
    assert_eq!(backward.len(), 1);
    assert_eq!(backward[0].id(), chunks[0]);
}

#[test]
fn max_hops_bounds_the_expansion() {
    let f = Fixture::new();
    let (_, chunks) = ingest(
        &f.db,
        "file:///a.txt",
        "A",
        &["a", "b", "c", "d", "e"],
        None,
    );

    let result =
        f.db.query(&Query::new().seed([chunks[0]]).traverse_with(
            vec![EdgeType::Precedes],
            2,
            Direction::Out,
            false,
        ))
        .unwrap();
    assert_eq!(result.len(), 2, "two hops should reach exactly two chunks");
}

#[test]
fn the_full_pipeline_composes_search_traverse_and_filter() {
    let f = Fixture::new();
    let (doc, _) = ingest(
        &f.db,
        "file:///a.txt",
        "A",
        &["alpha one", "beta two", "gamma three"],
        None,
    );

    let seed = seed_of("beta two");
    let result =
        f.db.query(
            &Query::new()
                .search(vector(seed), 3)
                .traverse(vec![EdgeType::PartOf], 1)
                .of_kind(NodeKind::Document)
                .latest(),
        )
        .unwrap();

    assert_eq!(result.len(), 1);
    assert_eq!(result.nodes[0].id(), doc);
    // Every stage should be accounted for.
    let stages: Vec<&str> = result
        .stats
        .stages
        .iter()
        .map(|s| s.stage.as_str())
        .collect();
    assert_eq!(stages, vec!["search", "traverse", "filter", "filter"]);
    assert!(result.stats.edges_traversed > 0);
}

#[test]
fn a_search_after_an_earlier_stage_is_scoped_to_it() {
    let f = Fixture::new();
    let (_, a_chunks) = ingest(&f.db, "file:///a.txt", "A", &["shared text"], None);
    ingest(&f.db, "file:///b.txt", "B", &["shared text"], None);

    // Both documents contain identical text, so both chunks share a vector.
    let seed = seed_of("shared text");
    let unscoped = f.db.query(&Query::new().search(vector(seed), 10)).unwrap();
    assert_eq!(
        unscoped.len(),
        2,
        "both chunks should match without a scope"
    );

    let scoped =
        f.db.query(&Query::new().seed(a_chunks.clone()).search(vector(seed), 10))
            .unwrap();
    assert_eq!(scoped.len(), 1);
    assert_eq!(scoped.nodes[0].id(), a_chunks[0]);
}

#[test]
fn reingestion_supersedes_the_previous_version() {
    let f = Fixture::new();
    let (v1, v1_chunks) = ingest(&f.db, "file:///a.txt", "A v1", &["original text"], None);
    let (v2, _) = ingest(&f.db, "file:///a.txt", "A v2", &["revised text"], Some(v1));

    // Both versions still exist.
    assert!(f.db.get_node(v1).unwrap().is_some());
    assert!(f.db.get_node(v2).unwrap().is_some());

    // But only the newest counts as latest.
    let latest =
        f.db.query(&Query::new().seed_kind(NodeKind::Document).latest())
            .unwrap();
    assert_eq!(latest.len(), 1);
    assert_eq!(latest.nodes[0].id(), v2);

    // Chunks of the superseded document are excluded from default retrieval
    // but still reachable when explicitly asked for.
    let live_chunks =
        f.db.query(&Query::new().seed_kind(NodeKind::Chunk).latest())
            .unwrap();
    assert!(!live_chunks.ids().contains(&v1_chunks[0]));

    let superseded =
        f.db.query(
            &Query::new()
                .seed_kind(NodeKind::Chunk)
                .filter(Predicate::Superseded),
        )
        .unwrap();
    assert_eq!(superseded.ids(), v1_chunks);

    // And the head pointer follows the newest version.
    assert_eq!(f.db.head_document("file:///a.txt").unwrap(), Some(v2));
}

#[test]
fn history_returns_the_whole_chain_oldest_first() {
    let f = Fixture::new();
    let (v1, _) = ingest(&f.db, "file:///a.txt", "v1", &["one"], None);
    let (v2, _) = ingest(&f.db, "file:///a.txt", "v2", &["two"], Some(v1));
    let (v3, _) = ingest(&f.db, "file:///a.txt", "v3", &["three"], Some(v2));

    // Asking from any point in the chain returns the whole chain.
    for anchor in [v1, v2, v3] {
        let chain = f.db.history(anchor).unwrap();
        let ids: Vec<NodeId> = chain.iter().map(Node::id).collect();
        assert_eq!(ids, vec![v1, v2, v3], "history from {anchor}");
        let versions: Vec<u32> = chain
            .iter()
            .map(|n| n.as_document().unwrap().version)
            .collect();
        assert_eq!(versions, vec![1, 2, 3]);
    }
}

#[test]
fn search_can_be_restricted_to_the_latest_version() {
    let f = Fixture::new();
    let (v1, _) = ingest(&f.db, "file:///a.txt", "v1", &["same text"], None);
    ingest(&f.db, "file:///a.txt", "v2", &["same text"], Some(v1));

    let seed = seed_of("same text");
    let all = f.db.query(&Query::new().search(vector(seed), 10)).unwrap();
    assert_eq!(all.len(), 2, "both versions' chunks are indexed");

    let latest =
        f.db.query(&Query::new().search(vector(seed), 10).latest())
            .unwrap();
    assert_eq!(latest.len(), 1, "default retrieval sees one version");
}

#[test]
fn entities_link_chunks_across_documents() {
    let f = Fixture::new();
    let (_, a_chunks) = ingest(&f.db, "file:///a.txt", "A", &["about ada"], None);
    let (_, b_chunks) = ingest(&f.db, "file:///b.txt", "B", &["also about ada"], None);

    let entity = Entity {
        id: NodeId::new(),
        name: "Ada Lovelace".into(),
        entity_type: "person".into(),
        canonical_id: None,
        embeddings: Vec::new(),
        created_at: Timestamp::now(),
        metadata: Metadata::new(),
    };
    let mut tx = f.db.begin();
    tx.put_node(entity.clone());
    tx.put_edge(Edge::new(a_chunks[0], entity.id, EdgeType::Mentions));
    tx.put_edge(Edge::new(b_chunks[0], entity.id, EdgeType::Mentions));
    tx.commit().unwrap();

    // From one chunk, two hops through the entity reaches the other chunk.
    let result =
        f.db.query(
            &Query::new()
                .seed([a_chunks[0]])
                .traverse(vec![EdgeType::Mentions], 2)
                .of_kind(NodeKind::Chunk),
        )
        .unwrap();
    assert_eq!(result.ids(), vec![b_chunks[0]]);
}

#[test]
fn property_filters_match_builtin_fields_and_metadata() {
    let f = Fixture::new();
    ingest(&f.db, "file:///a.txt", "Annual Report", &["body"], None);
    ingest(&f.db, "file:///b.txt", "Meeting Notes", &["body"], None);

    let by_title =
        f.db.query(
            &Query::new()
                .seed_kind(NodeKind::Document)
                .filter(Predicate::eq("title", "Annual Report")),
        )
        .unwrap();
    assert_eq!(by_title.len(), 1);

    let by_prefix =
        f.db.query(
            &Query::new()
                .seed_kind(NodeKind::Document)
                .filter(Predicate::Property {
                    key: "source_uri".into(),
                    op: nexum_core::CompareOp::StartsWith,
                    value: PropertyValue::Text("file:///b".into()),
                }),
        )
        .unwrap();
    assert_eq!(by_prefix.len(), 1);

    // Metadata written by the fixture.
    let by_metadata =
        f.db.query(
            &Query::new()
                .seed_kind(NodeKind::Document)
                .filter(Predicate::eq("lang", "en")),
        )
        .unwrap();
    assert_eq!(by_metadata.len(), 2);

    // A missing property matches nothing except an explicit inequality.
    let missing =
        f.db.query(
            &Query::new()
                .seed_kind(NodeKind::Document)
                .filter(Predicate::eq("nonexistent", "x")),
        )
        .unwrap();
    assert!(missing.is_empty());
}

#[test]
fn boolean_predicates_compose() {
    let f = Fixture::new();
    ingest(&f.db, "file:///a.txt", "A", &["body"], None);
    ingest(&f.db, "file:///b.txt", "B", &["body"], None);

    let result =
        f.db.query(
            &Query::new()
                .seed_kind(NodeKind::Document)
                .filter(Predicate::and([
                    Predicate::kind(NodeKind::Document),
                    Predicate::or([Predicate::eq("title", "A"), Predicate::eq("title", "B")]),
                    Predicate::not(Predicate::eq("title", "B")),
                ])),
        )
        .unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result.nodes[0].node.as_document().unwrap().title, "A");
}

#[test]
fn deleting_a_node_hides_it_from_search_and_traversal() {
    let f = Fixture::new();
    let (doc, chunks) = ingest(&f.db, "file:///a.txt", "A", &["alpha", "beta"], None);

    let mut tx = f.db.begin();
    tx.delete_node(chunks[0]);
    tx.commit().unwrap();

    assert!(f.db.get_node(chunks[0]).unwrap().is_none());

    let seed = seed_of("alpha");
    let hits = f.db.query(&Query::new().search(vector(seed), 10)).unwrap();
    assert!(!hits.ids().contains(&chunks[0]));

    let neighbors =
        f.db.query(&Query::new().seed([doc]).traverse_with(
            vec![EdgeType::PartOf],
            1,
            Direction::In,
            false,
        ))
        .unwrap();
    assert_eq!(neighbors.ids(), vec![chunks[1]]);
}

#[test]
fn node_detail_shows_edges_in_both_directions() {
    let f = Fixture::new();
    let (doc, chunks) = ingest(&f.db, "file:///a.txt", "A", &["one", "two"], None);

    let detail = f.db.node_detail(chunks[0]).unwrap().unwrap();
    let outgoing: Vec<EdgeType> = detail.outgoing.iter().map(|e| e.edge.edge_type).collect();
    assert!(outgoing.contains(&EdgeType::PartOf));
    assert!(outgoing.contains(&EdgeType::DerivedFrom));
    assert!(outgoing.contains(&EdgeType::EmbeddedBy));

    let doc_detail = f.db.node_detail(doc).unwrap().unwrap();
    assert!(
        doc_detail
            .incoming
            .iter()
            .any(|e| e.edge.edge_type == EdgeType::PartOf && e.other_id == chunks[0])
    );
    // Labels come along for the viewer's inspector.
    assert!(doc_detail.incoming.iter().all(|e| e.other_label.is_some()));
}

#[test]
fn multiple_embedding_models_coexist() {
    let f = Fixture::new();
    let (_, chunks) = ingest(&f.db, "file:///a.txt", "A", &["text"], None);

    // Re-embed the same chunk with a second, differently-sized model.
    let other = vector_namespace("other-model", 4);
    let mut tx = f.db.begin();
    tx.put_vector(&other, chunks[0], vec![1.0, 0.0, 0.0, 0.0]);
    tx.commit().unwrap();

    let mut namespaces = f.db.namespaces();
    namespaces.sort();
    assert_eq!(namespaces, vec![other.clone(), namespace()]);

    // Each index answers independently, and neither is picked by accident.
    let hit =
        f.db.query(&Query::new().search_with(vec![1.0, 0.0, 0.0, 0.0], 5, Some(other)))
            .unwrap();
    assert_eq!(hit.ids(), vec![chunks[0]]);
    assert!(
        f.db.query(&Query::new().search(vec![1.0, 0.0, 0.0, 0.0], 5))
            .is_err(),
        "an ambiguous model must not be guessed"
    );
}

#[test]
fn stats_count_what_was_written() {
    let f = Fixture::new();
    let (v1, _) = ingest(&f.db, "file:///a.txt", "v1", &["a", "b"], None);
    ingest(&f.db, "file:///a.txt", "v2", &["c"], Some(v1));

    let stats = f.db.stats().unwrap();
    assert_eq!(stats.documents, 2);
    assert_eq!(stats.latest_documents, 1);
    assert_eq!(stats.chunks, 3);
    assert_eq!(stats.pipeline_runs, 2);
    assert_eq!(stats.namespaces[&namespace()].count, 3);
    assert_eq!(stats.edges_by_type[EdgeType::Supersedes.as_str()], 1);
    assert_eq!(stats.edges_by_type[EdgeType::PartOf.as_str()], 3);
    assert!(stats.store_bytes > 0);
    assert!(stats.applied_lsn > 0);
}

#[test]
fn data_survives_reopen_with_indexes_intact() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db");

    let (doc, chunks) = {
        let db = Db::create(&path, DbConfig::default()).unwrap();
        let out = ingest(&db, "file:///a.txt", "A", &["alpha", "beta"], None);
        db.flush().unwrap();
        out
    };

    let db = Db::open(&path).unwrap();
    assert!(db.get_node(doc).unwrap().is_some());

    // Vector search must work immediately, without a manual reindex.
    let seed = seed_of("alpha");
    let hits = db.query(&Query::new().search(vector(seed), 5)).unwrap();
    assert_eq!(hits.nodes[0].id(), chunks[0]);
}

/// A crash leaves committed vectors on disk but no matching graph snapshot.
/// The engine must notice and rebuild rather than serve an index that is
/// missing whatever was written after the last flush.
#[test]
fn a_stale_index_snapshot_is_rebuilt_on_open() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db");

    let chunks = {
        let db = Db::create(&path, DbConfig::default()).unwrap();
        let (_, chunks) = ingest(&db, "file:///a.txt", "A", &["alpha", "beta"], None);
        chunks
    };

    // Invalidate the snapshot watermark, which is exactly the state an
    // unclean shutdown leaves behind.
    {
        let store = nexum_core::store::Store::open(path.join("nexum.redb")).unwrap();
        let batch = store.write().unwrap();
        batch
            .set_meta_u64(nexum_core::db::META_INDEX_SNAPSHOT_LSN, 0)
            .unwrap();
        batch.commit().unwrap();
    }

    let db = Db::open(&path).unwrap();
    let seed = seed_of("beta");
    let hits = db.query(&Query::new().search(vector(seed), 5)).unwrap();
    assert_eq!(
        hits.nodes[0].id(),
        chunks[1],
        "search must work after an unclean shutdown"
    );
}

/// A crash between "logged" and "applied" must not lose the write. The log is
/// the record of intent; opening the database replays whatever the tables
/// never absorbed.
#[test]
fn writes_logged_but_not_applied_are_replayed_on_open() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db");
    {
        let db = Db::create(&path, DbConfig::default()).unwrap();
        ingest(&db, "file:///a.txt", "A", &["one"], None);
    }

    // Append straight to the log without touching the tables.
    let orphan = Document {
        id: NodeId::new(),
        title: "logged but not applied".into(),
        source_uri: "file:///orphan.txt".into(),
        content_hash: ContentHash::of(b"orphan"),
        created_at: Timestamp::now(),
        version: 1,
        supersedes_id: None,
        run_id: None,
        metadata: Metadata::new(),
    };
    let orphan_chunk = Chunk {
        id: NodeId::new(),
        document_id: orphan.id,
        text: "orphan body".into(),
        chunk_index: 0,
        token_count: 2,
        embeddings: vec![EmbeddingMeta {
            model: MODEL.into(),
            dim: DIM,
            run_id: None,
            embedded_at: Timestamp::now(),
        }],
        created_at: Timestamp::now(),
        metadata: Metadata::new(),
    };
    {
        let mut wal = nexum_core::wal::Wal::open(path.join("wal.log")).unwrap();
        wal.append(nexum_core::wal::WalOp::PutNode(Box::new(Node::Document(
            orphan.clone(),
        ))))
        .unwrap();
        wal.append(nexum_core::wal::WalOp::PutNode(Box::new(Node::Chunk(
            orphan_chunk.clone(),
        ))))
        .unwrap();
        wal.append(nexum_core::wal::WalOp::PutEdge(Box::new(Edge::new(
            orphan_chunk.id,
            orphan.id,
            EdgeType::PartOf,
        ))))
        .unwrap();
        wal.append(nexum_core::wal::WalOp::PutVector {
            namespace: namespace(),
            node_id: orphan_chunk.id,
            vector: vector(seed_of("orphan body")),
        })
        .unwrap();
        wal.sync().unwrap();
    }

    let db = Db::open(&path).unwrap();
    assert!(
        db.get_node(orphan.id).unwrap().is_some(),
        "the logged node should have been replayed into the tables"
    );
    assert_eq!(
        db.head_document("file:///orphan.txt").unwrap(),
        Some(orphan.id)
    );
    // The replayed vector must be in the searchable index, not just on disk.
    let hits = db
        .query(&Query::new().search(vector(seed_of("orphan body")), 5))
        .unwrap();
    assert_eq!(hits.nodes[0].id(), orphan_chunk.id);
    // And the replayed edge must be traversable.
    let reached = db
        .query(
            &Query::new()
                .seed([orphan_chunk.id])
                .traverse(vec![EdgeType::PartOf], 1),
        )
        .unwrap();
    assert_eq!(reached.ids(), vec![orphan.id]);
}

/// Replay must be idempotent: opening twice must not double-apply.
#[test]
fn replaying_twice_changes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db");
    {
        let db = Db::create(&path, DbConfig::default()).unwrap();
        ingest(&db, "file:///a.txt", "A", &["one", "two"], None);
    }

    let first = {
        let db = Db::open(&path).unwrap();
        db.stats().unwrap()
    };
    let second = {
        let db = Db::open(&path).unwrap();
        db.stats().unwrap()
    };
    assert_eq!(first.documents, second.documents);
    assert_eq!(first.chunks, second.chunks);
    assert_eq!(first.edges, second.edges);
    assert_eq!(first.applied_lsn, second.applied_lsn);
}

#[test]
fn the_write_ahead_log_records_every_mutation() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db");
    {
        let db = Db::create(&path, DbConfig::default()).unwrap();
        ingest(&db, "file:///a.txt", "A", &["one"], None);
    }

    let records = nexum_core::wal::Wal::read_all(path.join("wal.log")).unwrap();
    let kinds: Vec<&str> = records.iter().map(|r| r.op.kind()).collect();
    assert!(kinds.contains(&"put_node"));
    assert!(kinds.contains(&"put_edge"));
    assert!(kinds.contains(&"put_vector"));
    // LSNs must be gapless and increasing — the audit trail depends on it.
    for (i, record) in records.iter().enumerate() {
        assert_eq!(record.lsn, i as u64 + 1);
    }
}

#[test]
fn compaction_reclaims_space_without_losing_data() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db");
    let mut db = Db::create(&path, DbConfig::default()).unwrap();

    let mut kept = Vec::new();
    for i in 0..40 {
        let (doc, chunks) = ingest(
            &db,
            &format!("file:///{i}.txt"),
            &format!("doc {i}"),
            &["some body text"],
            None,
        );
        kept.push((doc, chunks));
    }
    db.flush().unwrap();
    let before = db.stats().unwrap();

    let report = db.compact().unwrap();
    assert!(report.wal_bytes_reclaimed > 0, "the log should shrink");

    let after = db.stats().unwrap();
    assert_eq!(before.documents, after.documents);
    assert_eq!(before.chunks, after.chunks);
    assert_eq!(before.edges, after.edges);

    // Everything is still readable and searchable.
    for (doc, chunks) in &kept {
        assert!(db.get_node(*doc).unwrap().is_some());
        assert!(db.get_node(chunks[0]).unwrap().is_some());
    }
    let seed = seed_of("some body text");
    assert!(
        !db.query(&Query::new().search(vector(seed), 5))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn duplicate_edges_do_not_accumulate() {
    let f = Fixture::new();
    let (doc, chunks) = ingest(&f.db, "file:///a.txt", "A", &["one"], None);

    let before = f.db.stats().unwrap().edges;
    let mut tx = f.db.begin();
    tx.put_edge(Edge::new(chunks[0], doc, EdgeType::PartOf));
    tx.commit().unwrap();
    assert_eq!(f.db.stats().unwrap().edges, before);
}

#[test]
fn an_empty_transaction_commits_cleanly() {
    let f = Fixture::new();
    let tx = f.db.begin();
    assert!(tx.is_empty());
    tx.commit().unwrap();
}

#[test]
fn a_dropped_transaction_writes_nothing() {
    let f = Fixture::new();
    let before = f.db.stats().unwrap().documents;
    {
        let mut tx = f.db.begin();
        tx.put_node(Document {
            id: NodeId::new(),
            title: "ghost".into(),
            source_uri: "file:///ghost".into(),
            content_hash: ContentHash::of(b"g"),
            created_at: Timestamp::now(),
            version: 1,
            supersedes_id: None,
            run_id: None,
            metadata: Metadata::new(),
        });
        // Dropped without commit.
    }
    assert_eq!(f.db.stats().unwrap().documents, before);
}

#[test]
fn limit_truncates_without_reordering() {
    let f = Fixture::new();
    ingest(
        &f.db,
        "file:///a.txt",
        "A",
        &["a", "b", "c", "d", "e"],
        None,
    );

    let all =
        f.db.query(&Query::new().seed_kind(NodeKind::Chunk))
            .unwrap();
    let limited =
        f.db.query(&Query::new().seed_kind(NodeKind::Chunk).limit(3))
            .unwrap();
    assert_eq!(limited.len(), 3);
    assert_eq!(limited.ids(), all.ids()[..3]);
}
