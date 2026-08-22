//! Exercises the HTTP API against a real server over a real socket.

use nexum_client::{ClientConfig, Nexum};
use nexum_embed::EmbedderConfig;
use nexum_server::{ServerConfig, bind};
use serde_json::{Value, json};
use std::sync::Arc;

struct TestServer {
    _dir: tempfile::TempDir,
    base: String,
    client: reqwest::Client,
}

impl TestServer {
    async fn start() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let config = ClientConfig::default().with_embedder(EmbedderConfig::Hash { dim: 128 });
        let nexum = Arc::new(Nexum::create(dir.path().join("db"), config).await.unwrap());

        nexum
            .ingest_text(
                "mem:///ada",
                "Ada Lovelace",
                "Ada Lovelace wrote the first algorithm intended for a machine. \
                 She collaborated with Charles Babbage on the Analytical Engine."
                    .into(),
            )
            .await
            .unwrap();
        nexum
            .ingest_text(
                "mem:///bread",
                "Sourdough",
                "Combine flour, water and salt. Let the dough rise overnight before baking.".into(),
            )
            .await
            .unwrap();

        // Port 0 lets the OS pick, so parallel tests never collide.
        let (addr, server) = bind(
            nexum,
            ServerConfig {
                addr: ([127, 0, 0, 1], 0).into(),
                permissive_cors: true,
            },
        )
        .await
        .unwrap();
        tokio::spawn(server);

        TestServer {
            _dir: dir,
            base: format!("http://{addr}"),
            client: reqwest::Client::new(),
        }
    }

    async fn get(&self, path: &str) -> (reqwest::StatusCode, Value) {
        let response = self
            .client
            .get(format!("{}{path}", self.base))
            .send()
            .await
            .unwrap();
        let status = response.status();
        let body = response.json().await.unwrap_or(Value::Null);
        (status, body)
    }

    async fn post(&self, path: &str, body: Value) -> (reqwest::StatusCode, Value) {
        let response = self
            .client
            .post(format!("{}{path}", self.base))
            .json(&body)
            .send()
            .await
            .unwrap();
        let status = response.status();
        let body = response.json().await.unwrap_or(Value::Null);
        (status, body)
    }

    /// Any document's ID, for endpoints that need one.
    async fn a_document_id(&self) -> String {
        let (_, body) = self.get("/api/documents").await;
        body["items"][0]["id"].as_str().unwrap().to_string()
    }
}

#[tokio::test]
async fn health_reports_the_engine_version() {
    let server = TestServer::start().await;
    let (status, body) = server.get("/health").await;
    assert!(status.is_success());
    assert_eq!(body["status"], "ok");
    assert_eq!(body["engine_version"], nexum_core::VERSION);
}

#[tokio::test]
async fn stats_counts_what_was_ingested() {
    let server = TestServer::start().await;
    let (status, body) = server.get("/api/stats").await;
    assert!(status.is_success());
    assert_eq!(body["documents"], 2);
    assert!(body["chunks"].as_u64().unwrap() >= 2);
    assert!(body["edges"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn config_reports_the_embedding_model() {
    let server = TestServer::start().await;
    let (_, body) = server.get("/api/config").await;
    assert_eq!(body["embedding_model"], "hash-bow-v1-128");
    assert_eq!(body["embedding_dimensions"], 128);
    assert!(
        body["namespaces"]
            .as_array()
            .unwrap()
            .contains(&json!("hash-bow-v1-128:128"))
    );
}

#[tokio::test]
async fn documents_are_listed_with_chunk_counts() {
    let server = TestServer::start().await;
    let (status, body) = server.get("/api/documents").await;
    assert!(status.is_success());
    assert_eq!(body["total"], 2);

    let items = body["items"].as_array().unwrap();
    assert_eq!(items.len(), 2);
    for item in items {
        assert!(item["chunk_count"].as_u64().unwrap() >= 1);
        assert_eq!(item["is_latest"], true);
        assert!(item["title"].is_string());
        assert!(item["source_uri"].is_string());
    }
}

#[tokio::test]
async fn document_listing_paginates() {
    let server = TestServer::start().await;
    let (_, body) = server.get("/api/documents?limit=1&offset=0").await;
    assert_eq!(body["items"].as_array().unwrap().len(), 1);
    assert_eq!(
        body["total"], 2,
        "total should reflect everything, not the page"
    );

    let (_, second) = server.get("/api/documents?limit=1&offset=1").await;
    assert_ne!(body["items"][0]["id"], second["items"][0]["id"]);

    // An absurd limit is clamped rather than honoured.
    let (status, body) = server.get("/api/documents?limit=999999").await;
    assert!(status.is_success());
    assert!(body["limit"].as_u64().unwrap() <= 1_000);
}

#[tokio::test]
async fn a_document_and_its_chunks_are_fetchable() {
    let server = TestServer::start().await;
    let id = server.a_document_id().await;

    let (status, document) = server.get(&format!("/api/documents/{id}")).await;
    assert!(status.is_success());
    assert_eq!(document["id"], id);

    let (status, chunks) = server.get(&format!("/api/documents/{id}/chunks")).await;
    assert!(status.is_success());
    let items = chunks["items"].as_array().unwrap();
    assert!(!items.is_empty());
    // Chunks come back in reading order.
    let indices: Vec<u64> = items
        .iter()
        .map(|c| c["chunk_index"].as_u64().unwrap())
        .collect();
    assert!(indices.windows(2).all(|w| w[0] < w[1]));
}

#[tokio::test]
async fn search_ranks_by_relevance() {
    let server = TestServer::start().await;
    let (status, body) = server
        .post(
            "/api/search",
            json!({ "query": "analytical engine algorithm", "top_k": 5 }),
        )
        .await;
    assert!(status.is_success(), "{body}");

    let results = body["results"].as_array().unwrap();
    assert!(!results.is_empty());
    assert_eq!(body["query_model"], "hash-bow-v1-128");

    let top = results[0]["node"]["text"].as_str().unwrap().to_lowercase();
    assert!(
        top.contains("engine") || top.contains("algorithm"),
        "got: {top}"
    );
    // Scores present and descending.
    let scores: Vec<f64> = results
        .iter()
        .map(|r| r["score"].as_f64().unwrap())
        .collect();
    assert!(scores.windows(2).all(|w| w[0] >= w[1]));
}

#[tokio::test]
async fn search_can_expand_across_the_graph() {
    let server = TestServer::start().await;
    let (status, body) = server
        .post(
            "/api/search",
            json!({
                "query": "analytical engine",
                "top_k": 3,
                "expand": { "edge_types": ["MENTIONS", "PART_OF"], "max_hops": 2 }
            }),
        )
        .await;
    assert!(status.is_success(), "{body}");
    let results = body["results"].as_array().unwrap();
    assert!(
        results.iter().any(|r| !r["hops"].is_null()),
        "expansion should add graph-reached nodes"
    );
}

#[tokio::test]
async fn search_rejects_ambiguous_and_empty_requests() {
    let server = TestServer::start().await;

    let (status, body) = server.post("/api/search", json!({})).await;
    assert_eq!(status, 400);
    assert!(body["error"].as_str().unwrap().contains("query"));

    let (status, _) = server
        .post("/api/search", json!({ "query": "x", "vector": [0.1, 0.2] }))
        .await;
    assert_eq!(status, 400);
}

#[tokio::test]
async fn search_accepts_a_precomputed_vector() {
    let server = TestServer::start().await;
    let (status, body) = server
        .post(
            "/api/search",
            json!({ "vector": vec![0.1f32; 128], "top_k": 2 }),
        )
        .await;
    assert!(status.is_success(), "{body}");
    assert!(body["results"].as_array().unwrap().len() <= 2);
}

#[tokio::test]
async fn a_wrong_sized_vector_is_a_client_error_not_a_server_fault() {
    let server = TestServer::start().await;
    let (status, body) = server
        .post("/api/search", json!({ "vector": [0.1, 0.2, 0.3] }))
        .await;
    assert_eq!(status, 400, "got {body}");
    assert_eq!(body["kind"], "bad_request");
}

#[tokio::test]
async fn traversal_walks_from_a_node() {
    let server = TestServer::start().await;
    let id = server.a_document_id().await;
    let (status, body) = server
        .post(
            "/api/traverse",
            json!({ "start_ids": [id], "edge_types": ["PART_OF"], "max_hops": 1, "direction": "in" }),
        )
        .await;
    assert!(status.is_success(), "{body}");
    let nodes = body["nodes"].as_array().unwrap();
    assert!(!nodes.is_empty());
    assert!(nodes.iter().all(|n| n["node"]["kind"] == "Chunk"));
}

#[tokio::test]
async fn traversal_validates_its_input() {
    let server = TestServer::start().await;
    let (status, _) = server
        .post("/api/traverse", json!({ "start_ids": [] }))
        .await;
    assert_eq!(status, 400);

    let (status, body) = server
        .post("/api/traverse", json!({ "start_ids": ["not-a-uuid"] }))
        .await;
    assert_eq!(status, 400);
    assert!(body["error"].as_str().unwrap().contains("node id"));
}

#[tokio::test]
async fn a_raw_query_pipeline_runs() {
    let server = TestServer::start().await;
    let (status, body) = server
        .post(
            "/api/query",
            json!({
                "stages": [
                    { "stage": "seed_kind", "kind": "document" },
                    { "stage": "filter", "predicate": { "type": "latest_version" } }
                ]
            }),
        )
        .await;
    assert!(status.is_success(), "{body}");
    assert_eq!(body["nodes"].as_array().unwrap().len(), 2);
    assert_eq!(body["stats"]["stages"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn an_empty_query_is_rejected() {
    let server = TestServer::start().await;
    let (status, _) = server.post("/api/query", json!({ "stages": [] })).await;
    assert_eq!(status, 400);
}

#[tokio::test]
async fn node_detail_includes_edges_in_both_directions() {
    let server = TestServer::start().await;
    let id = server.a_document_id().await;
    let (status, body) = server.get(&format!("/api/nodes/{id}")).await;
    assert!(status.is_success());
    assert_eq!(body["node"]["kind"], "Document");
    assert!(!body["incoming"].as_array().unwrap().is_empty());
    // Labels come along so the inspector can render without extra fetches.
    assert!(body["incoming"][0]["other_label"].is_string());
}

#[tokio::test]
async fn the_graph_endpoint_returns_nodes_and_links() {
    let server = TestServer::start().await;
    let id = server.a_document_id().await;
    let (status, body) = server.get(&format!("/api/graph/{id}?hops=2")).await;
    assert!(status.is_success(), "{body}");

    assert_eq!(body["center"], id);
    let nodes = body["nodes"].as_array().unwrap();
    let links = body["links"].as_array().unwrap();
    assert!(nodes.len() > 1);
    assert!(!links.is_empty());

    // The centre is at hop 0, and every link connects nodes that are present.
    assert_eq!(nodes[0]["hops"], 0);
    let ids: Vec<&str> = nodes.iter().map(|n| n["id"].as_str().unwrap()).collect();
    for link in links {
        assert!(ids.contains(&link["source"].as_str().unwrap()));
        assert!(ids.contains(&link["target"].as_str().unwrap()));
        assert!(
            link["class"].is_string(),
            "links carry their edge class for styling"
        );
    }
}

#[tokio::test]
async fn the_graph_endpoint_reports_truncation() {
    let server = TestServer::start().await;
    let id = server.a_document_id().await;
    let (_, body) = server.get(&format!("/api/graph/{id}?hops=3&limit=1")).await;
    assert_eq!(body["truncated"], true, "a capped result must say so");
}

#[tokio::test]
async fn the_projection_endpoint_returns_placed_points() {
    let server = TestServer::start().await;
    let (status, body) = server.get("/api/projection").await;
    assert!(status.is_success(), "{body}");

    let points = body["points"].as_array().unwrap();
    assert!(!points.is_empty());
    for point in points {
        assert!(point["x"].as_f64().unwrap().is_finite());
        assert!(point["y"].as_f64().unwrap().is_finite());
        // Enough metadata to colour and label the scatter plot.
        assert!(point["document_id"].is_string());
        assert!(point["preview"].is_string());
    }
    assert_eq!(body["namespace"], "hash-bow-v1-128:128");
}

#[tokio::test]
async fn the_projection_method_is_selectable_and_reported() {
    let server = TestServer::start().await;
    // A projection needs more than a couple of points to mean anything.
    for i in 0..8 {
        server
            .post(
                "/api/ingest",
                json!({
                    "source_uri": format!("mem:///doc{i}"),
                    "title": format!("Doc {i}"),
                    "text": format!("Document number {i} discussing topic {} at some length.", i % 3)
                }),
            )
            .await;
    }

    let (_, body) = server.get("/api/projection?method=pca").await;
    assert_eq!(body["method"], "pca");
    assert!(body["explained_variance"].is_number());

    let (_, body) = server.get("/api/projection?method=neighborhood").await;
    assert_eq!(body["method"], "neighborhood");
}

#[tokio::test]
async fn ingesting_over_http_creates_a_document() {
    let server = TestServer::start().await;
    let (status, body) = server
        .post(
            "/api/ingest",
            json!({
                "source_uri": "mem:///new",
                "title": "New Document",
                "text": "Fresh content ingested over the wire."
            }),
        )
        .await;
    assert!(status.is_success(), "{body}");
    let reports = body.as_array().unwrap();
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0]["outcome"], "created");

    let (_, stats) = server.get("/api/stats").await;
    assert_eq!(stats["documents"], 3);
}

#[tokio::test]
async fn reingesting_over_http_supersedes_and_history_shows_it() {
    let server = TestServer::start().await;
    server
        .post(
            "/api/ingest",
            json!({ "source_uri": "mem:///v", "title": "V", "text": "Version one text." }),
        )
        .await;
    let (_, body) = server
        .post(
            "/api/ingest",
            json!({ "source_uri": "mem:///v", "title": "V", "text": "Version two text, changed." }),
        )
        .await;
    assert_eq!(body[0]["outcome"], "superseded");

    let id = body[0]["document_id"].as_str().unwrap();
    let (status, history) = server.get(&format!("/api/documents/{id}/history")).await;
    assert!(status.is_success());
    let versions: Vec<u64> = history
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d["version"].as_u64().unwrap())
        .collect();
    assert_eq!(versions, vec![1, 2]);

    // The default listing shows only the current version.
    let (_, listed) = server.get("/api/documents").await;
    let uris: Vec<&str> = listed["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d["source_uri"].as_str().unwrap())
        .collect();
    assert_eq!(uris.iter().filter(|u| **u == "mem:///v").count(), 1);

    let (_, all) = server.get("/api/documents?include_superseded=true").await;
    assert_eq!(all["total"], 4);
}

#[tokio::test]
async fn unknown_ids_return_404_and_malformed_ids_return_400() {
    let server = TestServer::start().await;
    let missing = nexum_core::NodeId::new();

    let (status, body) = server.get(&format!("/api/nodes/{missing}")).await;
    assert_eq!(status, 404);
    assert_eq!(body["kind"], "not_found");

    let (status, body) = server.get("/api/nodes/not-a-uuid").await;
    assert_eq!(status, 400);
    assert_eq!(body["kind"], "bad_request");

    let (status, _) = server.get(&format!("/api/documents/{missing}")).await;
    assert_eq!(status, 404);
}

#[tokio::test]
async fn bad_enum_values_are_rejected_with_a_useful_message() {
    let server = TestServer::start().await;
    let id = server.a_document_id().await;

    let (status, body) = server
        .get(&format!("/api/nodes/{id}/neighbors?edges=NOT_A_TYPE"))
        .await;
    assert_eq!(status, 400);
    assert!(body["error"].as_str().unwrap().contains("edge type"));

    let (status, body) = server
        .get(&format!("/api/nodes/{id}/neighbors?direction=sideways"))
        .await;
    assert_eq!(status, 400);
    assert!(body["error"].as_str().unwrap().contains("direction"));
}

#[tokio::test]
async fn cors_headers_are_present_for_the_desktop_viewer() {
    let server = TestServer::start().await;
    let response = server
        .client
        .get(format!("{}/health", server.base))
        .header("Origin", "tauri://localhost")
        .send()
        .await
        .unwrap();
    assert!(
        response
            .headers()
            .contains_key("access-control-allow-origin"),
        "the viewer runs on a different origin and needs CORS"
    );
}
