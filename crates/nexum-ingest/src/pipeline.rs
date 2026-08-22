//! The ingestion pipeline.
//!
//! Chunk, embed, extract, then write everything in one transaction. The write
//! is atomic on purpose: a half-ingested document — chunks present, edges
//! missing — is worse than no document, because retrieval would return it
//! while traversal would not.
//!
//! Every run mints a `PipelineRun` node carrying the chunker, embedding model,
//! extractor, and a hash over the effective config. Chunks and entities point
//! back at it, so any vector in the database can be traced to exactly what
//! produced it and when. That is the provenance guarantee the whole design
//! rests on, and it is cheap only because it is done at write time.

use crate::chunk::ChunkerConfig;
use crate::dedup::{self, Candidate, DedupConfig};
use crate::extract::{EntityExtractor, NullExtractor, RuleExtractor};
use crate::source::{self, SourceDocument};
use crate::{IngestError, Result};
use nexum_core::model::{Chunk, Document, EmbeddingMeta, Entity, PipelineRun};
use nexum_core::{
    ContentHash, Db, Edge, EdgeType, Metadata, Node, NodeId, PropertyValue, Timestamp,
    vector_namespace,
};
use nexum_embed::Embedder;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

/// How a run should behave.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IngestConfig {
    #[serde(default)]
    pub chunker: ChunkerConfig,
    /// Extract entities and semantic edges.
    #[serde(default = "default_true")]
    pub extract_entities: bool,
    #[serde(default)]
    pub dedup: DedupConfig,
    /// Skip a document whose content is byte-identical to its current version.
    #[serde(default = "default_true")]
    pub skip_unchanged: bool,
    /// Version string recorded on every run.
    #[serde(default = "default_pipeline_version")]
    pub pipeline_version: String,
    /// Ignore entities mentioned fewer than this many times in a chunk.
    #[serde(default = "default_min_mentions")]
    pub min_entity_mentions: u32,
}

fn default_true() -> bool {
    true
}
fn default_pipeline_version() -> String {
    format!("nexum-ingest/{}", env!("CARGO_PKG_VERSION"))
}
fn default_min_mentions() -> u32 {
    1
}

impl Default for IngestConfig {
    fn default() -> Self {
        IngestConfig {
            chunker: ChunkerConfig::default(),
            extract_entities: true,
            dedup: DedupConfig::default(),
            skip_unchanged: true,
            pipeline_version: default_pipeline_version(),
            min_entity_mentions: default_min_mentions(),
        }
    }
}

/// What happened to one document.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum IngestOutcome {
    /// Written as a new document.
    Created { version: u32 },
    /// Written as a new version superseding an existing one.
    Superseded { version: u32, supersedes: NodeId },
    /// Content identical to the current version; nothing written.
    Unchanged { existing: NodeId },
    /// Not ingested, with a reason.
    Skipped { reason: String },
}

impl IngestOutcome {
    /// Whether anything was written.
    pub fn wrote(&self) -> bool {
        matches!(
            self,
            IngestOutcome::Created { .. } | IngestOutcome::Superseded { .. }
        )
    }
}

/// The result of ingesting one document.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IngestReport {
    pub source_uri: String,
    pub title: String,
    pub outcome: IngestOutcome,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub document_id: Option<NodeId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<NodeId>,
    pub chunks: usize,
    pub entities: usize,
    pub edges: usize,
    pub aliases: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_model: Option<String>,
    pub duration_ms: u64,
}

impl IngestReport {
    fn skipped(source_uri: String, title: String, reason: impl Into<String>) -> Self {
        IngestReport {
            source_uri,
            title,
            outcome: IngestOutcome::Skipped {
                reason: reason.into(),
            },
            document_id: None,
            run_id: None,
            chunks: 0,
            entities: 0,
            edges: 0,
            aliases: 0,
            embedding_model: None,
            duration_ms: 0,
        }
    }
}

/// Runs the ingestion pipeline against a database.
pub struct Ingestor {
    db: Arc<Db>,
    embedder: Arc<dyn Embedder>,
    extractor: Arc<dyn EntityExtractor>,
    config: IngestConfig,
}

impl Ingestor {
    /// Build an ingestor. The extractor is chosen from the config.
    pub fn new(db: Arc<Db>, embedder: Arc<dyn Embedder>, config: IngestConfig) -> Self {
        let extractor: Arc<dyn EntityExtractor> = if config.extract_entities {
            Arc::new(RuleExtractor::new(config.min_entity_mentions, true))
        } else {
            Arc::new(NullExtractor)
        };
        Ingestor {
            db,
            embedder,
            extractor,
            config,
        }
    }

    /// Swap in a different entity extractor — an LLM or NER service.
    pub fn with_extractor(mut self, extractor: Arc<dyn EntityExtractor>) -> Self {
        self.extractor = extractor;
        self
    }

    pub fn config(&self) -> &IngestConfig {
        &self.config
    }

    pub fn embedder(&self) -> &dyn Embedder {
        self.embedder.as_ref()
    }

    /// The vector namespace this ingestor writes into.
    pub fn namespace(&self) -> String {
        vector_namespace(self.embedder.model_id(), self.embedder.dim())
    }

    /// Hash over everything that affects the output, so two runs with the same
    /// hash produced comparable data and two runs without did not.
    fn config_hash(&self) -> ContentHash {
        let fingerprint = format!(
            "chunker={}\nmodel={}\ndim={}\nextractor={}\npipeline={}\ndedup={}:{}\n",
            self.config.chunker.name(),
            self.embedder.model_id(),
            self.embedder.dim(),
            self.extractor.name(),
            self.config.pipeline_version,
            self.config.dedup.enabled,
            self.config.dedup.threshold,
        );
        ContentHash::of(fingerprint.as_bytes())
    }

    /// Ingest raw text under a stable URI.
    pub async fn ingest_text(
        &self,
        source_uri: impl Into<String>,
        title: impl Into<String>,
        text: String,
    ) -> Result<IngestReport> {
        self.ingest_document(SourceDocument::from_text(source_uri, title, text))
            .await
    }

    /// Ingest one file.
    pub async fn ingest_file(&self, path: &Path) -> Result<IngestReport> {
        match SourceDocument::from_path(path) {
            Ok(doc) => self.ingest_document(doc).await,
            // An unreadable file in a batch should be reported, not fatal.
            Err(IngestError::Unsupported { path, reason }) => {
                Ok(IngestReport::skipped(path.clone(), path, reason))
            }
            Err(e) => Err(e),
        }
    }

    /// Ingest a file or directory tree.
    pub async fn ingest_path(&self, path: &Path, recursive: bool) -> Result<Vec<IngestReport>> {
        let files = source::discover(path, recursive)?;
        let mut reports = Vec::with_capacity(files.len());
        for file in files {
            reports.push(self.ingest_file(&file).await?);
        }
        Ok(reports)
    }

    /// The core path: chunk, embed, extract, write.
    pub async fn ingest_document(&self, doc: SourceDocument) -> Result<IngestReport> {
        let started = std::time::Instant::now();

        // Where does this document stand relative to what is already stored?
        let previous = self.db.head_document(&doc.source_uri)?;
        let previous_document = match previous {
            Some(id) => self
                .db
                .get_node(id)?
                .and_then(|n| n.as_document().ok().cloned()),
            None => None,
        };

        if let Some(existing) = &previous_document
            && self.config.skip_unchanged
            && existing.content_hash == doc.content_hash
        {
            return Ok(IngestReport {
                source_uri: doc.source_uri,
                title: doc.title,
                outcome: IngestOutcome::Unchanged {
                    existing: existing.id,
                },
                document_id: Some(existing.id),
                run_id: None,
                chunks: 0,
                entities: 0,
                edges: 0,
                aliases: 0,
                embedding_model: Some(self.embedder.model_id().to_string()),
                duration_ms: started.elapsed().as_millis() as u64,
            });
        }

        let pieces = self.config.chunker.split(&doc.text);
        if pieces.is_empty() {
            return Ok(IngestReport::skipped(
                doc.source_uri,
                doc.title,
                "document has no text content",
            ));
        }

        // Embedding is the slow, fallible step — do it before opening the
        // transaction so a provider failure leaves nothing half-written.
        let texts: Vec<String> = pieces.iter().map(|c| c.text.clone()).collect();
        let embeddings = self.embedder.embed(&texts).await?;

        let now = Timestamp::now();
        let namespace = self.namespace();

        let run = PipelineRun {
            id: NodeId::new(),
            pipeline_version: self.config.pipeline_version.clone(),
            embedding_model: self.embedder.model_id().to_string(),
            run_at: now,
            config_hash: self.config_hash(),
            chunker: self.config.chunker.name(),
            metadata: Metadata::from([
                (
                    "extractor".to_string(),
                    PropertyValue::Text(self.extractor.name().to_string()),
                ),
                (
                    "embedding_dim".to_string(),
                    PropertyValue::Int(self.embedder.dim() as i64),
                ),
                (
                    "vector_namespace".to_string(),
                    PropertyValue::Text(namespace.clone()),
                ),
            ]),
        };

        let version = previous_document.as_ref().map_or(1, |d| d.version + 1);
        let document = Document {
            id: NodeId::new(),
            title: doc.title.clone(),
            source_uri: doc.source_uri.clone(),
            content_hash: doc.content_hash.clone(),
            created_at: now,
            version,
            supersedes_id: previous_document.as_ref().map(|d| d.id),
            run_id: Some(run.id),
            metadata: Metadata::from([(
                "chunk_count".to_string(),
                PropertyValue::Int(pieces.len() as i64),
            )]),
        };

        let mut tx = self.db.begin();
        let mut edges = 0usize;
        tx.put_node(run.clone());
        tx.put_node(document.clone());

        // The supersession edge points from new to old, which is what makes
        // "no inbound SUPERSEDES" mean "this is the current version".
        if let Some(prior) = &previous_document {
            tx.put_edge(
                Edge::new(document.id, prior.id, EdgeType::Supersedes)
                    .with("from_version", prior.version)
                    .with("to_version", version),
            );
            edges += 1;
        }

        let mut previous_chunk: Option<NodeId> = None;
        let mut entity_mentions: BTreeMap<NodeId, EntityAccumulator> = BTreeMap::new();
        let mut relation_pairs: BTreeMap<(NodeId, NodeId), (String, f32, u32)> = BTreeMap::new();

        for (piece, vector) in pieces.iter().zip(embeddings.vectors.iter()) {
            let chunk = Chunk {
                id: NodeId::new(),
                document_id: document.id,
                text: piece.text.clone(),
                chunk_index: piece.index,
                token_count: piece.estimated_tokens,
                embeddings: vec![EmbeddingMeta {
                    model: self.embedder.model_id().to_string(),
                    dim: self.embedder.dim(),
                    run_id: Some(run.id),
                    embedded_at: now,
                }],
                created_at: now,
                metadata: Metadata::new(),
            };

            tx.put_node(chunk.clone());
            tx.put_vector(&namespace, chunk.id, vector.clone());

            // Structural and provenance edges. PART_OF and DERIVED_FROM look
            // redundant but are not: one is containment, the other is lineage,
            // and a chunk can be derived from a document it is not part of
            // (a summary, say) once more pipelines exist.
            tx.put_edge(Edge::new(chunk.id, document.id, EdgeType::PartOf));
            tx.put_edge(Edge::new(chunk.id, document.id, EdgeType::DerivedFrom));
            tx.put_edge(Edge::new(chunk.id, run.id, EdgeType::EmbeddedBy));
            edges += 3;

            if let Some(prior) = previous_chunk {
                tx.put_edge_symmetric(Edge::new(prior, chunk.id, EdgeType::Precedes));
                edges += 2;
            }
            previous_chunk = Some(chunk.id);

            if self.config.extract_entities {
                let extraction = self.extractor.extract(&piece.text).await?;
                let mut in_chunk: BTreeMap<NodeId, ()> = BTreeMap::new();

                for entity in &extraction.entities {
                    let id = dedup::canonical_entity_id(&entity.name, &entity.entity_type);
                    in_chunk.insert(id, ());
                    let accumulator =
                        entity_mentions
                            .entry(id)
                            .or_insert_with(|| EntityAccumulator {
                                name: entity.name.clone(),
                                entity_type: entity.entity_type.clone(),
                                mentions: 0,
                                confidence: entity.confidence,
                            });
                    accumulator.mentions += entity.mentions;
                    // Keep the longest surface form seen anywhere in the run.
                    if entity.name.len() > accumulator.name.len() {
                        accumulator.name = entity.name.clone();
                    }
                    accumulator.confidence = accumulator.confidence.max(entity.confidence);

                    tx.put_edge(
                        Edge::new(chunk.id, id, EdgeType::Mentions)
                            .with("mentions", entity.mentions)
                            .with("confidence", entity.confidence),
                    );
                    edges += 1;
                }

                for relation in &extraction.relations {
                    let from = dedup::canonical_entity_id(
                        &relation.from,
                        &type_of(&extraction.entities, &relation.from),
                    );
                    let to = dedup::canonical_entity_id(
                        &relation.to,
                        &type_of(&extraction.entities, &relation.to),
                    );
                    if from == to {
                        continue;
                    }
                    // Order the pair so A-B and B-A accumulate together rather
                    // than producing two half-counted edges.
                    let key = if from < to { (from, to) } else { (to, from) };
                    let entry = relation_pairs.entry(key).or_insert((
                        relation.relation_type.clone(),
                        relation.confidence,
                        0,
                    ));
                    entry.2 += 1;
                }
            }
        }

        // Entity nodes are content-addressed, so re-ingesting the same name
        // updates one node instead of creating a duplicate. Merge with what is
        // already stored rather than resetting its counts.
        let entity_count = entity_mentions.len();
        for (id, accumulator) in &entity_mentions {
            let existing = self.db.get_node(*id)?;
            let (created_at, prior_mentions) = match &existing {
                Some(Node::Entity(e)) => (
                    e.created_at,
                    e.metadata
                        .get("mentions")
                        .and_then(PropertyValue::as_i64)
                        .unwrap_or(0),
                ),
                _ => (now, 0),
            };

            tx.put_node(Entity {
                id: *id,
                name: accumulator.name.clone(),
                entity_type: accumulator.entity_type.clone(),
                canonical_id: None,
                embeddings: Vec::new(),
                created_at,
                metadata: Metadata::from([
                    (
                        "mentions".to_string(),
                        PropertyValue::Int(prior_mentions + accumulator.mentions as i64),
                    ),
                    (
                        "confidence".to_string(),
                        PropertyValue::Float(accumulator.confidence as f64),
                    ),
                ]),
            });
            tx.put_edge(Edge::new(*id, run.id, EdgeType::ExtractedBy));
            edges += 1;
        }

        for ((from, to), (relation_type, confidence, count)) in &relation_pairs {
            tx.put_edge(
                Edge::new(*from, *to, EdgeType::RelatesTo)
                    .with("relation_type", relation_type.as_str())
                    .with("confidence", *confidence)
                    .with("co_occurrences", *count),
            );
            edges += 1;
        }

        // Fuzzy dedup runs last, over the entities this run touched, and links
        // rather than merges — a wrong fold stays visible and reversible.
        let candidates: Vec<Candidate> = entity_mentions
            .iter()
            .map(|(id, a)| Candidate {
                id: *id,
                name: a.name.clone(),
                entity_type: a.entity_type.clone(),
                mentions: a.mentions,
            })
            .collect();
        let aliases = dedup::find_aliases(&candidates, self.config.dedup);

        for alias in &aliases {
            if let Some(accumulator) = entity_mentions.get(&alias.alias_id) {
                tx.put_node(Entity {
                    id: alias.alias_id,
                    name: accumulator.name.clone(),
                    entity_type: accumulator.entity_type.clone(),
                    canonical_id: Some(alias.canonical_id),
                    embeddings: Vec::new(),
                    created_at: now,
                    metadata: Metadata::from([
                        (
                            "mentions".to_string(),
                            PropertyValue::Int(accumulator.mentions as i64),
                        ),
                        (
                            "alias_of".to_string(),
                            PropertyValue::Text(alias.canonical_name.clone()),
                        ),
                    ]),
                });
            }
            tx.put_edge(
                Edge::new(alias.alias_id, alias.canonical_id, EdgeType::RelatesTo)
                    .with("relation_type", "alias_of")
                    .with("similarity", alias.similarity),
            );
            edges += 1;
        }

        tx.commit()?;

        let outcome = match &previous_document {
            Some(prior) => IngestOutcome::Superseded {
                version,
                supersedes: prior.id,
            },
            None => IngestOutcome::Created { version },
        };

        Ok(IngestReport {
            source_uri: doc.source_uri,
            title: doc.title,
            outcome,
            document_id: Some(document.id),
            run_id: Some(run.id),
            chunks: pieces.len(),
            entities: entity_count,
            edges,
            aliases: aliases.len(),
            embedding_model: Some(self.embedder.model_id().to_string()),
            duration_ms: started.elapsed().as_millis() as u64,
        })
    }
}

/// Running totals for one entity across a document's chunks.
struct EntityAccumulator {
    name: String,
    entity_type: String,
    mentions: u32,
    confidence: f32,
}

/// The type the extractor assigned to a name, so a relation's endpoints get
/// the same derived ID the entity itself did.
fn type_of(entities: &[crate::extract::ExtractedEntity], name: &str) -> String {
    entities
        .iter()
        .find(|e| e.name == name)
        .map(|e| e.entity_type.clone())
        .unwrap_or_else(|| crate::extract::entity_type::CONCEPT.to_string())
}
