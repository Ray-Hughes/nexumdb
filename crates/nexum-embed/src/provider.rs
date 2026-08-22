//! The embedder interface and the configuration that selects one.

use crate::{EmbedError, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// A batch of embeddings, plus what producing them cost.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingBatch {
    pub vectors: Vec<Vec<f32>>,
    #[serde(default)]
    pub usage: Usage,
}

impl EmbeddingBatch {
    pub fn new(vectors: Vec<Vec<f32>>) -> Self {
        EmbeddingBatch {
            vectors,
            usage: Usage::default(),
        }
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
}

/// Token accounting, where the provider reports it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u64,
    /// Requests actually issued, which is how batching shows up in a run's
    /// provenance record.
    #[serde(default)]
    pub requests: u64,
}

impl Usage {
    pub fn merge(&mut self, other: Usage) {
        self.prompt_tokens += other.prompt_tokens;
        self.requests += other.requests;
    }
}

/// Something that turns text into vectors.
///
/// Implementations must be deterministic for a given model version: the whole
/// provenance model rests on "same model, same input, same vector", and a
/// provider that quietly changes behaviour makes stored embeddings
/// incomparable without any signal that it happened.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Model identifier written into `PipelineRun` and the vector namespace.
    /// Include a version when the provider exposes one.
    fn model_id(&self) -> &str;

    /// Output dimensionality.
    fn dim(&self) -> usize;

    /// Largest batch this provider accepts in one call.
    fn max_batch(&self) -> usize {
        64
    }

    /// Human-readable description for `nexum stats` and the viewer.
    fn describe(&self) -> String {
        format!("{} ({} dimensions)", self.model_id(), self.dim())
    }

    /// Embed one batch. Callers should prefer [`Embedder::embed`], which
    /// handles splitting and validation.
    async fn embed_batch(&self, texts: &[String]) -> Result<EmbeddingBatch>;

    /// Embed a query.
    ///
    /// Some models are trained with distinct query and passage prefixes;
    /// providers that need one override this. The default treats both alike.
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let batch = self.embed(std::slice::from_ref(&text.to_string())).await?;
        batch
            .vectors
            .into_iter()
            .next()
            .ok_or_else(|| EmbedError::CountMismatch {
                provider: self.model_id().to_string(),
                expected: 1,
                got: 0,
            })
    }

    /// Embed any number of texts, splitting into provider-sized batches and
    /// checking that what comes back matches what went in.
    async fn embed(&self, texts: &[String]) -> Result<EmbeddingBatch> {
        if texts.is_empty() {
            return Ok(EmbeddingBatch::default());
        }

        let mut vectors = Vec::with_capacity(texts.len());
        let mut usage = Usage::default();

        for chunk in texts.chunks(self.max_batch().max(1)) {
            let batch = self.embed_batch(chunk).await?;
            if batch.vectors.len() != chunk.len() {
                return Err(EmbedError::CountMismatch {
                    provider: self.model_id().to_string(),
                    expected: chunk.len(),
                    got: batch.vectors.len(),
                });
            }
            // A provider silently changing dimension would corrupt an index
            // that is namespaced by dimension, so it is a hard error.
            for vector in &batch.vectors {
                if vector.len() != self.dim() {
                    return Err(EmbedError::DimensionMismatch {
                        provider: self.model_id().to_string(),
                        expected: self.dim(),
                        got: vector.len(),
                    });
                }
            }
            vectors.extend(batch.vectors);
            usage.merge(batch.usage);
        }

        Ok(EmbeddingBatch { vectors, usage })
    }
}

/// Which provider to use, and how to reach it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "provider", rename_all = "snake_case")]
pub enum EmbedderConfig {
    /// Deterministic hashed bag-of-words. No model, no network, no API key —
    /// useful for tests, demos, and getting a pipeline working before
    /// committing to a real model.
    Hash {
        #[serde(default = "default_hash_dim")]
        dim: usize,
    },
    /// A sentence-transformer running locally via ONNX Runtime.
    Local {
        #[serde(default = "default_local_model")]
        model: String,
    },
    /// OpenAI's embeddings API, or anything that speaks its protocol.
    OpenAi {
        #[serde(default = "default_openai_model")]
        model: String,
        /// Read from `OPENAI_API_KEY` when omitted.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        api_key: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        base_url: Option<String>,
        /// Request a reduced dimensionality, for models that support it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        dimensions: Option<usize>,
    },
    /// A model served by a local Ollama instance.
    Ollama {
        model: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        base_url: Option<String>,
    },
}

fn default_hash_dim() -> usize {
    256
}
fn default_local_model() -> String {
    crate::local_default_model().to_string()
}
fn default_openai_model() -> String {
    "text-embedding-3-small".to_string()
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        // The zero-config default is a real local model where the build
        // includes one, and the deterministic hasher otherwise — so a fresh
        // install always works offline.
        #[cfg(feature = "local")]
        {
            EmbedderConfig::Local {
                model: default_local_model(),
            }
        }
        #[cfg(not(feature = "local"))]
        {
            EmbedderConfig::Hash {
                dim: default_hash_dim(),
            }
        }
    }
}

impl EmbedderConfig {
    /// Parse a CLI-style spec: `hash`, `local`, `local:<model>`,
    /// `openai:<model>`, `ollama:<model>`.
    pub fn parse(spec: &str) -> Result<Self> {
        let (kind, rest) = match spec.split_once(':') {
            Some((kind, rest)) => (kind, Some(rest)),
            None => (spec, None),
        };
        match kind.to_ascii_lowercase().as_str() {
            "hash" => Ok(EmbedderConfig::Hash {
                dim: rest
                    .map(|d| {
                        d.parse().map_err(|_| {
                            EmbedError::Other(format!("`{d}` is not a valid dimension"))
                        })
                    })
                    .transpose()?
                    .unwrap_or_else(default_hash_dim),
            }),
            "local" => Ok(EmbedderConfig::Local {
                model: rest.map_or_else(default_local_model, str::to_string),
            }),
            "openai" => Ok(EmbedderConfig::OpenAi {
                model: rest.map_or_else(default_openai_model, str::to_string),
                api_key: None,
                base_url: None,
                dimensions: None,
            }),
            "ollama" => Ok(EmbedderConfig::Ollama {
                model: rest
                    .ok_or_else(|| {
                        EmbedError::Other(
                            "ollama needs a model, e.g. `ollama:nomic-embed-text`".into(),
                        )
                    })?
                    .to_string(),
                base_url: None,
            }),
            other => Err(EmbedError::Other(format!(
                "unknown embedder `{other}` (expected hash, local, openai, or ollama)"
            ))),
        }
    }

    /// The provider name, for error messages.
    pub fn provider_name(&self) -> &'static str {
        match self {
            EmbedderConfig::Hash { .. } => "hash",
            EmbedderConfig::Local { .. } => "local",
            EmbedderConfig::OpenAi { .. } => "openai",
            EmbedderConfig::Ollama { .. } => "ollama",
        }
    }
}

/// Construct an embedder from its configuration.
pub async fn build(config: &EmbedderConfig) -> Result<Box<dyn Embedder>> {
    match config {
        EmbedderConfig::Hash { dim } => Ok(Box::new(crate::hash::HashEmbedder::new(*dim))),

        EmbedderConfig::OpenAi {
            model,
            api_key,
            base_url,
            dimensions,
        } => Ok(Box::new(crate::openai::OpenAiEmbedder::new(
            model.clone(),
            api_key.clone(),
            base_url.clone(),
            *dimensions,
        )?)),

        EmbedderConfig::Ollama { model, base_url } => Ok(Box::new(
            crate::ollama::OllamaEmbedder::connect(model.clone(), base_url.clone()).await?,
        )),

        #[cfg(feature = "local")]
        EmbedderConfig::Local { model } => {
            Ok(Box::new(crate::local::LocalEmbedder::load(model).await?))
        }

        #[cfg(not(feature = "local"))]
        EmbedderConfig::Local { model } => Err(EmbedError::NotConfigured {
            provider: "local".into(),
            reason: format!(
                "this build has no bundled ONNX runtime, so `{model}` cannot run in-process — \
                 rebuild with `--features local`, or use `ollama:` for a local server"
            ),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_parse() {
        assert_eq!(
            EmbedderConfig::parse("hash").unwrap(),
            EmbedderConfig::Hash { dim: 256 }
        );
        assert_eq!(
            EmbedderConfig::parse("hash:64").unwrap(),
            EmbedderConfig::Hash { dim: 64 }
        );
        assert!(matches!(
            EmbedderConfig::parse("openai:text-embedding-3-large").unwrap(),
            EmbedderConfig::OpenAi { model, .. } if model == "text-embedding-3-large"
        ));
        assert!(matches!(
            EmbedderConfig::parse("ollama:nomic-embed-text").unwrap(),
            EmbedderConfig::Ollama { model, .. } if model == "nomic-embed-text"
        ));
    }

    #[test]
    fn bad_specs_are_rejected_with_guidance() {
        let err = EmbedderConfig::parse("nope").unwrap_err().to_string();
        assert!(err.contains("hash, local, openai, or ollama"), "got: {err}");
        // Ollama has no sensible default model, so omitting one must not
        // silently pick something.
        assert!(EmbedderConfig::parse("ollama").is_err());
    }

    #[test]
    fn config_roundtrips_through_json() {
        for config in [
            EmbedderConfig::Hash { dim: 128 },
            EmbedderConfig::Local { model: "m".into() },
            EmbedderConfig::OpenAi {
                model: "text-embedding-3-small".into(),
                api_key: None,
                base_url: Some("https://example.invalid".into()),
                dimensions: Some(512),
            },
            EmbedderConfig::Ollama {
                model: "nomic-embed-text".into(),
                base_url: None,
            },
        ] {
            let json = serde_json::to_string(&config).unwrap();
            assert_eq!(config, serde_json::from_str(&json).unwrap(), "{json}");
        }
    }
}
