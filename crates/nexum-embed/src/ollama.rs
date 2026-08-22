//! Embeddings from a local Ollama server.
//!
//! Ollama is the pragmatic way to run a real embedding model without linking a
//! native runtime into this binary: the model lives in a separate process the
//! user already manages.

use crate::provider::{Embedder, EmbeddingBatch, Usage};
use crate::{EmbedError, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

const DEFAULT_BASE_URL: &str = "http://localhost:11434";

pub struct OllamaEmbedder {
    client: reqwest::Client,
    model: String,
    base_url: String,
    /// Discovered by embedding a probe string at connect time — Ollama does
    /// not advertise dimensionality, and the index has to be sized up front.
    dim: usize,
}

impl OllamaEmbedder {
    /// Connect and determine the model's output dimension.
    pub async fn connect(model: String, base_url: Option<String>) -> Result<Self> {
        let base_url = base_url
            .or_else(|| std::env::var("OLLAMA_HOST").ok())
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string())
            .trim_end_matches('/')
            .to_string();

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .map_err(|e| EmbedError::Transport {
                provider: "ollama".into(),
                source: Box::new(e),
            })?;

        let mut embedder = OllamaEmbedder {
            client,
            model,
            base_url,
            dim: 0,
        };

        let probe = embedder
            .request(&["dimension probe".to_string()])
            .await
            .map_err(|e| match e {
                EmbedError::Transport { source, .. } => EmbedError::NotConfigured {
                    provider: "ollama".into(),
                    reason: format!(
                        "could not reach Ollama at {} ({source}) — is it running?",
                        embedder.base_url
                    ),
                },
                other => other,
            })?;

        embedder.dim = probe
            .first()
            .map(Vec::len)
            .filter(|d| *d > 0)
            .ok_or_else(|| EmbedError::Api {
                provider: "ollama".into(),
                message: format!(
                    "`{}` returned no vector — is it an embedding model?",
                    embedder.model
                ),
            })?;

        Ok(embedder)
    }

    async fn request(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let response = self
            .client
            .post(format!("{}/api/embed", self.base_url))
            .json(&EmbedRequest {
                model: &self.model,
                input: texts,
            })
            .send()
            .await
            .map_err(|e| EmbedError::Transport {
                provider: "ollama".into(),
                source: Box::new(e),
            })?;

        let status = response.status();
        let body = response.text().await.map_err(|e| EmbedError::Transport {
            provider: "ollama".into(),
            source: Box::new(e),
        })?;

        if !status.is_success() {
            let message = serde_json::from_str::<serde_json::Value>(&body)
                .ok()
                .and_then(|v| v.get("error").and_then(|e| e.as_str()).map(str::to_string))
                .unwrap_or_else(|| body.chars().take(300).collect());
            // Ollama returns 404 for a model it has not pulled, which is the
            // single most common failure here and deserves the real fix.
            if status.as_u16() == 404 {
                return Err(EmbedError::UnknownModel(format!(
                    "{} — run `ollama pull {}`",
                    self.model, self.model
                )));
            }
            return Err(EmbedError::Api {
                provider: "ollama".into(),
                message: format!("HTTP {status}: {message}"),
            });
        }

        let parsed: EmbedResponse = serde_json::from_str(&body).map_err(|e| EmbedError::Api {
            provider: "ollama".into(),
            message: format!("could not parse response: {e}"),
        })?;
        Ok(parsed.embeddings)
    }
}

impl std::fmt::Debug for OllamaEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OllamaEmbedder")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("dim", &self.dim)
            .finish()
    }
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct EmbedResponse {
    #[serde(default)]
    embeddings: Vec<Vec<f32>>,
}

#[async_trait]
impl Embedder for OllamaEmbedder {
    fn model_id(&self) -> &str {
        &self.model
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn max_batch(&self) -> usize {
        // Ollama runs these serially on one model instance; large batches just
        // lengthen the window in which a timeout loses everything.
        32
    }

    fn describe(&self) -> String {
        format!(
            "{} via Ollama at {} ({} dimensions)",
            self.model, self.base_url, self.dim
        )
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<EmbeddingBatch> {
        let vectors = self.request(texts).await?;
        Ok(EmbeddingBatch {
            vectors,
            usage: Usage {
                prompt_tokens: 0,
                requests: 1,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn an_unreachable_server_says_so() {
        let err = OllamaEmbedder::connect(
            "nomic-embed-text".into(),
            // Port 1 on loopback refuses immediately, so this asserts the
            // error message without spending the request timeout waiting.
            Some("http://127.0.0.1:1".into()),
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(err.contains("Ollama"), "got: {err}");
    }

    #[test]
    fn responses_parse() {
        let parsed: EmbedResponse =
            serde_json::from_str(r#"{"embeddings":[[1.0,2.0],[3.0,4.0]]}"#).unwrap();
        assert_eq!(parsed.embeddings.len(), 2);
        assert_eq!(parsed.embeddings[0], vec![1.0, 2.0]);
    }

    #[test]
    fn a_response_without_embeddings_parses_as_empty() {
        let parsed: EmbedResponse = serde_json::from_str(r#"{"model":"x"}"#).unwrap();
        assert!(parsed.embeddings.is_empty());
    }
}
