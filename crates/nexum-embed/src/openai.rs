//! OpenAI-compatible embeddings.
//!
//! Also covers Azure OpenAI, together.ai, vLLM, LM Studio and anything else
//! exposing `POST /v1/embeddings` — point `base_url` at it.

use crate::provider::{Embedder, EmbeddingBatch, Usage};
use crate::{EmbedError, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Known output dimensions, so a database's namespace is decided before the
/// first request rather than after it.
fn known_dimension(model: &str) -> Option<usize> {
    match model {
        "text-embedding-3-small" => Some(1536),
        "text-embedding-3-large" => Some(3072),
        "text-embedding-ada-002" => Some(1536),
        _ => None,
    }
}

pub struct OpenAiEmbedder {
    client: reqwest::Client,
    model: String,
    api_key: String,
    base_url: String,
    dim: usize,
    /// Sent as `dimensions` when the caller asked for a reduced size.
    requested_dimensions: Option<usize>,
}

impl OpenAiEmbedder {
    /// Build an embedder. The API key falls back to `OPENAI_API_KEY`.
    pub fn new(
        model: String,
        api_key: Option<String>,
        base_url: Option<String>,
        dimensions: Option<usize>,
    ) -> Result<Self> {
        let api_key = api_key
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .filter(|k| !k.trim().is_empty())
            .ok_or_else(|| EmbedError::NotConfigured {
                provider: "openai".into(),
                reason: "no API key — set OPENAI_API_KEY or pass one in the config".into(),
            })?;

        let dim = dimensions
            .or_else(|| known_dimension(&model))
            .ok_or_else(|| EmbedError::NotConfigured {
                provider: "openai".into(),
                reason: format!(
                    "the output dimension of `{model}` is not known ahead of time; \
                     set `dimensions` explicitly so the vector index can be sized"
                ),
            })?;

        let base_url = base_url
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string())
            .trim_end_matches('/')
            .to_string();

        Ok(OpenAiEmbedder {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .map_err(|e| EmbedError::Transport {
                    provider: "openai".into(),
                    source: Box::new(e),
                })?,
            model,
            api_key,
            base_url,
            dim,
            requested_dimensions: dimensions,
        })
    }
}

/// Redacts the API key — a `Debug` print of this struct lands in logs and
/// error reports, and a leaked key there is a real incident.
impl std::fmt::Debug for OpenAiEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiEmbedder")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("dim", &self.dim)
            .field("api_key", &"<redacted>")
            .finish()
    }
}

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
    #[serde(default)]
    usage: Option<ApiUsage>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    /// Position in the request. The API documents results as ordered, but
    /// sorting by this makes the guarantee ours rather than theirs.
    #[serde(default)]
    index: usize,
}

#[derive(Deserialize)]
struct ApiUsage {
    #[serde(default)]
    prompt_tokens: u64,
}

#[async_trait]
impl Embedder for OpenAiEmbedder {
    fn model_id(&self) -> &str {
        &self.model
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn max_batch(&self) -> usize {
        // The API caps a request at 2048 inputs; well below that keeps
        // individual failures cheap to retry.
        128
    }

    fn describe(&self) -> String {
        format!("{} via OpenAI API ({} dimensions)", self.model, self.dim)
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<EmbeddingBatch> {
        let response = self
            .client
            .post(format!("{}/embeddings", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&EmbeddingRequest {
                model: &self.model,
                input: texts,
                dimensions: self.requested_dimensions,
            })
            .send()
            .await
            .map_err(|e| EmbedError::Transport {
                provider: "openai".into(),
                source: Box::new(e),
            })?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(EmbedError::Api {
                provider: "openai".into(),
                message: format!("HTTP {status}: {}", extract_api_message(&body)),
            });
        }

        let mut parsed: EmbeddingResponse =
            response.json().await.map_err(|e| EmbedError::Transport {
                provider: "openai".into(),
                source: Box::new(e),
            })?;

        parsed.data.sort_by_key(|d| d.index);
        Ok(EmbeddingBatch {
            vectors: parsed.data.into_iter().map(|d| d.embedding).collect(),
            usage: Usage {
                prompt_tokens: parsed.usage.map_or(0, |u| u.prompt_tokens),
                requests: 1,
            },
        })
    }
}

/// Pull the human-readable message out of an error body, falling back to the
/// raw text when it is not the shape we expect.
fn extract_api_message(body: &str) -> String {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|v| {
            v.get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| body.chars().take(300).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_api_key_is_reported_clearly() {
        // SAFETY: single-threaded test, and the variable is restored below.
        let previous = std::env::var("OPENAI_API_KEY").ok();
        unsafe { std::env::remove_var("OPENAI_API_KEY") };

        let err = OpenAiEmbedder::new("text-embedding-3-small".into(), None, None, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("OPENAI_API_KEY"), "got: {err}");

        if let Some(previous) = previous {
            unsafe { std::env::set_var("OPENAI_API_KEY", previous) };
        }
    }

    #[test]
    fn known_models_get_their_dimension_without_a_request() {
        let e = OpenAiEmbedder::new(
            "text-embedding-3-small".into(),
            Some("k".into()),
            None,
            None,
        )
        .unwrap();
        assert_eq!(e.dim(), 1536);
        let e = OpenAiEmbedder::new(
            "text-embedding-3-large".into(),
            Some("k".into()),
            None,
            None,
        )
        .unwrap();
        assert_eq!(e.dim(), 3072);
    }

    #[test]
    fn an_unknown_model_must_declare_its_dimension() {
        let err = OpenAiEmbedder::new("some-new-model".into(), Some("k".into()), None, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("dimensions"), "got: {err}");

        let ok = OpenAiEmbedder::new("some-new-model".into(), Some("k".into()), None, Some(768))
            .unwrap();
        assert_eq!(ok.dim(), 768);
    }

    #[test]
    fn a_reduced_dimension_overrides_the_model_default() {
        let e = OpenAiEmbedder::new(
            "text-embedding-3-large".into(),
            Some("k".into()),
            None,
            Some(256),
        )
        .unwrap();
        assert_eq!(e.dim(), 256);
    }

    #[test]
    fn base_url_trailing_slashes_are_normalised() {
        let e = OpenAiEmbedder::new(
            "text-embedding-3-small".into(),
            Some("k".into()),
            Some("https://example.invalid/v1/".into()),
            None,
        )
        .unwrap();
        assert_eq!(e.base_url, "https://example.invalid/v1");
    }

    #[test]
    fn api_error_messages_are_unwrapped() {
        let body =
            r#"{"error":{"message":"Incorrect API key provided","type":"invalid_request_error"}}"#;
        assert_eq!(extract_api_message(body), "Incorrect API key provided");
        assert_eq!(
            extract_api_message("plain text failure"),
            "plain text failure"
        );
    }

    #[test]
    fn out_of_order_results_are_restored_to_input_order() {
        let json = r#"{"data":[
            {"embedding":[2.0],"index":1},
            {"embedding":[0.0],"index":0}
        ],"usage":{"prompt_tokens":7}}"#;
        let mut parsed: EmbeddingResponse = serde_json::from_str(json).unwrap();
        parsed.data.sort_by_key(|d| d.index);
        let vectors: Vec<Vec<f32>> = parsed.data.into_iter().map(|d| d.embedding).collect();
        assert_eq!(vectors, vec![vec![0.0], vec![2.0]]);
    }
}
