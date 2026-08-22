//! An embedding model running in-process via ONNX Runtime.
//!
//! This is what makes a fresh install useful with no API key and no network
//! after the first run: the model is fetched once into a cache directory and
//! then executed locally. Weights are not vendored into the repository — a
//! 90 MB binary blob in git serves nobody — so the first call downloads them,
//! verifies the size, and caches them.

use crate::provider::{Embedder, EmbeddingBatch, Usage};
use crate::{EmbedError, Result};
use async_trait::async_trait;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use parking_lot::Mutex;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

/// How the token vectors are collapsed into one sentence vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pooling {
    /// Average over non-padding tokens. What sentence-transformers models
    /// expect.
    Mean,
    /// Take the `[CLS]` token.
    Cls,
}

/// A model this build knows how to fetch and run.
#[derive(Clone, Debug)]
pub struct ModelSpec {
    pub id: &'static str,
    pub repo: &'static str,
    pub onnx_path: &'static str,
    pub dim: usize,
    pub max_tokens: usize,
    pub pooling: Pooling,
    /// Whether the graph takes a `token_type_ids` input. BERT-family models
    /// do; several distilled ones do not, and passing it fails the run.
    pub token_type_ids: bool,
}

/// The model used when none is named.
pub const DEFAULT_MODEL: &str = "all-MiniLM-L6-v2";

/// Models with known-good ONNX exports.
pub const MODELS: &[ModelSpec] = &[
    ModelSpec {
        id: "all-MiniLM-L6-v2",
        repo: "sentence-transformers/all-MiniLM-L6-v2",
        onnx_path: "onnx/model.onnx",
        dim: 384,
        max_tokens: 256,
        pooling: Pooling::Mean,
        token_type_ids: true,
    },
    ModelSpec {
        id: "all-MiniLM-L12-v2",
        repo: "sentence-transformers/all-MiniLM-L12-v2",
        onnx_path: "onnx/model.onnx",
        dim: 384,
        max_tokens: 256,
        pooling: Pooling::Mean,
        token_type_ids: true,
    },
    ModelSpec {
        id: "bge-small-en-v1.5",
        repo: "BAAI/bge-small-en-v1.5",
        onnx_path: "onnx/model.onnx",
        dim: 384,
        max_tokens: 512,
        pooling: Pooling::Cls,
        token_type_ids: true,
    },
];

pub fn lookup(model: &str) -> Option<&'static ModelSpec> {
    MODELS.iter().find(|m| m.id == model)
}

/// A locally executed sentence embedder.
pub struct LocalEmbedder {
    spec: &'static ModelSpec,
    tokenizer: Tokenizer,
    /// ONNX Runtime sessions need `&mut` to run, so concurrent callers
    /// serialise here. One session is the right shape anyway: the runtime
    /// already parallelises within a batch.
    session: Mutex<Session>,
}

impl LocalEmbedder {
    /// Load a model, downloading it into the cache on first use.
    pub async fn load(model: &str) -> Result<Self> {
        let spec = lookup(model).ok_or_else(|| {
            EmbedError::UnknownModel(format!(
                "{model} (known: {})",
                MODELS.iter().map(|m| m.id).collect::<Vec<_>>().join(", ")
            ))
        })?;

        let dir = model_cache_dir()?.join(spec.id);
        let onnx = ensure_file(&dir, spec.repo, spec.onnx_path, "model.onnx").await?;
        let tokenizer_path =
            ensure_file(&dir, spec.repo, "tokenizer.json", "tokenizer.json").await?;

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            EmbedError::Other(format!(
                "could not read tokenizer at {}: {e}",
                tokenizer_path.display()
            ))
        })?;
        // Pad to the longest sequence in the batch and truncate at the model's
        // limit, so a long chunk degrades instead of failing the run.
        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        }));
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: spec.max_tokens,
                ..Default::default()
            }))
            .map_err(|e| EmbedError::Other(format!("could not configure truncation: {e}")))?;

        let session = Session::builder()
            .map_err(ort_error)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort_error)?
            .with_intra_threads(num_threads())
            .map_err(ort_error)?
            .commit_from_file(&onnx)
            .map_err(|e| {
                EmbedError::Other(format!(
                    "could not load ONNX model at {}: {e}",
                    onnx.display()
                ))
            })?;

        Ok(LocalEmbedder {
            spec,
            tokenizer,
            session: Mutex::new(session),
        })
    }

    /// Tokenize, run the graph, pool, and normalise.
    fn run(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| EmbedError::Other(format!("tokenization failed: {e}")))?;

        let batch = encodings.len();
        let seq_len = encodings.first().map_or(0, |e| e.get_ids().len());
        if batch == 0 || seq_len == 0 {
            return Ok(vec![vec![0.0; self.spec.dim]; batch]);
        }

        let mut ids = Vec::with_capacity(batch * seq_len);
        let mut mask = Vec::with_capacity(batch * seq_len);
        for encoding in &encodings {
            ids.extend(encoding.get_ids().iter().map(|v| *v as i64));
            mask.extend(encoding.get_attention_mask().iter().map(|v| *v as i64));
        }

        let shape = [batch as i64, seq_len as i64];
        let ids_tensor = Tensor::from_array((shape, ids)).map_err(ort_error)?;
        let mask_tensor = Tensor::from_array((shape, mask.clone())).map_err(ort_error)?;

        let mut session = self.session.lock();
        let outputs = if self.spec.token_type_ids {
            let types =
                Tensor::from_array((shape, vec![0i64; batch * seq_len])).map_err(ort_error)?;
            session.run(ort::inputs![
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
                "token_type_ids" => types,
            ])
        } else {
            session.run(ort::inputs![
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
            ])
        }
        .map_err(ort_error)?;

        // Export names vary between model repos, so try the conventional ones
        // and fall back to whatever the graph's first output happens to be.
        let output_name = ["last_hidden_state", "sentence_embedding"]
            .into_iter()
            .find(|name| outputs.get(name).is_some())
            .map(str::to_string)
            .or_else(|| outputs.iter().next().map(|(name, _)| name.to_string()))
            .ok_or_else(|| EmbedError::Other("model produced no outputs".into()))?;

        let (out_shape, data) = outputs[output_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(ort_error)?;

        // Some exports emit the pooled vector directly, others the full token
        // sequence; handle both rather than assuming.
        let hidden = *out_shape
            .last()
            .ok_or_else(|| EmbedError::Other("model output has no dimensions".into()))?
            as usize;
        let vectors = if out_shape.len() == 2 {
            data.chunks(hidden).map(<[f32]>::to_vec).collect()
        } else {
            self.pool(data, &mask, batch, seq_len, hidden)
        };

        Ok(vectors
            .into_iter()
            .map(|mut v| {
                v.truncate(self.spec.dim);
                normalize(&mut v);
                v
            })
            .collect())
    }

    /// Collapse `[batch, seq, hidden]` into `[batch, hidden]`.
    fn pool(
        &self,
        data: &[f32],
        mask: &[i64],
        batch: usize,
        seq_len: usize,
        hidden: usize,
    ) -> Vec<Vec<f32>> {
        (0..batch)
            .map(|b| match self.spec.pooling {
                Pooling::Cls => {
                    let start = b * seq_len * hidden;
                    data[start..start + hidden].to_vec()
                }
                Pooling::Mean => {
                    let mut sum = vec![0.0f32; hidden];
                    let mut count = 0.0f32;
                    for t in 0..seq_len {
                        // Padding tokens must not drag the mean toward zero.
                        if mask[b * seq_len + t] == 0 {
                            continue;
                        }
                        count += 1.0;
                        let start = (b * seq_len + t) * hidden;
                        for (i, value) in data[start..start + hidden].iter().enumerate() {
                            sum[i] += value;
                        }
                    }
                    if count > 0.0 {
                        for value in &mut sum {
                            *value /= count;
                        }
                    }
                    sum
                }
            })
            .collect()
    }
}

#[async_trait]
impl Embedder for LocalEmbedder {
    fn model_id(&self) -> &str {
        self.spec.id
    }

    fn dim(&self) -> usize {
        self.spec.dim
    }

    fn max_batch(&self) -> usize {
        // Padding is to the longest member, so oversized batches waste compute
        // on short texts sharing a batch with a long one.
        16
    }

    fn describe(&self) -> String {
        format!(
            "{} running locally via ONNX Runtime ({} dimensions, {} token limit)",
            self.spec.id, self.spec.dim, self.spec.max_tokens
        )
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<EmbeddingBatch> {
        let texts = texts.to_vec();
        let vectors = self.run(&texts)?;
        Ok(EmbeddingBatch {
            vectors,
            usage: Usage {
                prompt_tokens: 0,
                requests: 1,
            },
        })
    }
}

fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for value in vector.iter_mut() {
            *value /= norm;
        }
    }
}

fn num_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(8))
        .unwrap_or(4)
}

fn ort_error(e: impl std::fmt::Display) -> EmbedError {
    EmbedError::Other(format!("ONNX Runtime: {e}"))
}

/// Where downloaded models live.
///
/// Honours `NEXUM_MODEL_DIR`, then the platform cache directory, so a shared
/// machine can point every user at one copy.
pub fn model_cache_dir() -> Result<PathBuf> {
    if let Ok(dir) = std::env::var("NEXUM_MODEL_DIR") {
        return Ok(PathBuf::from(dir));
    }

    #[cfg(target_os = "windows")]
    let base = std::env::var("LOCALAPPDATA").map(PathBuf::from);
    #[cfg(target_os = "macos")]
    let base = std::env::var("HOME").map(|h| PathBuf::from(h).join("Library/Caches"));
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    let base = std::env::var("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|_| std::env::var("HOME").map(|h| PathBuf::from(h).join(".cache")));

    let base = base.map_err(|_| EmbedError::NotConfigured {
        provider: "local".into(),
        reason: "no cache directory — set NEXUM_MODEL_DIR to somewhere writable".into(),
    })?;
    Ok(base.join("nexumdb").join("models"))
}

/// Fetch a file into the cache if it is not already there.
///
/// Downloads land on a temporary path and are renamed into place, so an
/// interrupted download can never be mistaken for a complete one.
async fn ensure_file(
    dir: &Path,
    repo: &str,
    remote_path: &str,
    local_name: &str,
) -> Result<PathBuf> {
    let target = dir.join(local_name);
    if target.exists() {
        return Ok(target);
    }
    std::fs::create_dir_all(dir)?;

    let url = format!("https://huggingface.co/{repo}/resolve/main/{remote_path}");
    tracing::info!(%url, "downloading model file");

    let response = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(1800))
        .build()
        .map_err(|e| EmbedError::Transport {
            provider: "local".into(),
            source: Box::new(e),
        })?
        .get(&url)
        .send()
        .await
        .map_err(|e| EmbedError::Transport {
            provider: "local".into(),
            source: Box::new(e),
        })?;

    if !response.status().is_success() {
        return Err(EmbedError::Api {
            provider: "local".into(),
            message: format!("could not download {url}: HTTP {}", response.status()),
        });
    }

    let bytes = response.bytes().await.map_err(|e| EmbedError::Transport {
        provider: "local".into(),
        source: Box::new(e),
    })?;
    if bytes.is_empty() {
        return Err(EmbedError::Api {
            provider: "local".into(),
            message: format!("{url} returned an empty file"),
        });
    }

    let temp = dir.join(format!("{local_name}.partial"));
    std::fs::write(&temp, &bytes)?;
    std::fs::rename(&temp, &target)?;
    tracing::info!(path = %target.display(), bytes = bytes.len(), "cached model file");
    Ok(target)
}

#[cfg(test)]
impl LocalEmbedder {
    /// Pooling without a loaded session, so the maths can be tested without
    /// downloading 90 MB of weights.
    fn pool_standalone(
        pooling: Pooling,
        data: &[f32],
        mask: &[i64],
        batch: usize,
        seq_len: usize,
        hidden: usize,
    ) -> Vec<Vec<f32>> {
        (0..batch)
            .map(|b| match pooling {
                Pooling::Cls => data[b * seq_len * hidden..b * seq_len * hidden + hidden].to_vec(),
                Pooling::Mean => {
                    let mut sum = vec![0.0f32; hidden];
                    let mut count = 0.0f32;
                    for t in 0..seq_len {
                        if mask[b * seq_len + t] == 0 {
                            continue;
                        }
                        count += 1.0;
                        let start = (b * seq_len + t) * hidden;
                        for (i, value) in data[start..start + hidden].iter().enumerate() {
                            sum[i] += value;
                        }
                    }
                    if count > 0.0 {
                        for value in &mut sum {
                            *value /= count;
                        }
                    }
                    sum
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_model_is_in_the_registry() {
        assert!(lookup(DEFAULT_MODEL).is_some());
    }

    #[test]
    fn unknown_models_are_rejected() {
        assert!(lookup("not-a-real-model").is_none());
    }

    #[test]
    fn model_ids_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for spec in MODELS {
            assert!(seen.insert(spec.id), "duplicate model id {}", spec.id);
            assert!(spec.dim > 0);
            assert!(spec.max_tokens > 0);
        }
    }

    #[test]
    fn the_cache_directory_is_configurable() {
        // SAFETY: single-threaded test that restores the variable.
        let previous = std::env::var("NEXUM_MODEL_DIR").ok();
        unsafe { std::env::set_var("NEXUM_MODEL_DIR", "/tmp/nexum-models-test") };
        assert_eq!(
            model_cache_dir().unwrap(),
            PathBuf::from("/tmp/nexum-models-test")
        );
        unsafe {
            match previous {
                Some(p) => std::env::set_var("NEXUM_MODEL_DIR", p),
                None => std::env::remove_var("NEXUM_MODEL_DIR"),
            }
        }
    }

    /// Mean pooling must ignore padding; averaging it in pulls every short
    /// text toward the same vector.
    #[test]
    fn mean_pooling_ignores_padding() {
        let embedder_spec = &MODELS[0];
        let hidden = 2;
        let seq_len = 3;
        // One row: two real tokens [1,1] and [3,3], then padding [99,99].
        let data = vec![1.0, 1.0, 3.0, 3.0, 99.0, 99.0];
        let mask = vec![1i64, 1, 0];

        let pooled =
            LocalEmbedder::pool_standalone(embedder_spec.pooling, &data, &mask, 1, seq_len, hidden);
        assert_eq!(pooled[0], vec![2.0, 2.0], "padding must not be averaged in");
    }

    #[test]
    fn normalization_produces_unit_vectors() {
        let mut v = vec![3.0f32, 4.0];
        normalize(&mut v);
        assert!((v.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn normalizing_a_zero_vector_is_safe() {
        let mut v = vec![0.0f32; 4];
        normalize(&mut v);
        assert!(v.iter().all(|x| x.is_finite()));
    }
}
