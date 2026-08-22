//! A deterministic embedder with no model and no network.
//!
//! This is a hashed bag-of-words projection, not a stand-in for a real model:
//! it captures lexical overlap and nothing semantic. It exists so that a fresh
//! install can ingest and search immediately, so tests can assert on exact
//! vectors, and so a pipeline can be debugged without burning API calls. Its
//! model ID carries the dimension, so vectors from it can never be silently
//! mixed with a real model's.

use crate::Result;
use crate::provider::{Embedder, EmbeddingBatch, Usage};
use async_trait::async_trait;

/// Deterministic hashed bag-of-words embedder.
pub struct HashEmbedder {
    dim: usize,
    model_id: String,
}

impl HashEmbedder {
    pub fn new(dim: usize) -> Self {
        let dim = dim.max(2);
        HashEmbedder {
            dim,
            model_id: format!("hash-bow-v1-{dim}"),
        }
    }

    /// Project one text into the vector space.
    ///
    /// Each token contributes to two dimensions with signs drawn from its
    /// hash — the signed variant of the hashing trick, which keeps collisions
    /// from systematically inflating a dimension. Bigrams carry a little word
    /// order, which pure unigrams throw away entirely.
    pub fn embed_text(&self, text: &str) -> Vec<f32> {
        let mut vector = vec![0.0f32; self.dim];
        let tokens: Vec<String> = tokenize(text);

        for token in &tokens {
            self.accumulate(&mut vector, token, 1.0);
        }
        for pair in tokens.windows(2) {
            self.accumulate(&mut vector, &format!("{} {}", pair[0], pair[1]), 0.5);
        }

        normalize(&mut vector);
        vector
    }

    fn accumulate(&self, vector: &mut [f32], token: &str, weight: f32) {
        let hash = fnv1a(token.as_bytes());
        let index = (hash % self.dim as u64) as usize;
        let sign = if (hash >> 32) & 1 == 0 { 1.0 } else { -1.0 };
        vector[index] += weight * sign;

        // A second, independent slot reduces the damage from any one
        // collision without needing a wider vector.
        let secondary = fnv1a(&hash.to_le_bytes());
        let index2 = (secondary % self.dim as u64) as usize;
        let sign2 = if (secondary >> 32) & 1 == 0 {
            1.0
        } else {
            -1.0
        };
        vector[index2] += weight * sign2 * 0.5;
    }
}

#[async_trait]
impl Embedder for HashEmbedder {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn max_batch(&self) -> usize {
        // Nothing to batch against; the cap only bounds memory.
        1024
    }

    fn describe(&self) -> String {
        format!(
            "{} — deterministic hashed bag-of-words, lexical similarity only",
            self.model_id
        )
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<EmbeddingBatch> {
        let vectors = texts.iter().map(|t| self.embed_text(t)).collect();
        Ok(EmbeddingBatch {
            vectors,
            usage: Usage {
                prompt_tokens: texts.iter().map(|t| tokenize(t).len() as u64).sum(),
                requests: 1,
            },
        })
    }
}

/// Lowercase, split on anything that is not alphanumeric.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for v in vector.iter_mut() {
            *v /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[tokio::test]
    async fn embeddings_are_deterministic() {
        let e = HashEmbedder::new(128);
        let a = e.embed_text("the quick brown fox");
        let b = e.embed_text("the quick brown fox");
        assert_eq!(a, b);
    }

    #[tokio::test]
    async fn vectors_are_unit_length() {
        let e = HashEmbedder::new(128);
        let v = e.embed_text("some text here");
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-4);
    }

    #[tokio::test]
    async fn overlapping_text_scores_higher_than_unrelated_text() {
        let e = HashEmbedder::new(256);
        let query = e.embed_text("graph database vector search");
        let related = e.embed_text("vector search in a graph database");
        let unrelated = e.embed_text("banana bread recipe with walnuts");
        assert!(
            cosine(&query, &related) > cosine(&query, &unrelated),
            "related {:.3} should beat unrelated {:.3}",
            cosine(&query, &related),
            cosine(&query, &unrelated)
        );
    }

    #[tokio::test]
    async fn tokenization_ignores_case_and_punctuation() {
        let e = HashEmbedder::new(128);
        assert_eq!(e.embed_text("Hello, World!"), e.embed_text("hello world"));
    }

    #[tokio::test]
    async fn word_order_is_not_entirely_discarded() {
        let e = HashEmbedder::new(512);
        let a = e.embed_text("dog bites man");
        let b = e.embed_text("man bites dog");
        assert_ne!(a, b, "bigrams should distinguish these");
    }

    #[tokio::test]
    async fn empty_text_yields_a_finite_zero_vector() {
        let e = HashEmbedder::new(64);
        let v = e.embed_text("");
        assert_eq!(v.len(), 64);
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[tokio::test]
    async fn batching_matches_one_at_a_time() {
        let e = HashEmbedder::new(64);
        let texts: Vec<String> = (0..10).map(|i| format!("document number {i}")).collect();
        let batch = e.embed(&texts).await.unwrap();
        assert_eq!(batch.len(), 10);
        for (i, text) in texts.iter().enumerate() {
            assert_eq!(batch.vectors[i], e.embed_text(text));
        }
        assert_eq!(batch.usage.requests, 1);
    }

    #[tokio::test]
    async fn model_id_encodes_the_dimension() {
        assert_eq!(HashEmbedder::new(128).model_id(), "hash-bow-v1-128");
        assert_ne!(
            HashEmbedder::new(128).model_id(),
            HashEmbedder::new(256).model_id()
        );
    }

    #[tokio::test]
    async fn embedding_nothing_returns_nothing() {
        let e = HashEmbedder::new(64);
        assert!(e.embed(&[]).await.unwrap().is_empty());
    }
}
