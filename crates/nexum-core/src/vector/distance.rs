//! Distance metrics for vector search.
//!
//! Everything downstream treats these as *distances* — smaller is closer — so
//! cosine is expressed as `1 - similarity`. The loops are written in a shape
//! LLVM reliably auto-vectorises; benchmark before reaching for explicit SIMD.

use serde::{Deserialize, Serialize};
use std::str::FromStr;

use crate::error::{Error, Result};

/// How closeness between two vectors is measured.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    /// Angular distance, `1 - cos(a, b)`. The right default for text
    /// embeddings, whose magnitude carries no meaning.
    #[default]
    Cosine,
    /// Straight-line distance.
    Euclidean,
    /// Negated inner product, for models trained with a dot-product objective.
    DotProduct,
}

impl Metric {
    pub const fn as_str(self) -> &'static str {
        match self {
            Metric::Cosine => "cosine",
            Metric::Euclidean => "euclidean",
            Metric::DotProduct => "dot_product",
        }
    }

    /// Whether vectors should be unit-normalised before indexing.
    ///
    /// For cosine this turns every later distance computation into a plain dot
    /// product, which is the single biggest win available here.
    pub const fn normalizes(self) -> bool {
        matches!(self, Metric::Cosine)
    }

    /// Distance between two vectors of equal length.
    ///
    /// For [`Metric::Cosine`] the inputs are assumed already normalised — see
    /// [`Metric::prepare`].
    pub fn distance(self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "distance over mismatched dimensions");
        match self {
            Metric::Cosine => 1.0 - dot(a, b),
            Metric::DotProduct => -dot(a, b),
            Metric::Euclidean => squared_l2(a, b).sqrt(),
        }
    }

    /// Convert a raw vector into the form the index stores.
    pub fn prepare(self, vector: &[f32]) -> Vec<f32> {
        if self.normalizes() {
            normalize(vector)
        } else {
            vector.to_vec()
        }
    }

    /// Turn a distance back into a 0..=1 similarity score for display.
    ///
    /// Ranking is unaffected — this exists so the CLI and viewer can show a
    /// number that reads the way users expect, with 1.0 meaning identical.
    pub fn similarity(self, distance: f32) -> f32 {
        match self {
            Metric::Cosine => (1.0 - distance).clamp(-1.0, 1.0),
            Metric::DotProduct => -distance,
            Metric::Euclidean => 1.0 / (1.0 + distance.max(0.0)),
        }
    }
}

impl FromStr for Metric {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().replace(['-', '_'], "").as_str() {
            "cosine" | "cos" => Ok(Metric::Cosine),
            "euclidean" | "l2" => Ok(Metric::Euclidean),
            "dotproduct" | "dot" | "ip" => Ok(Metric::DotProduct),
            _ => Err(Error::InvalidArgument(format!("unknown metric `{s}`"))),
        }
    }
}

impl std::fmt::Display for Metric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Inner product.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    // Four independent accumulators break the serial dependency chain and let
    // the vectoriser fold the tail cleanly.
    let mut acc = [0.0f32; 4];
    let chunks = a.len() / 4;
    for i in 0..chunks {
        let j = i * 4;
        acc[0] += a[j] * b[j];
        acc[1] += a[j + 1] * b[j + 1];
        acc[2] += a[j + 2] * b[j + 2];
        acc[3] += a[j + 3] * b[j + 3];
    }
    let mut total = acc[0] + acc[1] + acc[2] + acc[3];
    for i in (chunks * 4)..a.len() {
        total += a[i] * b[i];
    }
    total
}

/// Squared Euclidean distance.
pub fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 4];
    let chunks = a.len() / 4;
    for i in 0..chunks {
        let j = i * 4;
        let d0 = a[j] - b[j];
        let d1 = a[j + 1] - b[j + 1];
        let d2 = a[j + 2] - b[j + 2];
        let d3 = a[j + 3] - b[j + 3];
        acc[0] += d0 * d0;
        acc[1] += d1 * d1;
        acc[2] += d2 * d2;
        acc[3] += d3 * d3;
    }
    let mut total = acc[0] + acc[1] + acc[2] + acc[3];
    for i in (chunks * 4)..a.len() {
        let d = a[i] - b[i];
        total += d * d;
    }
    total
}

/// Scale a vector to unit length. A zero vector is returned unchanged rather
/// than producing NaNs — an all-zero embedding is a degenerate but legal input.
pub fn normalize(vector: &[f32]) -> Vec<f32> {
    let norm = dot(vector, vector).sqrt();
    if norm <= f32::EPSILON {
        return vector.to_vec();
    }
    vector.iter().map(|v| v / norm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_are_at_zero_cosine_distance() {
        let v = Metric::Cosine.prepare(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(Metric::Cosine.distance(&v, &v).abs() < 1e-5);
    }

    #[test]
    fn orthogonal_vectors_are_at_cosine_distance_one() {
        let a = Metric::Cosine.prepare(&[1.0, 0.0]);
        let b = Metric::Cosine.prepare(&[0.0, 1.0]);
        assert!((Metric::Cosine.distance(&a, &b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_ignores_magnitude() {
        let a = Metric::Cosine.prepare(&[1.0, 2.0, 3.0]);
        let b = Metric::Cosine.prepare(&[10.0, 20.0, 30.0]);
        assert!(Metric::Cosine.distance(&a, &b).abs() < 1e-5);
    }

    #[test]
    fn euclidean_matches_the_textbook_value() {
        let d = Metric::Euclidean.distance(&[0.0, 0.0], &[3.0, 4.0]);
        assert!((d - 5.0).abs() < 1e-5);
    }

    #[test]
    fn dot_and_l2_handle_lengths_that_are_not_multiples_of_four() {
        for len in 1..=17usize {
            let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..len).map(|i| (len - i) as f32).collect();
            let expected_dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            assert!((dot(&a, &b) - expected_dot).abs() < 1e-3, "len {len}");
            let expected_l2: f32 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
            assert!((squared_l2(&a, &b) - expected_l2).abs() < 1e-3, "len {len}");
        }
    }

    #[test]
    fn normalizing_a_zero_vector_does_not_produce_nans() {
        let n = normalize(&[0.0, 0.0, 0.0]);
        assert!(n.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn similarity_is_one_for_an_exact_match() {
        for metric in [Metric::Cosine, Metric::Euclidean] {
            assert!((metric.similarity(0.0) - 1.0).abs() < 1e-5, "{metric}");
        }
    }

    #[test]
    fn metrics_roundtrip_through_strings() {
        for m in [Metric::Cosine, Metric::Euclidean, Metric::DotProduct] {
            assert_eq!(m, m.as_str().parse::<Metric>().unwrap());
        }
    }
}
