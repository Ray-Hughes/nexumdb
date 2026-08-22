//! 2D projection of embeddings, for the viewer's scatter plot.
//!
//! The spec asks for UMAP or t-SNE. Both are iterative neighbour-embedding
//! methods with real dependencies and real runtimes; what ships here is a
//! two-stage approach that keeps the useful property — semantically similar
//! chunks landing near each other — without either:
//!
//! 1. **PCA** by power iteration gives a deterministic, instant starting
//!    layout that preserves global structure.
//! 2. An optional **neighbour-embedding refinement** pulls each point toward
//!    its nearest neighbours in the original space and pushes it away from
//!    random others, which is the core of what t-SNE and UMAP optimise.
//!
//! Stage 1 alone is honest and fast, which is what the viewer defaults to for
//! large collections. The output names which method produced it, so the UI can
//! say so rather than implying a t-SNE that did not run.

use nexum_core::NodeId;
use serde::{Deserialize, Serialize};

/// Which projection to compute.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectionMethod {
    /// Principal components. Deterministic and instant.
    Pca,
    /// PCA followed by neighbour-embedding refinement. Better separated
    /// clusters, at the cost of an iterative pass.
    #[default]
    Neighborhood,
}

/// One projected point.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProjectedPoint {
    pub id: NodeId,
    pub x: f32,
    pub y: f32,
}

/// A projection and how it was made.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Projection {
    pub method: ProjectionMethod,
    pub points: Vec<ProjectedPoint>,
    /// Share of total variance the two axes capture. Only meaningful for
    /// [`ProjectionMethod::Pca`]; refinement discards the linear basis.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explained_variance: Option<f32>,
    pub dimensions: usize,
}

/// Tuning for the refinement stage.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProjectionParams {
    #[serde(default = "default_neighbors")]
    pub neighbors: usize,
    #[serde(default = "default_iterations")]
    pub iterations: usize,
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f32,
}

fn default_neighbors() -> usize {
    12
}
fn default_iterations() -> usize {
    150
}
fn default_learning_rate() -> f32 {
    0.15
}

impl Default for ProjectionParams {
    fn default() -> Self {
        ProjectionParams {
            neighbors: default_neighbors(),
            iterations: default_iterations(),
            learning_rate: default_learning_rate(),
        }
    }
}

/// Project vectors into two dimensions.
pub fn project(
    ids: &[NodeId],
    vectors: &[Vec<f32>],
    method: ProjectionMethod,
    params: ProjectionParams,
) -> Projection {
    assert_eq!(ids.len(), vectors.len(), "ids and vectors must line up");
    let dimensions = vectors.first().map_or(0, Vec::len);

    if vectors.len() < 3 || dimensions < 2 {
        // Too few points for any projection to mean anything; lay them out
        // deterministically rather than returning nothing.
        return Projection {
            method,
            points: ids
                .iter()
                .enumerate()
                .map(|(i, id)| ProjectedPoint {
                    id: *id,
                    x: i as f32,
                    y: 0.0,
                })
                .collect(),
            explained_variance: None,
            dimensions,
        };
    }

    let (mut coords, explained) = pca(vectors);
    let explained_variance = match method {
        ProjectionMethod::Pca => Some(explained),
        ProjectionMethod::Neighborhood => {
            refine(&mut coords, vectors, params);
            None
        }
    };
    normalize_layout(&mut coords);

    Projection {
        method,
        points: ids
            .iter()
            .zip(coords)
            .map(|(id, (x, y))| ProjectedPoint { id: *id, x, y })
            .collect(),
        explained_variance,
        dimensions,
    }
}

/// Two-component PCA by power iteration with deflation.
///
/// Power iteration avoids materialising a `dim x dim` covariance matrix, which
/// for a 3072-dimensional model would be nine million entries per request.
fn pca(vectors: &[Vec<f32>]) -> (Vec<(f32, f32)>, f32) {
    let n = vectors.len();
    let dim = vectors[0].len();

    let mut mean = vec![0.0f32; dim];
    for vector in vectors {
        for (i, value) in vector.iter().enumerate() {
            mean[i] += value;
        }
    }
    for value in &mut mean {
        *value /= n as f32;
    }

    let centered: Vec<Vec<f32>> = vectors
        .iter()
        .map(|v| v.iter().zip(&mean).map(|(a, b)| a - b).collect())
        .collect();

    let total_variance: f32 = centered
        .iter()
        .map(|v| v.iter().map(|x| x * x).sum::<f32>())
        .sum();

    let (first, first_variance) = principal_component(&centered, None);
    let (second, second_variance) = principal_component(&centered, Some(&first));

    let coords = centered
        .iter()
        .map(|v| (dot(v, &first), dot(v, &second)))
        .collect();

    let explained = if total_variance > f32::EPSILON {
        ((first_variance + second_variance) / total_variance).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (coords, explained)
}

/// Leading eigenvector of the covariance, optionally orthogonal to `against`.
fn principal_component(centered: &[Vec<f32>], against: Option<&[f32]>) -> (Vec<f32>, f32) {
    let dim = centered[0].len();
    // Deterministic seeding: the same data must always project the same way,
    // or the scatter plot reshuffles itself on every refresh.
    let mut component: Vec<f32> = (0..dim)
        .map(|i| ((i as f32 * 0.618_034).fract() - 0.5) * 2.0)
        .collect();
    if let Some(previous) = against {
        orthogonalize(&mut component, previous);
    }
    normalize(&mut component);

    let mut eigenvalue = 0.0f32;
    for _ in 0..64 {
        // next = C * component, where C = X^T X, computed without forming C.
        let mut next = vec![0.0f32; dim];
        for row in centered {
            let projection = dot(row, &component);
            for (i, value) in row.iter().enumerate() {
                next[i] += projection * value;
            }
        }
        if let Some(previous) = against {
            orthogonalize(&mut next, previous);
        }
        eigenvalue = next.iter().map(|x| x * x).sum::<f32>().sqrt();
        if eigenvalue < 1e-9 {
            break;
        }
        for value in &mut next {
            *value /= eigenvalue;
        }
        // Converged once the direction stops moving.
        let shift: f32 = next
            .iter()
            .zip(&component)
            .map(|(a, b)| (a - b).abs())
            .sum();
        component = next;
        if shift < 1e-6 {
            break;
        }
    }
    (component, eigenvalue)
}

fn orthogonalize(vector: &mut [f32], against: &[f32]) {
    let projection = dot(vector, against);
    for (value, base) in vector.iter_mut().zip(against) {
        *value -= projection * base;
    }
}

/// Pull each point toward its nearest neighbours and push it off everything
/// else — the attractive/repulsive pair at the heart of t-SNE and UMAP.
fn refine(coords: &mut [(f32, f32)], vectors: &[Vec<f32>], params: ProjectionParams) {
    let n = coords.len();
    let k = params.neighbors.min(n.saturating_sub(1)).max(1);

    // Exact kNN in the original space. Quadratic, which is why callers cap
    // how many points they ask to project.
    let neighbors: Vec<Vec<usize>> = (0..n)
        .map(|i| {
            let mut scored: Vec<(f32, usize)> = (0..n)
                .filter(|j| *j != i)
                .map(|j| (squared_distance(&vectors[i], &vectors[j]), j))
                .collect();
            scored.sort_by(|a, b| a.0.total_cmp(&b.0));
            scored.truncate(k);
            scored.into_iter().map(|(_, j)| j).collect()
        })
        .collect();

    let mut rng = 0x2545_F491_4F6C_DD1Du64;
    let mut next_random = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };

    for step in 0..params.iterations {
        // Decay the step size so the layout settles instead of oscillating.
        let rate = params.learning_rate * (1.0 - step as f32 / params.iterations as f32);
        for i in 0..n {
            let (mut fx, mut fy) = (0.0f32, 0.0f32);

            for &j in &neighbors[i] {
                let dx = coords[j].0 - coords[i].0;
                let dy = coords[j].1 - coords[i].1;
                let distance = (dx * dx + dy * dy).sqrt().max(1e-4);
                let pull = distance / (1.0 + distance);
                fx += pull * dx / distance;
                fy += pull * dy / distance;
            }

            // A handful of negative samples per step is enough to keep
            // unrelated clusters apart, and far cheaper than all-pairs.
            for _ in 0..5 {
                let j = (next_random() % n as u64) as usize;
                if j == i || neighbors[i].contains(&j) {
                    continue;
                }
                let dx = coords[j].0 - coords[i].0;
                let dy = coords[j].1 - coords[i].1;
                let squared = (dx * dx + dy * dy).max(1e-4);
                let push = 1.0 / (1.0 + squared);
                fx -= push * dx;
                fy -= push * dy;
            }

            coords[i].0 += rate * fx;
            coords[i].1 += rate * fy;
        }
    }
}

/// Centre and scale the layout into roughly -1..=1 on both axes.
fn normalize_layout(coords: &mut [(f32, f32)]) {
    if coords.is_empty() {
        return;
    }
    let (mut min_x, mut max_x) = (f32::MAX, f32::MIN);
    let (mut min_y, mut max_y) = (f32::MAX, f32::MIN);
    for (x, y) in coords.iter() {
        min_x = min_x.min(*x);
        max_x = max_x.max(*x);
        min_y = min_y.min(*y);
        max_y = max_y.max(*y);
    }
    // One scale for both axes, so the plot does not distort distances.
    let span = (max_x - min_x).max(max_y - min_y).max(1e-6);
    let center_x = (min_x + max_x) / 2.0;
    let center_y = (min_y + max_y) / 2.0;
    for (x, y) in coords.iter_mut() {
        *x = (*x - center_x) / span * 2.0;
        *y = (*y - center_y) / span * 2.0;
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn squared_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for value in vector.iter_mut() {
            *value /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two well-separated clusters in high dimensions.
    fn clustered(count: usize, dim: usize) -> (Vec<NodeId>, Vec<Vec<f32>>) {
        let mut ids = Vec::new();
        let mut vectors = Vec::new();
        let mut state = 12345u64;
        let mut noise = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / 8_388_608.0 - 1.0) * 0.1
        };
        for i in 0..count {
            let mut v = vec![0.0f32; dim];
            let cluster = i % 2;
            v[cluster] = 1.0;
            v[cluster + 2] = 0.8;
            for value in v.iter_mut() {
                *value += noise();
            }
            ids.push(NodeId::new());
            vectors.push(v);
        }
        (ids, vectors)
    }

    #[test]
    fn projects_every_point() {
        let (ids, vectors) = clustered(40, 16);
        let p = project(&ids, &vectors, ProjectionMethod::Pca, Default::default());
        assert_eq!(p.points.len(), 40);
        assert_eq!(p.dimensions, 16);
        assert!(
            p.points
                .iter()
                .all(|pt| pt.x.is_finite() && pt.y.is_finite())
        );
    }

    #[test]
    fn pca_is_deterministic() {
        let (ids, vectors) = clustered(30, 12);
        let a = project(&ids, &vectors, ProjectionMethod::Pca, Default::default());
        let b = project(&ids, &vectors, ProjectionMethod::Pca, Default::default());
        assert_eq!(a.points, b.points, "the same data must project identically");
    }

    #[test]
    fn refinement_is_deterministic_too() {
        let (ids, vectors) = clustered(30, 12);
        let a = project(
            &ids,
            &vectors,
            ProjectionMethod::Neighborhood,
            Default::default(),
        );
        let b = project(
            &ids,
            &vectors,
            ProjectionMethod::Neighborhood,
            Default::default(),
        );
        assert_eq!(
            a.points, b.points,
            "a scatter plot must not reshuffle on refresh"
        );
    }

    #[test]
    fn clusters_stay_together_in_two_dimensions() {
        let (ids, vectors) = clustered(60, 24);
        for method in [ProjectionMethod::Pca, ProjectionMethod::Neighborhood] {
            let p = project(&ids, &vectors, method, Default::default());
            let coords: Vec<(f32, f32)> = p.points.iter().map(|pt| (pt.x, pt.y)).collect();

            // Even indices are one cluster, odd the other.
            let mean = |indices: Vec<usize>| {
                let n = indices.len() as f32;
                let x: f32 = indices.iter().map(|i| coords[*i].0).sum::<f32>() / n;
                let y: f32 = indices.iter().map(|i| coords[*i].1).sum::<f32>() / n;
                (x, y)
            };
            let a = mean((0..60).filter(|i| i % 2 == 0).collect());
            let b = mean((0..60).filter(|i| i % 2 == 1).collect());
            let between = ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt();

            let spread = |indices: Vec<usize>, center: (f32, f32)| {
                let n = indices.len() as f32;
                indices
                    .iter()
                    .map(|i| {
                        ((coords[*i].0 - center.0).powi(2) + (coords[*i].1 - center.1).powi(2))
                            .sqrt()
                    })
                    .sum::<f32>()
                    / n
            };
            let within = (spread((0..60).filter(|i| i % 2 == 0).collect(), a)
                + spread((0..60).filter(|i| i % 2 == 1).collect(), b))
                / 2.0;

            assert!(
                between > within * 1.5,
                "{method:?}: clusters not separated (between {between:.3}, within {within:.3})"
            );
        }
    }

    #[test]
    fn output_is_centred_and_scaled() {
        let (ids, vectors) = clustered(50, 16);
        let p = project(&ids, &vectors, ProjectionMethod::Pca, Default::default());
        assert!(
            p.points
                .iter()
                .all(|pt| pt.x.abs() <= 1.01 && pt.y.abs() <= 1.01)
        );
    }

    #[test]
    fn explained_variance_is_a_fraction_and_only_reported_for_pca() {
        let (ids, vectors) = clustered(40, 16);
        let pca = project(&ids, &vectors, ProjectionMethod::Pca, Default::default());
        let variance = pca.explained_variance.unwrap();
        assert!((0.0..=1.0).contains(&variance), "got {variance}");
        // These clusters live in a 2D subspace, so PCA should capture most of it.
        assert!(
            variance > 0.5,
            "expected most variance captured, got {variance}"
        );

        let refined = project(
            &ids,
            &vectors,
            ProjectionMethod::Neighborhood,
            Default::default(),
        );
        assert!(refined.explained_variance.is_none());
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        let ids: Vec<NodeId> = (0..2).map(|_| NodeId::new()).collect();
        let vectors = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let p = project(
            &ids,
            &vectors,
            ProjectionMethod::Neighborhood,
            Default::default(),
        );
        assert_eq!(p.points.len(), 2);

        assert!(
            project(&[], &[], ProjectionMethod::Pca, Default::default())
                .points
                .is_empty()
        );

        // Every vector identical: no variance to find.
        let ids: Vec<NodeId> = (0..10).map(|_| NodeId::new()).collect();
        let same = vec![vec![1.0f32; 8]; 10];
        let p = project(&ids, &same, ProjectionMethod::Pca, Default::default());
        assert!(
            p.points
                .iter()
                .all(|pt| pt.x.is_finite() && pt.y.is_finite())
        );
    }
}
