//! Hierarchical Navigable Small World index.
//!
//! Implemented in-tree per Malkov & Yashunin (2016) rather than pulled from a
//! crate, because this database needs three things off-the-shelf HNSW crates
//! do not all offer together: incremental insertion (documents arrive over
//! time), tombstone deletion (superseded versions leave the live set), and
//! *filtered* search (the default query is "latest version only", and applying
//! that filter after a top-k search silently returns too few rows).
//!
//! Level assignment is derived from the node ID rather than an RNG, so
//! rebuilding an index from the same data produces the same graph — which
//! makes bugs reproducible and index rebuilds verifiable.

use crate::error::{Error, Result};
use crate::id::NodeId;
use crate::vector::distance::Metric;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Hard ceiling on layer count. With the default `M` the expected top layer
/// for even a billion vectors is ~7, so this is pure defence against a
/// pathological hash rather than a tuning knob.
const MAX_LEVEL_CAP: usize = 32;

/// Index construction and search parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HnswParams {
    /// Max connections per node on layers above 0.
    pub m: usize,
    /// Max connections on layer 0. Conventionally `2 * m`.
    pub m0: usize,
    /// Candidate-list size during insertion. Higher builds a better graph
    /// more slowly.
    pub ef_construction: usize,
    /// Default candidate-list size at query time. Higher recall, slower.
    pub ef_search: usize,
    pub metric: Metric,
}

impl Default for HnswParams {
    fn default() -> Self {
        HnswParams {
            m: 16,
            m0: 32,
            ef_construction: 200,
            ef_search: 64,
            metric: Metric::Cosine,
        }
    }
}

impl HnswParams {
    /// Max connections at a given layer.
    fn max_links(&self, level: usize) -> usize {
        if level == 0 { self.m0 } else { self.m }
    }

    /// Level-generation normalisation factor, `1 / ln(M)`.
    fn level_multiplier(&self) -> f64 {
        1.0 / (self.m.max(2) as f64).ln()
    }
}

/// One indexed vector.
#[derive(Clone, Debug)]
struct Point {
    id: NodeId,
    /// Stored in prepared form — unit-normalised when the metric asks for it.
    vector: Vec<f32>,
    level: usize,
}

/// A scored candidate, ordered by distance.
///
/// `f32` is not `Ord`, so ordering goes through `total_cmp`; NaN distances
/// therefore sort consistently instead of poisoning the heap.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Candidate {
    distance: f32,
    idx: u32,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.idx.cmp(&other.idx))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Min-heap ordering wrapper.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Nearest(Candidate);

impl Ord for Nearest {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.cmp(&self.0)
    }
}

impl PartialOrd for Nearest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// One search result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoredNode {
    pub id: NodeId,
    /// Metric distance — smaller is closer.
    pub distance: f32,
}

impl ScoredNode {
    /// Distance re-expressed as a 0..=1 similarity for display.
    pub fn similarity(&self, metric: Metric) -> f32 {
        metric.similarity(self.distance)
    }
}

/// The index.
pub struct Hnsw {
    params: HnswParams,
    dim: usize,
    points: Vec<Point>,
    /// `links[idx][level]` — neighbours of `idx` on that layer.
    links: Vec<Vec<Vec<u32>>>,
    id_to_idx: HashMap<NodeId, u32>,
    entry_point: Option<u32>,
    max_level: usize,
    /// Tombstoned points. Still traversed (they hold the graph together) but
    /// never returned.
    deleted: HashSet<u32>,
}

impl Hnsw {
    /// Create an empty index over `dim`-dimensional vectors.
    pub fn new(dim: usize, params: HnswParams) -> Self {
        Hnsw {
            params,
            dim,
            points: Vec::new(),
            links: Vec::new(),
            id_to_idx: HashMap::new(),
            entry_point: None,
            max_level: 0,
            deleted: HashSet::new(),
        }
    }

    pub fn params(&self) -> HnswParams {
        self.params
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn metric(&self) -> Metric {
        self.params.metric
    }

    /// Number of live (non-tombstoned) vectors.
    pub fn len(&self) -> usize {
        self.points.len() - self.deleted.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total points including tombstones — what the graph actually costs.
    pub fn capacity_used(&self) -> usize {
        self.points.len()
    }

    pub fn contains(&self, id: NodeId) -> bool {
        self.id_to_idx
            .get(&id)
            .is_some_and(|idx| !self.deleted.contains(idx))
    }

    /// Insert or replace a vector.
    ///
    /// Re-inserting an existing ID overwrites its vector in place and relinks
    /// it, which is what happens when a chunk is re-embedded by the same model.
    pub fn insert(&mut self, id: NodeId, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dim {
            return Err(Error::DimensionMismatch {
                namespace: format!("dim {}", self.dim),
                expected: self.dim,
                got: vector.len(),
            });
        }

        if let Some(&existing) = self.id_to_idx.get(&id) {
            // Replacing: drop it out of the graph, then re-insert cleanly so
            // its neighbours reflect the new position.
            self.unlink(existing);
            self.points[existing as usize].vector = self.params.metric.prepare(vector);
            self.deleted.remove(&existing);
            self.link_into_graph(existing)?;
            return Ok(());
        }

        let level = self.level_for(id);
        let idx = self.points.len() as u32;
        self.points.push(Point {
            id,
            vector: self.params.metric.prepare(vector),
            level,
        });
        self.links.push(vec![Vec::new(); level + 1]);
        self.id_to_idx.insert(id, idx);
        self.link_into_graph(idx)?;
        Ok(())
    }

    /// Tombstone a vector. Its links stay in place so the graph does not
    /// fragment; call [`Hnsw::rebuild_needed`] to decide when to compact.
    pub fn remove(&mut self, id: NodeId) -> bool {
        match self.id_to_idx.get(&id) {
            Some(&idx) => {
                let newly = self.deleted.insert(idx);
                if newly && self.entry_point == Some(idx) {
                    // Keep the entry point on a live node where possible,
                    // otherwise search starts from something invisible.
                    self.entry_point = self
                        .points
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| !self.deleted.contains(&(*i as u32)))
                        .max_by_key(|(_, p)| p.level)
                        .map(|(i, _)| i as u32);
                    self.max_level = self
                        .entry_point
                        .map(|e| self.points[e as usize].level)
                        .unwrap_or(0);
                }
                newly
            }
            None => false,
        }
    }

    /// Fraction of the graph that is tombstoned. Past roughly a third, a
    /// rebuild pays for itself in both memory and search time.
    pub fn tombstone_ratio(&self) -> f32 {
        if self.points.is_empty() {
            return 0.0;
        }
        self.deleted.len() as f32 / self.points.len() as f32
    }

    /// Whether the index has accumulated enough tombstones to be worth rebuilding.
    pub fn rebuild_needed(&self) -> bool {
        self.points.len() > 1_000 && self.tombstone_ratio() > 0.33
    }

    /// Nearest neighbours to `query`.
    ///
    /// `filter` decides which IDs may be *returned*; filtered-out nodes are
    /// still traversed, because they are often the only path to the ones that
    /// qualify. When a selective filter starves the result list, the search
    /// widens `ef` and retries rather than handing back a short list.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
        filter: Option<&dyn Fn(NodeId) -> bool>,
    ) -> Result<Vec<ScoredNode>> {
        if query.len() != self.dim {
            return Err(Error::DimensionMismatch {
                namespace: format!("dim {}", self.dim),
                expected: self.dim,
                got: query.len(),
            });
        }
        if k == 0 || self.is_empty() {
            return Ok(Vec::new());
        }

        let query = self.params.metric.prepare(query);
        let base_ef = ef.unwrap_or(self.params.ef_search).max(k);
        let ceiling = self.points.len().max(base_ef);

        let mut ef = base_ef;
        loop {
            let results = self.search_once(&query, k, ef, filter);
            let exhausted = ef >= ceiling;
            if results.len() >= k || exhausted || filter.is_none() {
                return Ok(results);
            }
            // A selective filter emptied the candidate list — widen and retry.
            ef = (ef * 4).min(ceiling);
        }
    }

    fn search_once(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter: Option<&dyn Fn(NodeId) -> bool>,
    ) -> Vec<ScoredNode> {
        let Some(entry) = self.entry_point else {
            return Vec::new();
        };

        // Greedy descent through the upper layers narrows the entry point.
        let mut current = entry;
        for level in (1..=self.max_level).rev() {
            current = self.greedy_step(query, current, level);
        }

        let candidates = self.search_layer(query, &[current], ef, 0);

        let mut out: Vec<ScoredNode> = candidates
            .into_iter()
            .filter(|c| !self.deleted.contains(&c.idx))
            .map(|c| ScoredNode {
                id: self.points[c.idx as usize].id,
                distance: c.distance,
            })
            .filter(|s| filter.is_none_or(|f| f(s.id)))
            .collect();

        out.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        out.truncate(k);
        out
    }

    /// Walk downhill on one layer until no neighbour is closer.
    fn greedy_step(&self, query: &[f32], start: u32, level: usize) -> u32 {
        let mut current = start;
        let mut current_distance = self.distance_to(query, current);
        loop {
            let mut improved = false;
            for &neighbor in self.links_at(current, level) {
                let d = self.distance_to(query, neighbor);
                if d < current_distance {
                    current_distance = d;
                    current = neighbor;
                    improved = true;
                }
            }
            if !improved {
                return current;
            }
        }
    }

    /// Best-first search on one layer, returning up to `ef` candidates.
    fn search_layer(&self, query: &[f32], entries: &[u32], ef: usize, level: usize) -> Vec<Candidate> {
        let mut visited: HashSet<u32> = HashSet::with_capacity(ef * 2);
        let mut frontier: BinaryHeap<Nearest> = BinaryHeap::new();
        let mut best: BinaryHeap<Candidate> = BinaryHeap::new();

        for &entry in entries {
            if !visited.insert(entry) {
                continue;
            }
            let candidate = Candidate {
                distance: self.distance_to(query, entry),
                idx: entry,
            };
            frontier.push(Nearest(candidate));
            best.push(candidate);
        }

        while let Some(Nearest(current)) = frontier.pop() {
            // Everything left in the frontier is further than our worst keeper.
            if best.len() >= ef
                && let Some(worst) = best.peek()
                && current.distance > worst.distance
            {
                break;
            }

            for &neighbor in self.links_at(current.idx, level) {
                if !visited.insert(neighbor) {
                    continue;
                }
                let candidate = Candidate {
                    distance: self.distance_to(query, neighbor),
                    idx: neighbor,
                };
                let worst = best.peek().map(|c| c.distance);
                if best.len() < ef || worst.is_none_or(|w| candidate.distance < w) {
                    frontier.push(Nearest(candidate));
                    best.push(candidate);
                    if best.len() > ef {
                        best.pop();
                    }
                }
            }
        }

        let mut out = best.into_vec();
        out.sort();
        out
    }

    /// Wire a point into every layer it belongs to.
    fn link_into_graph(&mut self, idx: u32) -> Result<()> {
        let level = self.points[idx as usize].level;
        let vector = self.points[idx as usize].vector.clone();

        let Some(entry) = self.entry_point else {
            self.entry_point = Some(idx);
            self.max_level = level;
            return Ok(());
        };

        let mut current = entry;
        if self.max_level > level {
            for l in ((level + 1)..=self.max_level).rev() {
                current = self.greedy_step(&vector, current, l);
            }
        }

        let top = level.min(self.max_level);
        for l in (0..=top).rev() {
            let candidates = self.search_layer(&vector, &[current], self.params.ef_construction, l);
            if let Some(closest) = candidates.first() {
                current = closest.idx;
            }
            let selected = self.select_neighbors(&vector, &candidates, self.params.max_links(l), idx);

            self.set_links(idx, l, selected.clone());
            for neighbor in selected {
                self.add_link_pruned(neighbor, idx, l);
            }
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry_point = Some(idx);
        }
        Ok(())
    }

    /// Detach a point from the graph, leaving its slot reusable.
    fn unlink(&mut self, idx: u32) {
        let levels = self.links[idx as usize].len();
        for level in 0..levels {
            let neighbors = std::mem::take(&mut self.links[idx as usize][level]);
            for neighbor in neighbors {
                self.links[neighbor as usize][level].retain(|&n| n != idx);
            }
        }
        if self.entry_point == Some(idx) {
            self.entry_point = self
                .points
                .iter()
                .enumerate()
                .filter(|(i, _)| *i as u32 != idx && !self.deleted.contains(&(*i as u32)))
                .max_by_key(|(_, p)| p.level)
                .map(|(i, _)| i as u32);
            self.max_level = self
                .entry_point
                .map(|e| self.points[e as usize].level)
                .unwrap_or(0);
        }
    }

    /// Neighbour selection heuristic (Malkov & Yashunin, Algorithm 4).
    ///
    /// Prefers candidates that are closer to the new point than to any already
    /// chosen neighbour. That keeps links pointing in diverse directions
    /// instead of clustering on one side, which is what makes the graph
    /// navigable rather than merely dense.
    fn select_neighbors(
        &self,
        vector: &[f32],
        candidates: &[Candidate],
        m: usize,
        exclude: u32,
    ) -> Vec<u32> {
        let mut selected: Vec<u32> = Vec::with_capacity(m);
        let mut discarded: Vec<u32> = Vec::new();

        for candidate in candidates {
            if candidate.idx == exclude {
                continue;
            }
            if selected.len() >= m {
                break;
            }
            let diverse = selected.iter().all(|&chosen| {
                let to_chosen = self.params.metric.distance(
                    &self.points[candidate.idx as usize].vector,
                    &self.points[chosen as usize].vector,
                );
                to_chosen > candidate.distance
            });
            if diverse {
                selected.push(candidate.idx);
            } else {
                discarded.push(candidate.idx);
            }
        }

        // Top up from the discards rather than returning an under-connected
        // node; a sparse layer costs recall.
        for idx in discarded {
            if selected.len() >= m {
                break;
            }
            selected.push(idx);
        }
        let _ = vector;
        selected
    }

    fn set_links(&mut self, idx: u32, level: usize, neighbors: Vec<u32>) {
        let entry = &mut self.links[idx as usize];
        while entry.len() <= level {
            entry.push(Vec::new());
        }
        entry[level] = neighbors;
    }

    /// Add a back-link, pruning the neighbour's list if it overflows.
    fn add_link_pruned(&mut self, node: u32, new_neighbor: u32, level: usize) {
        {
            let entry = &mut self.links[node as usize];
            while entry.len() <= level {
                entry.push(Vec::new());
            }
            if entry[level].contains(&new_neighbor) {
                return;
            }
            entry[level].push(new_neighbor);
            if entry[level].len() <= self.params.max_links(level) {
                return;
            }
        }

        // Over budget: re-run selection over the full neighbour set so the
        // link that gets dropped is the least useful one, not the newest.
        let node_vector = self.points[node as usize].vector.clone();
        let mut candidates: Vec<Candidate> = self.links[node as usize][level]
            .iter()
            .map(|&n| Candidate {
                distance: self
                    .params
                    .metric
                    .distance(&node_vector, &self.points[n as usize].vector),
                idx: n,
            })
            .collect();
        candidates.sort();
        let kept = self.select_neighbors(&node_vector, &candidates, self.params.max_links(level), node);
        self.links[node as usize][level] = kept;
    }

    fn links_at(&self, idx: u32, level: usize) -> &[u32] {
        self.links[idx as usize]
            .get(level)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    fn distance_to(&self, query: &[f32], idx: u32) -> f32 {
        self.params
            .metric
            .distance(query, &self.points[idx as usize].vector)
    }

    /// Deterministic layer assignment from the node ID.
    ///
    /// The canonical algorithm draws `floor(-ln(U) * mL)` from a uniform RNG;
    /// hashing the ID gives the same distribution while making index builds
    /// reproducible.
    fn level_for(&self, id: NodeId) -> usize {
        let digest = blake3::hash(id.as_bytes());
        let raw = u64::from_le_bytes(digest.as_bytes()[..8].try_into().unwrap());
        // Map into (0, 1]; never exactly 0, whose log is -inf.
        let uniform = ((raw >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
        let level = (-uniform.ln() * self.params.level_multiplier()).floor();
        (level.max(0.0) as usize).min(MAX_LEVEL_CAP)
    }

    /// Every live ID in the index.
    pub fn ids(&self) -> Vec<NodeId> {
        self.points
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.deleted.contains(&(*i as u32)))
            .map(|(_, p)| p.id)
            .collect()
    }

    /// Exhaustive nearest-neighbour scan. Exact by construction, and used by
    /// the tests as the ground truth that recall is measured against.
    pub fn search_exact(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&dyn Fn(NodeId) -> bool>,
    ) -> Result<Vec<ScoredNode>> {
        if query.len() != self.dim {
            return Err(Error::DimensionMismatch {
                namespace: format!("dim {}", self.dim),
                expected: self.dim,
                got: query.len(),
            });
        }
        let query = self.params.metric.prepare(query);
        let mut scored: Vec<ScoredNode> = self
            .points
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.deleted.contains(&(*i as u32)))
            .map(|(_, p)| ScoredNode {
                id: p.id,
                distance: self.params.metric.distance(&query, &p.vector),
            })
            .filter(|s| filter.is_none_or(|f| f(s.id)))
            .collect();
        scored.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        scored.truncate(k);
        Ok(scored)
    }

    /// Serialise the graph structure, without the vectors.
    ///
    /// Vectors already live in the vector table; storing them again here would
    /// double the database's largest cost for nothing. [`Hnsw::restore`] reads
    /// them back from the store.
    pub fn snapshot(&self) -> HnswSnapshot {
        HnswSnapshot {
            params: self.params,
            dim: self.dim,
            ids: self.points.iter().map(|p| p.id).collect(),
            levels: self.points.iter().map(|p| p.level as u8).collect(),
            links: self.links.clone(),
            entry_point: self.entry_point,
            max_level: self.max_level,
            deleted: self.deleted.iter().copied().collect(),
        }
    }

    /// Rebuild an index from a snapshot, pulling vectors from `lookup`.
    ///
    /// A node whose vector has vanished is dropped from the graph rather than
    /// resurrected as a zero vector, which would quietly corrupt every
    /// subsequent search.
    pub fn restore(
        snapshot: HnswSnapshot,
        mut lookup: impl FnMut(NodeId) -> Result<Option<Vec<f32>>>,
    ) -> Result<Self> {
        let mut index = Hnsw::new(snapshot.dim, snapshot.params);
        let mut missing = Vec::new();

        for (i, id) in snapshot.ids.iter().enumerate() {
            match lookup(*id)? {
                Some(vector) if vector.len() == snapshot.dim => {
                    index.points.push(Point {
                        id: *id,
                        vector: snapshot.params.metric.prepare(&vector),
                        level: snapshot.levels.get(i).copied().unwrap_or(0) as usize,
                    });
                }
                _ => {
                    missing.push(i as u32);
                    // Keep the slot so link indices stay valid, then tombstone it.
                    index.points.push(Point {
                        id: *id,
                        vector: vec![0.0; snapshot.dim],
                        level: snapshot.levels.get(i).copied().unwrap_or(0) as usize,
                    });
                }
            }
            index.id_to_idx.insert(*id, i as u32);
        }

        index.links = snapshot.links;
        index.entry_point = snapshot.entry_point;
        index.max_level = snapshot.max_level;
        index.deleted = snapshot.deleted.into_iter().collect();
        for idx in &missing {
            index.deleted.insert(*idx);
        }
        if !missing.is_empty() {
            tracing::warn!(
                count = missing.len(),
                "vector index references vectors missing from the store; tombstoning them"
            );
        }

        // Guard against a snapshot whose links array is shorter than its
        // point list — the graph is unusable rather than merely degraded.
        if index.links.len() != index.points.len() {
            return Err(Error::Storage(format!(
                "hnsw snapshot has {} link entries for {} points",
                index.links.len(),
                index.points.len()
            )));
        }
        Ok(index)
    }

    /// Build a fresh graph from scratch, discarding tombstones.
    pub fn rebuild(&self) -> Result<Self> {
        let mut fresh = Hnsw::new(self.dim, self.params);
        for (i, point) in self.points.iter().enumerate() {
            if self.deleted.contains(&(i as u32)) {
                continue;
            }
            // `insert` re-prepares the vector, so hand it the stored form —
            // normalising an already-normalised vector is a no-op.
            fresh.insert(point.id, &point.vector)?;
        }
        Ok(fresh)
    }
}

/// The serialisable form of an index.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HnswSnapshot {
    pub params: HnswParams,
    pub dim: usize,
    pub ids: Vec<NodeId>,
    pub levels: Vec<u8>,
    pub links: Vec<Vec<Vec<u32>>>,
    pub entry_point: Option<u32>,
    pub max_level: usize,
    pub deleted: Vec<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random vectors — no `rand` dependency, and a
    /// failing test reproduces exactly.
    fn vectors(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).max(1);
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..count)
            .map(|_| (0..dim).map(|_| (next() as f32) * 2.0 - 1.0).collect())
            .collect()
    }

    fn build(count: usize, dim: usize, seed: u64) -> (Hnsw, Vec<NodeId>, Vec<Vec<f32>>) {
        let mut index = Hnsw::new(dim, HnswParams::default());
        let data = vectors(count, dim, seed);
        let ids: Vec<NodeId> = (0..count).map(|_| NodeId::new()).collect();
        for (id, vector) in ids.iter().zip(&data) {
            index.insert(*id, vector).unwrap();
        }
        (index, ids, data)
    }

    #[test]
    fn empty_index_returns_nothing() {
        let index = Hnsw::new(4, HnswParams::default());
        assert!(index.is_empty());
        assert!(index.search(&[1.0, 0.0, 0.0, 0.0], 5, None, None).unwrap().is_empty());
    }

    #[test]
    fn finds_an_exact_match_first() {
        let (index, ids, data) = build(200, 32, 42);
        for probe in [0usize, 57, 199] {
            let hits = index.search(&data[probe], 1, None, None).unwrap();
            assert_eq!(hits[0].id, ids[probe], "probe {probe}");
            assert!(hits[0].distance < 1e-4);
        }
    }

    #[test]
    fn recall_against_exhaustive_search_is_high() {
        let (index, _ids, _data) = build(1_000, 48, 7);
        let queries = vectors(40, 48, 999);
        let k = 10;

        let mut hits = 0usize;
        let mut total = 0usize;
        for query in &queries {
            let truth: HashSet<NodeId> = index
                .search_exact(query, k, None)
                .unwrap()
                .into_iter()
                .map(|s| s.id)
                .collect();
            let got = index.search(query, k, Some(128), None).unwrap();
            hits += got.iter().filter(|s| truth.contains(&s.id)).count();
            total += truth.len();
        }
        let recall = hits as f64 / total as f64;
        assert!(recall > 0.95, "recall@{k} was {recall:.3}, expected > 0.95");
    }

    #[test]
    fn results_come_back_sorted_by_distance() {
        let (index, _, data) = build(300, 16, 3);
        let hits = index.search(&data[10], 20, None, None).unwrap();
        assert!(hits.windows(2).all(|w| w[0].distance <= w[1].distance));
    }

    #[test]
    fn removed_vectors_stop_being_returned() {
        let (mut index, ids, data) = build(200, 16, 11);
        let before = index.len();
        assert!(index.remove(ids[5]));
        assert!(!index.remove(ids[5]), "second remove should be a no-op");
        assert_eq!(index.len(), before - 1);
        assert!(!index.contains(ids[5]));

        let hits = index.search(&data[5], 10, None, None).unwrap();
        assert!(hits.iter().all(|h| h.id != ids[5]));
    }

    #[test]
    fn removing_the_entry_point_keeps_the_index_searchable() {
        let (mut index, ids, data) = build(300, 16, 21);
        // Remove whichever point search currently starts from.
        let entry = index.entry_point.unwrap();
        let entry_id = index.points[entry as usize].id;
        assert!(index.remove(entry_id));

        let hits = index.search(&data[0], 5, None, None).unwrap();
        assert!(!hits.is_empty(), "index must stay searchable");
        assert!(hits.iter().all(|h| h.id != entry_id));
        assert!(hits.iter().any(|h| h.id == ids[0]));
    }

    #[test]
    fn reinserting_an_id_replaces_its_vector() {
        let mut index = Hnsw::new(4, HnswParams::default());
        let id = NodeId::new();
        index.insert(id, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert(id, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        assert_eq!(index.len(), 1);

        let hits = index.search(&[0.0, 1.0, 0.0, 0.0], 1, None, None).unwrap();
        assert_eq!(hits[0].id, id);
        assert!(hits[0].distance < 1e-4, "should match the replacement vector");
    }

    #[test]
    fn a_selective_filter_still_fills_the_result_list() {
        let (index, ids, data) = build(800, 32, 5);
        // Keep only 1 in 40 candidates — far too few to survive a plain top-k.
        let keep: HashSet<NodeId> = ids.iter().step_by(40).copied().collect();
        let filter = |id: NodeId| keep.contains(&id);

        let hits = index.search(&data[0], 10, Some(32), Some(&filter)).unwrap();
        assert_eq!(hits.len(), 10, "search should widen ef rather than return short");
        assert!(hits.iter().all(|h| keep.contains(&h.id)));
    }

    #[test]
    fn a_filter_matching_nothing_returns_empty_without_hanging() {
        let (index, _, data) = build(200, 16, 8);
        let filter = |_: NodeId| false;
        assert!(index.search(&data[0], 10, None, Some(&filter)).unwrap().is_empty());
    }

    #[test]
    fn dimension_mismatch_is_rejected_on_insert_and_search() {
        let mut index = Hnsw::new(4, HnswParams::default());
        assert!(index.insert(NodeId::new(), &[1.0, 2.0]).is_err());
        index.insert(NodeId::new(), &[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(index.search(&[1.0, 2.0], 1, None, None).is_err());
    }

    #[test]
    fn snapshot_and_restore_preserve_search_results() {
        let (index, ids, data) = build(400, 24, 13);
        let lookup: HashMap<NodeId, Vec<f32>> = ids.iter().copied().zip(data.clone()).collect();

        let restored = Hnsw::restore(index.snapshot(), |id| Ok(lookup.get(&id).cloned())).unwrap();
        assert_eq!(restored.len(), index.len());

        for probe in [0usize, 100, 399] {
            let before = index.search(&data[probe], 5, None, None).unwrap();
            let after = restored.search(&data[probe], 5, None, None).unwrap();
            assert_eq!(
                before.iter().map(|s| s.id).collect::<Vec<_>>(),
                after.iter().map(|s| s.id).collect::<Vec<_>>(),
                "probe {probe}"
            );
        }
    }

    #[test]
    fn restore_tombstones_vectors_missing_from_the_store() {
        let (index, ids, data) = build(100, 8, 17);
        let mut lookup: HashMap<NodeId, Vec<f32>> = ids.iter().copied().zip(data.clone()).collect();
        lookup.remove(&ids[3]);

        let restored = Hnsw::restore(index.snapshot(), |id| Ok(lookup.get(&id).cloned())).unwrap();
        assert_eq!(restored.len(), index.len() - 1);
        assert!(!restored.contains(ids[3]));
        let hits = restored.search(&data[3], 10, None, None).unwrap();
        assert!(hits.iter().all(|h| h.id != ids[3]));
    }

    #[test]
    fn rebuild_drops_tombstones_and_keeps_results() {
        let (mut index, ids, data) = build(500, 16, 19);
        for id in ids.iter().take(100) {
            index.remove(*id);
        }
        let rebuilt = index.rebuild().unwrap();
        assert_eq!(rebuilt.capacity_used(), 400);
        assert_eq!(rebuilt.len(), 400);

        let hits = rebuilt.search(&data[250], 1, None, None).unwrap();
        assert_eq!(hits[0].id, ids[250]);
    }

    #[test]
    fn levels_are_deterministic_for_an_id() {
        let index = Hnsw::new(4, HnswParams::default());
        let id = NodeId::new();
        assert_eq!(index.level_for(id), index.level_for(id));
    }

    #[test]
    fn level_distribution_is_geometric() {
        let index = Hnsw::new(4, HnswParams::default());
        let mut counts = [0usize; 6];
        let n = 40_000;
        for _ in 0..n {
            let level = index.level_for(NodeId::new()).min(5);
            counts[level] += 1;
        }
        // l = floor(-ln(U) / ln M), so P(l = 0) = 1 - 1/M and each subsequent
        // level is 1/M as likely as the one below it. With M = 16 that puts
        // 93.75% of nodes on layer 0 alone.
        let level0 = counts[0] as f64 / n as f64;
        assert!((0.925..0.950).contains(&level0), "level-0 share was {level0:.4}");
        let level1 = counts[1] as f64 / n as f64;
        assert!((0.050..0.075).contains(&level1), "level-1 share was {level1:.4}");
        assert!(counts[0] > counts[1] && counts[1] > counts[2]);
    }

    #[test]
    fn euclidean_metric_ranks_correctly() {
        let params = HnswParams {
            metric: Metric::Euclidean,
            ..Default::default()
        };
        let mut index = Hnsw::new(2, params);
        let near = NodeId::new();
        let far = NodeId::new();
        index.insert(near, &[1.0, 1.0]).unwrap();
        index.insert(far, &[50.0, 50.0]).unwrap();

        let hits = index.search(&[0.0, 0.0], 2, None, None).unwrap();
        assert_eq!(hits[0].id, near);
        assert!((hits[0].distance - 2.0f32.sqrt()).abs() < 1e-4);
    }

    #[test]
    fn single_point_index_answers_queries() {
        let mut index = Hnsw::new(3, HnswParams::default());
        let id = NodeId::new();
        index.insert(id, &[1.0, 0.0, 0.0]).unwrap();
        let hits = index.search(&[1.0, 0.0, 0.0], 5, None, None).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, id);
    }
}
