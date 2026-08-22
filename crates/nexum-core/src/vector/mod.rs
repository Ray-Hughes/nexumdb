//! Vector indexing.
//!
//! One HNSW graph per `(model, dim)` namespace, so a database can hold several
//! embedding models at once and an embedding-model upgrade adds an index
//! rather than invalidating one.

pub mod distance;
pub mod hnsw;

pub use distance::Metric;
pub use hnsw::{Hnsw, HnswParams, HnswSnapshot, ScoredNode};

use crate::codec;
use crate::error::{Error, Result};
use crate::id::NodeId;
use crate::store::{ReadView, Store, WriteBatch};
use std::collections::BTreeMap;

/// Every vector index in a database, keyed by namespace.
///
/// Held in memory and persisted on commit. Graph structure is serialised into
/// the store; the vectors themselves stay in the vector table and are read
/// back on open.
pub struct VectorIndexSet {
    indexes: BTreeMap<String, Hnsw>,
    params: HnswParams,
}

impl VectorIndexSet {
    pub fn new(params: HnswParams) -> Self {
        VectorIndexSet {
            indexes: BTreeMap::new(),
            params,
        }
    }

    /// Load every namespace from the store.
    ///
    /// A namespace with a persisted graph is restored; one without — a
    /// database whose indexes were never flushed, or whose graph was
    /// invalidated — is rebuilt from its vectors.
    pub fn load(store: &Store, params: HnswParams) -> Result<Self> {
        let view = store.read()?;
        let mut set = VectorIndexSet::new(params);

        for (namespace, info) in view.namespaces()? {
            let index = match view.get_hnsw(&namespace)? {
                Some(bytes) => {
                    let snapshot: HnswSnapshot = codec::decode(&bytes)?;
                    Hnsw::restore(snapshot, |id| view.get_vector(&namespace, id))?
                }
                None => {
                    tracing::info!(%namespace, count = info.count, "rebuilding vector index");
                    Self::build_from_store(&view, &namespace, info.dim, params)?
                }
            };
            set.indexes.insert(namespace, index);
        }
        Ok(set)
    }

    fn build_from_store(
        view: &ReadView,
        namespace: &str,
        dim: usize,
        params: HnswParams,
    ) -> Result<Hnsw> {
        let mut index = Hnsw::new(dim, params);
        view.scan_vectors(namespace, |id, vector| index.insert(id, vector))?;
        Ok(index)
    }

    /// Namespaces currently loaded.
    pub fn namespaces(&self) -> Vec<&str> {
        self.indexes.keys().map(String::as_str).collect()
    }

    pub fn get(&self, namespace: &str) -> Option<&Hnsw> {
        self.indexes.get(namespace)
    }

    /// Total live vectors across every namespace.
    pub fn total_vectors(&self) -> usize {
        self.indexes.values().map(Hnsw::len).sum()
    }

    /// Add a vector, creating the namespace's index on first use.
    pub fn insert(&mut self, namespace: &str, id: NodeId, vector: &[f32]) -> Result<()> {
        let index = self
            .indexes
            .entry(namespace.to_string())
            .or_insert_with(|| Hnsw::new(vector.len(), self.params));

        if index.dim() != vector.len() {
            return Err(Error::DimensionMismatch {
                namespace: namespace.to_string(),
                expected: index.dim(),
                got: vector.len(),
            });
        }
        index.insert(id, vector)
    }

    /// Drop a node from one namespace.
    pub fn remove(&mut self, namespace: &str, id: NodeId) -> bool {
        self.indexes
            .get_mut(namespace)
            .is_some_and(|index| index.remove(id))
    }

    /// Drop a node from every namespace — what a node tombstone implies.
    pub fn remove_everywhere(&mut self, id: NodeId) -> usize {
        self.indexes
            .values_mut()
            .map(|index| index.remove(id))
            .filter(|removed| *removed)
            .count()
    }

    /// Nearest neighbours within one namespace.
    pub fn search(
        &self,
        namespace: &str,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
        filter: Option<&dyn Fn(NodeId) -> bool>,
    ) -> Result<Vec<ScoredNode>> {
        let index = self
            .indexes
            .get(namespace)
            .ok_or_else(|| Error::UnknownNamespace(namespace.to_string()))?;
        index.search(query, k, ef, filter)
    }

    /// The only namespace, when there is exactly one.
    ///
    /// Lets the CLI and API omit `--model` in the common single-model case
    /// without guessing when the choice is genuinely ambiguous.
    pub fn sole_namespace(&self) -> Option<&str> {
        match self.indexes.len() {
            1 => self.indexes.keys().next().map(String::as_str),
            _ => None,
        }
    }

    /// Resolve a namespace from an optional model hint.
    pub fn resolve(&self, model: Option<&str>) -> Result<String> {
        match model {
            Some(model) => {
                if self.indexes.contains_key(model) {
                    return Ok(model.to_string());
                }
                // Callers usually pass a bare model name; match the `model:dim`
                // namespace it implies, as long as it is unambiguous.
                let matches: Vec<&String> = self
                    .indexes
                    .keys()
                    .filter(|ns| ns.rsplit_once(':').is_some_and(|(m, _)| m == model))
                    .collect();
                match matches.as_slice() {
                    [one] => Ok((*one).clone()),
                    [] => Err(Error::UnknownNamespace(model.to_string())),
                    many => Err(Error::InvalidArgument(format!(
                        "model `{model}` is indexed at several dimensions ({}); pass the full namespace",
                        many.iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))),
                }
            }
            None => {
                self.sole_namespace()
                    .map(str::to_string)
                    .ok_or_else(|| match self.indexes.len() {
                        0 => Error::InvalidArgument(
                            "this database has no embeddings yet — ingest something first".into(),
                        ),
                        _ => Error::InvalidArgument(format!(
                            "several embedding models present ({}); specify one",
                            self.namespaces().join(", ")
                        )),
                    })
            }
        }
    }

    /// Write every index's graph into the batch.
    pub fn persist(&self, batch: &WriteBatch) -> Result<()> {
        for (namespace, index) in &self.indexes {
            batch.put_hnsw(namespace, &codec::encode(&index.snapshot())?)?;
        }
        Ok(())
    }

    /// Rebuild any index that has accumulated too many tombstones.
    pub fn compact(&mut self) -> Result<Vec<String>> {
        let mut rebuilt = Vec::new();
        for (namespace, index) in self.indexes.iter_mut() {
            if index.rebuild_needed() {
                *index = index.rebuild()?;
                rebuilt.push(namespace.clone());
            }
        }
        Ok(rebuilt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set() -> VectorIndexSet {
        VectorIndexSet::new(HnswParams::default())
    }

    #[test]
    fn namespaces_are_independent() {
        let mut set = set();
        let a = NodeId::new();
        let b = NodeId::new();
        set.insert("m1:3", a, &[1.0, 0.0, 0.0]).unwrap();
        set.insert("m2:3", b, &[1.0, 0.0, 0.0]).unwrap();

        let hits = set.search("m1:3", &[1.0, 0.0, 0.0], 5, None, None).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, a);
        assert_eq!(set.total_vectors(), 2);
    }

    #[test]
    fn searching_an_unknown_namespace_is_an_error() {
        let set = set();
        assert!(matches!(
            set.search("nope:3", &[1.0, 0.0, 0.0], 1, None, None),
            Err(Error::UnknownNamespace(_))
        ));
    }

    #[test]
    fn dimension_mismatch_within_a_namespace_is_rejected() {
        let mut set = set();
        set.insert("m:3", NodeId::new(), &[1.0, 0.0, 0.0]).unwrap();
        assert!(matches!(
            set.insert("m:3", NodeId::new(), &[1.0, 0.0]),
            Err(Error::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn resolve_picks_the_sole_namespace() {
        let mut set = set();
        set.insert("m:3", NodeId::new(), &[1.0, 0.0, 0.0]).unwrap();
        assert_eq!(set.resolve(None).unwrap(), "m:3");
        assert_eq!(set.resolve(Some("m")).unwrap(), "m:3");
        assert_eq!(set.resolve(Some("m:3")).unwrap(), "m:3");
    }

    #[test]
    fn resolve_refuses_to_guess_between_models() {
        let mut set = set();
        set.insert("a:3", NodeId::new(), &[1.0, 0.0, 0.0]).unwrap();
        set.insert("b:3", NodeId::new(), &[1.0, 0.0, 0.0]).unwrap();
        assert!(set.resolve(None).is_err());
        assert_eq!(set.resolve(Some("a")).unwrap(), "a:3");
    }

    #[test]
    fn resolve_refuses_an_ambiguous_model_across_dimensions() {
        let mut set = set();
        set.insert("m:3", NodeId::new(), &[1.0, 0.0, 0.0]).unwrap();
        set.insert("m:2", NodeId::new(), &[1.0, 0.0]).unwrap();
        let err = set.resolve(Some("m")).unwrap_err().to_string();
        assert!(err.contains("several dimensions"), "got: {err}");
    }

    #[test]
    fn resolve_on_an_empty_database_explains_itself() {
        let err = set().resolve(None).unwrap_err().to_string();
        assert!(err.contains("no embeddings"), "got: {err}");
    }

    #[test]
    fn remove_everywhere_clears_every_namespace() {
        let mut set = set();
        let id = NodeId::new();
        set.insert("a:2", id, &[1.0, 0.0]).unwrap();
        set.insert("b:2", id, &[0.0, 1.0]).unwrap();
        assert_eq!(set.remove_everywhere(id), 2);
        assert_eq!(set.total_vectors(), 0);
    }
}
