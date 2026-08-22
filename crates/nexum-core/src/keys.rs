//! Composite key encoding for the key-value tables.
//!
//! Keys are laid out so that the prefix of a lookup is always a contiguous
//! byte range: adjacency keys put `(node, direction, edge_type)` up front so a
//! neighbour lookup is one range scan, and the kind index leads with the node
//! kind so "every document" is likewise one scan. Nothing here is allowed to
//! change without a format version bump — these bytes are the on-disk layout.

use crate::error::{Error, Result};
use crate::id::NodeId;
use crate::model::{Direction, EdgeType};

/// Length of an encoded adjacency key.
pub const ADJACENCY_KEY_LEN: usize = 16 + 1 + 1 + 16;
/// Length of the `(node, direction, edge_type)` prefix of an adjacency key.
pub const ADJACENCY_PREFIX_LEN: usize = 16 + 1 + 1;

/// Physical direction stored in an adjacency key.
///
/// Distinct from [`Direction`], which is a *query* concept and may be `Both`;
/// on disk every edge is written twice, once each way.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StoredDirection {
    Out = 0,
    In = 1,
}

impl StoredDirection {
    /// The stored directions a query direction needs to scan.
    pub fn for_query(direction: Direction) -> &'static [StoredDirection] {
        match direction {
            Direction::Out => &[StoredDirection::Out],
            Direction::In => &[StoredDirection::In],
            Direction::Both => &[StoredDirection::Out, StoredDirection::In],
        }
    }
}

/// `node || direction || edge_type || other`
pub fn adjacency(
    node: NodeId,
    direction: StoredDirection,
    edge_type: EdgeType,
    other: NodeId,
) -> Vec<u8> {
    let mut key = Vec::with_capacity(ADJACENCY_KEY_LEN);
    key.extend_from_slice(node.as_bytes());
    key.push(direction as u8);
    key.push(edge_type.code());
    key.extend_from_slice(other.as_bytes());
    key
}

/// The inclusive/exclusive range covering every neighbour of `node` along
/// `edge_type` in `direction`.
pub fn adjacency_range(
    node: NodeId,
    direction: StoredDirection,
    edge_type: EdgeType,
) -> (Vec<u8>, Vec<u8>) {
    let mut lo = Vec::with_capacity(ADJACENCY_KEY_LEN);
    lo.extend_from_slice(node.as_bytes());
    lo.push(direction as u8);
    lo.push(edge_type.code());
    let mut hi = lo.clone();
    lo.extend_from_slice(&[0x00; 16]);
    hi.extend_from_slice(&[0xff; 16]);
    (lo, hi)
}

/// Pull the far endpoint back out of an adjacency key.
pub fn adjacency_target(key: &[u8]) -> Result<NodeId> {
    if key.len() != ADJACENCY_KEY_LEN {
        return Err(Error::Storage(format!(
            "adjacency key is {} bytes, expected {ADJACENCY_KEY_LEN}",
            key.len()
        )));
    }
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&key[ADJACENCY_PREFIX_LEN..]);
    Ok(NodeId::from_bytes(bytes))
}

/// Pull the edge type back out of an adjacency key.
pub fn adjacency_edge_type(key: &[u8]) -> Result<EdgeType> {
    if key.len() != ADJACENCY_KEY_LEN {
        return Err(Error::Storage("malformed adjacency key".into()));
    }
    EdgeType::from_code(key[17])
        .ok_or_else(|| Error::Storage(format!("unknown edge type code {}", key[17])))
}

/// `kind || node`
pub fn kind_index(kind: crate::model::NodeKind, node: NodeId) -> Vec<u8> {
    let mut key = Vec::with_capacity(17);
    key.push(kind_code(kind));
    key.extend_from_slice(node.as_bytes());
    key
}

/// The range covering every node of one kind.
pub fn kind_index_range(kind: crate::model::NodeKind) -> (Vec<u8>, Vec<u8>) {
    let code = kind_code(kind);
    let mut lo = vec![code];
    let mut hi = vec![code];
    lo.extend_from_slice(&[0x00; 16]);
    hi.extend_from_slice(&[0xff; 16]);
    (lo, hi)
}

/// Recover the node ID from a kind-index key.
pub fn kind_index_node(key: &[u8]) -> Result<NodeId> {
    if key.len() != 17 {
        return Err(Error::Storage("malformed kind index key".into()));
    }
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&key[1..]);
    Ok(NodeId::from_bytes(bytes))
}

/// Stable on-disk discriminant for a node kind. Do not renumber.
pub const fn kind_code(kind: crate::model::NodeKind) -> u8 {
    match kind {
        crate::model::NodeKind::Document => 1,
        crate::model::NodeKind::Chunk => 2,
        crate::model::NodeKind::Entity => 3,
        crate::model::NodeKind::PipelineRun => 4,
    }
}

/// `namespace || 0x00 || node`
///
/// Namespaces are `model:dim`; a NUL separator keeps `m:384` from colliding
/// with a hypothetical `m:3` whose first ID byte happened to be `56`.
pub fn vector(namespace: &str, node: NodeId) -> Vec<u8> {
    let mut key = Vec::with_capacity(namespace.len() + 17);
    key.extend_from_slice(namespace.as_bytes());
    key.push(0x00);
    key.extend_from_slice(node.as_bytes());
    key
}

/// The range covering every vector in one namespace.
pub fn vector_range(namespace: &str) -> (Vec<u8>, Vec<u8>) {
    let mut lo = Vec::with_capacity(namespace.len() + 17);
    lo.extend_from_slice(namespace.as_bytes());
    lo.push(0x00);
    let mut hi = lo.clone();
    lo.extend_from_slice(&[0x00; 16]);
    hi.extend_from_slice(&[0xff; 16]);
    (lo, hi)
}

/// Recover the node ID from a vector key.
pub fn vector_node(key: &[u8]) -> Result<NodeId> {
    if key.len() < 17 {
        return Err(Error::Storage("malformed vector key".into()));
    }
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&key[key.len() - 16..]);
    Ok(NodeId::from_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::NodeKind;

    #[test]
    fn adjacency_keys_roundtrip() {
        let a = NodeId::new();
        let b = NodeId::new();
        let key = adjacency(a, StoredDirection::Out, EdgeType::Mentions, b);
        assert_eq!(key.len(), ADJACENCY_KEY_LEN);
        assert_eq!(adjacency_target(&key).unwrap(), b);
        assert_eq!(adjacency_edge_type(&key).unwrap(), EdgeType::Mentions);
    }

    #[test]
    fn adjacency_range_brackets_matching_keys_only() {
        let a = NodeId::new();
        let (lo, hi) = adjacency_range(a, StoredDirection::Out, EdgeType::Mentions);
        for _ in 0..50 {
            let target = NodeId::new();
            let hit = adjacency(a, StoredDirection::Out, EdgeType::Mentions, target);
            assert!(hit >= lo && hit <= hi);

            // Different edge type, direction, or origin must fall outside.
            let other_type = adjacency(a, StoredDirection::Out, EdgeType::PartOf, target);
            assert!(other_type < lo || other_type > hi);
            let other_dir = adjacency(a, StoredDirection::In, EdgeType::Mentions, target);
            assert!(other_dir < lo || other_dir > hi);
            let other_node = adjacency(NodeId::new(), StoredDirection::Out, EdgeType::Mentions, target);
            assert!(other_node < lo || other_node > hi);
        }
    }

    #[test]
    fn kind_index_keys_roundtrip_and_separate_kinds() {
        for kind in NodeKind::ALL {
            let node = NodeId::new();
            let key = kind_index(kind, node);
            assert_eq!(kind_index_node(&key).unwrap(), node);
            let (lo, hi) = kind_index_range(kind);
            assert!(key >= lo && key <= hi);
            for other in NodeKind::ALL.into_iter().filter(|k| *k != kind) {
                let (olo, ohi) = kind_index_range(other);
                assert!(key < olo || key > ohi);
            }
        }
    }

    #[test]
    fn vector_keys_roundtrip_and_namespaces_do_not_collide() {
        let node = NodeId::new();
        let key = vector("model-a:384", node);
        assert_eq!(vector_node(&key).unwrap(), node);
        let (lo, hi) = vector_range("model-a:384");
        assert!(key >= lo && key <= hi);
        let (olo, ohi) = vector_range("model-a:3");
        assert!(key < olo || key > ohi);
    }
}
