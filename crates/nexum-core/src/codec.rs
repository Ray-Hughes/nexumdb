//! Binary serialisation for on-disk records.
//!
//! Node and edge records are encoded with bincode rather than the protobuf the
//! spec suggested. Both are compact and fast; bincode needs no codegen step and
//! no `protoc` on the build host, which keeps Windows and Linux builds as cheap
//! as macOS ones. The wire format for clients is JSON either way, so this is a
//! purely internal choice — see `docs/architecture.md`.

use crate::error::{Error, Result};
use serde::Serialize;
use serde::de::DeserializeOwned;

/// On-disk format version. Bumped whenever a record layout changes
/// incompatibly; `Db::open` refuses databases it does not understand.
pub const FORMAT_VERSION: u32 = 1;

const fn config() -> bincode::config::Configuration {
    bincode::config::standard()
}

/// Encode a value to its on-disk representation.
pub fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    bincode::serde::encode_to_vec(value, config()).map_err(Error::codec)
}

/// Decode a value from its on-disk representation.
pub fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    let (value, _) = bincode::serde::decode_from_slice(bytes, config()).map_err(Error::codec)?;
    Ok(value)
}

/// Encode a `f32` vector as little-endian bytes.
///
/// Vectors are the bulk of the database, so they skip the generic path and go
/// straight to raw floats — no length prefix, no per-element framing.
pub fn encode_vector(vector: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vector.len() * 4);
    for v in vector {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Decode a `f32` vector from little-endian bytes.
pub fn decode_vector(bytes: &[u8]) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(Error::Codec(format!(
            "vector payload of {} bytes is not a whole number of f32s",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vectors_roundtrip() {
        let v = vec![0.0f32, 1.5, -2.25, f32::MIN_POSITIVE];
        assert_eq!(v, decode_vector(&encode_vector(&v)).unwrap());
    }

    #[test]
    fn empty_vector_roundtrips() {
        assert_eq!(Vec::<f32>::new(), decode_vector(&encode_vector(&[])).unwrap());
    }

    #[test]
    fn ragged_vector_payload_is_rejected() {
        assert!(decode_vector(&[0, 1, 2]).is_err());
    }

    #[test]
    fn structs_roundtrip() {
        let original = crate::model::Edge::new(
            crate::id::NodeId::new(),
            crate::id::NodeId::new(),
            crate::model::EdgeType::Mentions,
        )
        .with("relation_type", "cites");
        let decoded: crate::model::Edge = decode(&encode(&original).unwrap()).unwrap();
        assert_eq!(original, decoded);
    }
}
