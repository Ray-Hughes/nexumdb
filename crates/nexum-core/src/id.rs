//! Identifiers and timestamps.
//!
//! Node IDs are UUIDv7 so that raw byte order matches creation order — range
//! scans over the node table come back oldest-first for free.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Identifier for any node in the graph.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NodeId(uuid::Uuid);

impl NodeId {
    /// Mint a fresh, time-ordered identifier.
    pub fn new() -> Self {
        NodeId(uuid::Uuid::now_v7())
    }

    /// Derive a stable identifier from a namespace and seed bytes.
    ///
    /// Used for content-addressed nodes (entity canonicalisation, dedup) where
    /// the same logical thing must land on the same ID across runs.
    pub fn derived(namespace: &str, seed: &[u8]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(namespace.as_bytes());
        hasher.update(b"\x00");
        hasher.update(seed);
        let digest = hasher.finalize();
        let mut bytes = [0u8; 16];
        bytes.copy_from_slice(&digest.as_bytes()[..16]);
        // Stamp UUIDv8 (custom) so derived IDs are distinguishable from v7.
        bytes[6] = (bytes[6] & 0x0f) | 0x80;
        bytes[8] = (bytes[8] & 0x3f) | 0x80;
        NodeId(uuid::Uuid::from_bytes(bytes))
    }

    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        NodeId(uuid::Uuid::from_bytes(bytes))
    }

    pub fn as_bytes(&self) -> &[u8; 16] {
        self.0.as_bytes()
    }

    pub fn to_bytes(self) -> [u8; 16] {
        *self.0.as_bytes()
    }
}

impl Default for NodeId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Debug for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NodeId({})", self.0)
    }
}

impl FromStr for NodeId {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        uuid::Uuid::parse_str(s)
            .map(NodeId)
            .map_err(|_| Error::InvalidId(s.to_string()))
    }
}

/// Milliseconds since the Unix epoch.
///
/// Stored as a plain integer so timestamps sort correctly as raw bytes and
/// serialise identically in every language that talks to the HTTP API.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Timestamp(pub i64);

impl Timestamp {
    /// Current wall-clock time.
    pub fn now() -> Self {
        let now = time::OffsetDateTime::now_utc();
        Timestamp((now.unix_timestamp_nanos() / 1_000_000) as i64)
    }

    pub const fn from_millis(ms: i64) -> Self {
        Timestamp(ms)
    }

    pub const fn as_millis(self) -> i64 {
        self.0
    }

    /// Render as RFC 3339, e.g. `2026-08-22T17:06:00.123Z`.
    pub fn to_rfc3339(self) -> String {
        let nanos = (self.0 as i128) * 1_000_000;
        match time::OffsetDateTime::from_unix_timestamp_nanos(nanos) {
            Ok(dt) => dt
                .format(&time::format_description::well_known::Rfc3339)
                .unwrap_or_else(|_| self.0.to_string()),
            Err(_) => self.0.to_string(),
        }
    }

    /// Parse an RFC 3339 timestamp.
    pub fn parse_rfc3339(s: &str) -> Result<Self> {
        let dt = time::OffsetDateTime::parse(s, &time::format_description::well_known::Rfc3339)
            .map_err(|e| Error::InvalidArgument(format!("bad timestamp `{s}`: {e}")))?;
        Ok(Timestamp((dt.unix_timestamp_nanos() / 1_000_000) as i64))
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_rfc3339())
    }
}

impl fmt::Debug for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Timestamp({})", self.to_rfc3339())
    }
}

/// A BLAKE3 content hash, rendered as lowercase hex.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ContentHash(pub String);

impl ContentHash {
    pub fn of(bytes: &[u8]) -> Self {
        ContentHash(blake3::hash(bytes).to_hex().to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl fmt::Debug for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ContentHash({})", &self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_ids_are_time_ordered() {
        let a = NodeId::new();
        let b = NodeId::new();
        assert!(a.to_bytes() < b.to_bytes() || a.to_bytes()[..6] == b.to_bytes()[..6]);
    }

    #[test]
    fn derived_ids_are_stable() {
        let a = NodeId::derived("entity", b"Ada Lovelace|person");
        let b = NodeId::derived("entity", b"Ada Lovelace|person");
        let c = NodeId::derived("entity", b"Alan Turing|person");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn node_id_roundtrips_through_string() {
        let id = NodeId::new();
        assert_eq!(id, id.to_string().parse::<NodeId>().unwrap());
    }

    #[test]
    fn timestamp_roundtrips_through_rfc3339() {
        let ts = Timestamp::from_millis(1_755_882_360_123);
        assert_eq!(ts, Timestamp::parse_rfc3339(&ts.to_rfc3339()).unwrap());
    }
}
