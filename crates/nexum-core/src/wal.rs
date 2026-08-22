//! Write-ahead log.
//!
//! Every mutation is appended here and fsynced *before* it is applied to the
//! key-value tables, and the tables record the highest LSN they have absorbed.
//! On open, anything past that watermark is replayed. redb is already
//! transactional, so this is not strictly required for crash safety — the log
//! earns its place as the durable audit trail the spec asks for, and recovery
//! falls out of it for free.
//!
//! Record framing: `[magic u32][len u32][checksum u32][payload len bytes]`.
//! A torn tail (partial write at the moment of a crash) is detected by the
//! checksum and truncated on the next open rather than treated as corruption.

use crate::codec;
use crate::error::{Error, Result};
use crate::id::{NodeId, Timestamp};
use crate::model::{Edge, Node};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const MAGIC: u32 = 0x4E58_5701; // "NXW\x01"
const HEADER_LEN: usize = 12;

/// Log sequence number. Monotonic, starts at 1.
pub type Lsn = u64;

/// One mutation, as recorded in the log.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum WalOp {
    PutNode(Box<Node>),
    PutEdge(Box<Edge>),
    PutVector {
        namespace: String,
        node_id: NodeId,
        vector: Vec<f32>,
    },
    /// Tombstone a node: hides it from reads without rewriting history.
    DeleteNode(NodeId),
    /// Marks a point at which the tables were fully caught up.
    Checkpoint,
}

impl WalOp {
    /// Short label for audit output.
    pub fn kind(&self) -> &'static str {
        match self {
            WalOp::PutNode(_) => "put_node",
            WalOp::PutEdge(_) => "put_edge",
            WalOp::PutVector { .. } => "put_vector",
            WalOp::DeleteNode(_) => "delete_node",
            WalOp::Checkpoint => "checkpoint",
        }
    }
}

/// A framed log entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WalRecord {
    pub lsn: Lsn,
    pub at: Timestamp,
    pub op: WalOp,
}

/// Append-only log file handle.
pub struct Wal {
    path: PathBuf,
    writer: BufWriter<File>,
    next_lsn: Lsn,
    /// Bytes written since the last fsync.
    dirty: bool,
}

impl Wal {
    /// Open (or create) the log at `path`, recovering the next LSN from its
    /// contents and truncating any torn tail.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let (next_lsn, valid_len) = Self::scan_tail(&path)?;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;

        // Drop a partially-written trailing record so the next append starts
        // from a clean frame boundary.
        if file.metadata()?.len() != valid_len {
            tracing::warn!(
                path = %path.display(),
                from = file.metadata()?.len(),
                to = valid_len,
                "truncating torn write-ahead log tail"
            );
            file.set_len(valid_len)?;
        }

        let mut writer = BufWriter::new(file);
        writer.seek(SeekFrom::End(0))?;

        Ok(Wal {
            path,
            writer,
            next_lsn,
            dirty: false,
        })
    }

    /// Walk the log from the start, returning the next free LSN and the byte
    /// offset just past the last intact record.
    fn scan_tail(path: &Path) -> Result<(Lsn, u64)> {
        if !path.exists() {
            return Ok((1, 0));
        }
        let mut reader = BufReader::new(File::open(path)?);
        let mut offset = 0u64;
        let mut next_lsn = 1u64;

        loop {
            let mut header = [0u8; HEADER_LEN];
            match reader.read_exact(&mut header) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
            let magic = u32::from_le_bytes(header[0..4].try_into().unwrap());
            if magic != MAGIC {
                return Err(Error::WalCorrupt {
                    offset,
                    reason: format!("bad magic {magic:#010x}"),
                });
            }
            let len = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
            let checksum = u32::from_le_bytes(header[8..12].try_into().unwrap());

            let mut payload = vec![0u8; len];
            if reader.read_exact(&mut payload).is_err() {
                break; // torn tail
            }
            if checksum_of(&payload) != checksum {
                break; // torn tail
            }
            let record: WalRecord = codec::decode(&payload)?;
            next_lsn = record.lsn + 1;
            offset += (HEADER_LEN + len) as u64;
        }

        Ok((next_lsn, offset))
    }

    /// The LSN that the next appended record will receive.
    pub fn next_lsn(&self) -> Lsn {
        self.next_lsn
    }

    /// Append one operation. Returns its LSN. Not durable until `sync`.
    pub fn append(&mut self, op: WalOp) -> Result<Lsn> {
        let record = WalRecord {
            lsn: self.next_lsn,
            at: Timestamp::now(),
            op,
        };
        let payload = codec::encode(&record)?;
        self.writer.write_all(&MAGIC.to_le_bytes())?;
        self.writer.write_all(&(payload.len() as u32).to_le_bytes())?;
        self.writer.write_all(&checksum_of(&payload).to_le_bytes())?;
        self.writer.write_all(&payload)?;
        self.next_lsn += 1;
        self.dirty = true;
        Ok(record.lsn)
    }

    /// Flush and fsync, making everything appended so far durable.
    pub fn sync(&mut self) -> Result<()> {
        if !self.dirty {
            return Ok(());
        }
        self.writer.flush()?;
        self.writer.get_ref().sync_data()?;
        self.dirty = false;
        Ok(())
    }

    /// Replay records with `lsn > after`, oldest first.
    pub fn replay_from(&self, after: Lsn) -> Result<Vec<WalRecord>> {
        Self::read_all(&self.path)?
            .into_iter()
            .filter(|r| r.lsn > after)
            .map(Ok)
            .collect()
    }

    /// Read every intact record in the log.
    pub fn read_all(path: impl AsRef<Path>) -> Result<Vec<WalRecord>> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Vec::new());
        }
        let mut reader = BufReader::new(File::open(path)?);
        let mut records = Vec::new();

        loop {
            let mut header = [0u8; HEADER_LEN];
            match reader.read_exact(&mut header) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
            let len = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
            let checksum = u32::from_le_bytes(header[8..12].try_into().unwrap());
            let mut payload = vec![0u8; len];
            if reader.read_exact(&mut payload).is_err() || checksum_of(&payload) != checksum {
                break;
            }
            records.push(codec::decode(&payload)?);
        }
        Ok(records)
    }

    /// Size of the log on disk, in bytes.
    pub fn size_bytes(&self) -> Result<u64> {
        Ok(self.writer.get_ref().metadata()?.len())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Discard every record at or below `through`, keeping the rest.
    ///
    /// Only safe once the tables have absorbed `through`; the caller owns that
    /// invariant. Rewrites via a temp file so an interrupted compaction leaves
    /// the original log intact.
    pub fn compact(&mut self, through: Lsn) -> Result<u64> {
        self.sync()?;
        let kept: Vec<WalRecord> = Self::read_all(&self.path)?
            .into_iter()
            .filter(|r| r.lsn > through)
            .collect();

        let tmp = self.path.with_extension("log.compacting");
        {
            let mut out = BufWriter::new(File::create(&tmp)?);
            for record in &kept {
                let payload = codec::encode(record)?;
                out.write_all(&MAGIC.to_le_bytes())?;
                out.write_all(&(payload.len() as u32).to_le_bytes())?;
                out.write_all(&checksum_of(&payload).to_le_bytes())?;
                out.write_all(&payload)?;
            }
            out.flush()?;
            out.get_ref().sync_data()?;
        }
        std::fs::rename(&tmp, &self.path)?;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.path)?;
        let reclaimed = file.metadata()?.len();
        self.writer = BufWriter::new(file);
        self.writer.seek(SeekFrom::End(0))?;
        self.dirty = false;
        Ok(reclaimed)
    }
}

impl Drop for Wal {
    fn drop(&mut self) {
        // Best-effort: a failure here has nowhere useful to go, but losing the
        // tail silently would be worse than a log line.
        if let Err(e) = self.sync() {
            tracing::error!(error = %e, "failed to sync write-ahead log on close");
        }
    }
}

fn checksum_of(payload: &[u8]) -> u32 {
    let digest = blake3::hash(payload);
    u32::from_le_bytes(digest.as_bytes()[..4].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::NodeId;
    use crate::model::EdgeType;

    fn temp_wal() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.log");
        (dir, path)
    }

    #[test]
    fn append_then_replay_returns_records_in_order() {
        let (_dir, path) = temp_wal();
        let mut wal = Wal::open(&path).unwrap();
        let a = NodeId::new();
        let b = NodeId::new();
        wal.append(WalOp::DeleteNode(a)).unwrap();
        wal.append(WalOp::PutEdge(Box::new(Edge::new(a, b, EdgeType::PartOf))))
            .unwrap();
        wal.sync().unwrap();

        let records = wal.replay_from(0).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].lsn, 1);
        assert_eq!(records[1].lsn, 2);
        assert_eq!(records[0].op, WalOp::DeleteNode(a));
    }

    #[test]
    fn replay_respects_the_watermark() {
        let (_dir, path) = temp_wal();
        let mut wal = Wal::open(&path).unwrap();
        for _ in 0..5 {
            wal.append(WalOp::Checkpoint).unwrap();
        }
        wal.sync().unwrap();
        assert_eq!(wal.replay_from(3).unwrap().len(), 2);
    }

    #[test]
    fn lsn_continues_across_reopen() {
        let (_dir, path) = temp_wal();
        {
            let mut wal = Wal::open(&path).unwrap();
            wal.append(WalOp::Checkpoint).unwrap();
            wal.append(WalOp::Checkpoint).unwrap();
            wal.sync().unwrap();
        }
        let wal = Wal::open(&path).unwrap();
        assert_eq!(wal.next_lsn(), 3);
    }

    #[test]
    fn torn_tail_is_truncated_not_fatal() {
        let (_dir, path) = temp_wal();
        {
            let mut wal = Wal::open(&path).unwrap();
            wal.append(WalOp::Checkpoint).unwrap();
            wal.append(WalOp::Checkpoint).unwrap();
            wal.sync().unwrap();
        }
        // Simulate a crash midway through a third record.
        {
            let file = OpenOptions::new().append(true).open(&path).unwrap();
            let mut w = BufWriter::new(file);
            w.write_all(&MAGIC.to_le_bytes()).unwrap();
            w.write_all(&99u32.to_le_bytes()).unwrap();
            w.write_all(&0u32.to_le_bytes()).unwrap();
            w.write_all(b"partial").unwrap();
            w.flush().unwrap();
        }

        let wal = Wal::open(&path).unwrap();
        assert_eq!(wal.next_lsn(), 3, "torn record must not consume an LSN");
        assert_eq!(wal.replay_from(0).unwrap().len(), 2);
    }

    #[test]
    fn compaction_drops_absorbed_records_and_keeps_the_rest() {
        let (_dir, path) = temp_wal();
        let mut wal = Wal::open(&path).unwrap();
        for _ in 0..10 {
            wal.append(WalOp::Checkpoint).unwrap();
        }
        wal.sync().unwrap();
        wal.compact(7).unwrap();

        let remaining = wal.replay_from(0).unwrap();
        assert_eq!(remaining.len(), 3);
        assert_eq!(remaining[0].lsn, 8);

        // The log must still be appendable, with LSNs continuing forward.
        assert_eq!(wal.append(WalOp::Checkpoint).unwrap(), 11);
    }
}
