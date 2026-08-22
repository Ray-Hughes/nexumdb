//! Resolving user-typed node references.
//!
//! Full UUIDs are unpleasant to type and worse to read off a terminal, so
//! anywhere the CLI takes a node ID it also takes a unique prefix. An
//! ambiguous prefix is an error listing the candidates rather than a silent
//! pick — resolving to the wrong node would be a wrong answer, not a slow one.

use anyhow::{Result, bail};
use nexum_client::Nexum;
use nexum_core::{Node, NodeId, NodeKind, Query};
use std::str::FromStr;

/// Shortest prefix that is still worth accepting.
const MIN_PREFIX: usize = 4;

/// Resolve a full ID or unique prefix to one node.
pub fn node_id(nexum: &Nexum, reference: &str) -> Result<NodeId> {
    let reference = reference.trim();
    if reference.is_empty() {
        bail!("expected a node id");
    }

    // A complete UUID needs no search.
    if let Ok(id) = NodeId::from_str(reference) {
        if nexum.get(id)?.is_some() {
            return Ok(id);
        }
        bail!("no node with id {id}");
    }

    let normalized = reference.to_lowercase();
    if normalized.len() < MIN_PREFIX {
        bail!(
            "`{reference}` is too short to identify a node — give at least {MIN_PREFIX} characters"
        );
    }
    if !normalized
        .chars()
        .all(|c| c.is_ascii_hexdigit() || c == '-')
    {
        bail!("`{reference}` is not a node id or id prefix");
    }

    // Match the printed short form (a suffix) as well as a leading prefix, so
    // an ID copied out of `nexum show` can be pasted straight back in.
    let needle: String = normalized.chars().filter(|c| *c != '-').collect();
    let mut matches: Vec<Node> = Vec::new();
    for kind in NodeKind::ALL {
        for item in nexum.query(&Query::new().seed_kind(kind))?.nodes {
            let hex: String = item
                .id()
                .to_string()
                .chars()
                .filter(|c| *c != '-')
                .collect();
            if hex.starts_with(&needle) || hex.ends_with(&needle) {
                matches.push(item.node);
            }
        }
    }

    match matches.len() {
        1 => Ok(matches[0].id()),
        0 => bail!("no node whose id starts with `{reference}`"),
        _ => {
            let listed = matches
                .iter()
                .take(5)
                .map(|n| {
                    format!(
                        "\n  {}  {} {}",
                        n.id(),
                        n.kind(),
                        crate::style::truncate(&n.label(), 50)
                    )
                })
                .collect::<String>();
            let more = matches.len().saturating_sub(5);
            bail!(
                "`{reference}` matches {} nodes — be more specific:{listed}{}",
                matches.len(),
                if more > 0 {
                    format!("\n  … and {more} more")
                } else {
                    String::new()
                }
            )
        }
    }
}

/// Parse a comma-separated edge type list, with a message naming the valid
/// options rather than just rejecting the input.
pub fn edge_types(raw: Option<&[String]>) -> Result<Vec<nexum_core::EdgeType>> {
    let Some(raw) = raw else {
        return Ok(Vec::new());
    };
    raw.iter()
        .flat_map(|s| s.split(','))
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            nexum_core::EdgeType::from_str(s).map_err(|_| {
                anyhow::anyhow!(
                    "unknown edge type `{s}` — valid types are: {}",
                    nexum_core::EdgeType::ALL
                        .iter()
                        .map(|e| e.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
        })
        .collect()
}

/// Parse a traversal direction.
pub fn direction(raw: &str) -> Result<nexum_core::Direction> {
    nexum_core::Direction::from_str(raw)
        .map_err(|_| anyhow::anyhow!("unknown direction `{raw}` — use out, in, or both"))
}
