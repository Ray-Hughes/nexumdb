//! Terminal styling.
//!
//! Colour is suppressed when output is piped, when `NO_COLOR` is set, or when
//! `TERM=dumb` — a CLI whose `--json` output contains escape codes is not
//! scriptable, which is the whole point of having `--json`.

use std::io::IsTerminal;
use std::sync::OnceLock;

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        if std::env::var_os("NO_COLOR").is_some() {
            return false;
        }
        if std::env::var("TERM").is_ok_and(|t| t == "dumb") {
            return false;
        }
        std::io::stdout().is_terminal()
    })
}

fn wrap(code: &str, text: &str) -> String {
    if enabled() {
        format!("\x1b[{code}m{text}\x1b[0m")
    } else {
        text.to_string()
    }
}

pub fn bold(text: &str) -> String {
    wrap("1", text)
}

pub fn dim(text: &str) -> String {
    wrap("2", text)
}

pub fn cyan(text: &str) -> String {
    wrap("36", text)
}

pub fn green(text: &str) -> String {
    wrap("32", text)
}

pub fn yellow(text: &str) -> String {
    wrap("33", text)
}

pub fn red(text: &str) -> String {
    wrap("31", text)
}

pub fn magenta(text: &str) -> String {
    wrap("35", text)
}

/// Colour a node kind consistently everywhere it appears.
pub fn kind(kind: nexum_core::NodeKind) -> String {
    use nexum_core::NodeKind::*;
    match kind {
        Document => cyan("Document"),
        Chunk => green("Chunk"),
        Entity => magenta("Entity"),
        PipelineRun => yellow("PipelineRun"),
    }
}

/// Colour an edge type by family, so structure, meaning, and lineage are
/// distinguishable at a glance.
pub fn edge(edge_type: nexum_core::EdgeType) -> String {
    use nexum_core::EdgeClass::*;
    match edge_type.class() {
        Structural => cyan(edge_type.as_str()),
        Semantic => magenta(edge_type.as_str()),
        Provenance => yellow(edge_type.as_str()),
    }
}

/// Render a byte count in units a human reads.
pub fn bytes(count: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = count as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{count} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

/// Short, human-quotable form of a node ID.
///
/// Node IDs are UUIDv7, whose leading bytes are a millisecond timestamp — so
/// every node written in the same batch shares a prefix and a leading
/// substring identifies nothing. The entropy lives in the tail, so that is
/// what gets shown.
pub fn short_id(id: nexum_core::NodeId) -> String {
    let text = id.to_string();
    let hex: String = text.chars().filter(|c| *c != '-').collect();
    hex.chars().skip(hex.len().saturating_sub(8)).collect()
}

/// Truncate to `width` display columns, adding an ellipsis when it cuts.
pub fn truncate(text: &str, width: usize) -> String {
    let cleaned: String = text
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    let collapsed = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= width {
        return collapsed;
    }
    let kept: String = collapsed.chars().take(width.saturating_sub(1)).collect();
    format!("{kept}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_counts_are_readable() {
        assert_eq!(bytes(0), "0 B");
        assert_eq!(bytes(512), "512 B");
        assert_eq!(bytes(2048), "2.0 KB");
        assert_eq!(bytes(5 * 1024 * 1024), "5.0 MB");
    }

    #[test]
    fn truncation_collapses_whitespace_and_marks_cuts() {
        assert_eq!(truncate("hello   world", 20), "hello world");
        assert_eq!(truncate("hello world", 8), "hello w…");
        assert_eq!(truncate("exact", 5), "exact");
    }

    #[test]
    fn short_ids_distinguish_nodes_created_together() {
        // UUIDv7 timestamps make leading characters identical for a batch,
        // which is exactly the case a short id has to survive.
        let ids: Vec<nexum_core::NodeId> = (0..64).map(|_| nexum_core::NodeId::new()).collect();
        let shorts: std::collections::HashSet<String> = ids.iter().copied().map(short_id).collect();
        assert_eq!(
            shorts.len(),
            ids.len(),
            "short ids collided within one batch"
        );
        assert!(shorts.iter().all(|s| s.len() == 8));
    }

    #[test]
    fn control_characters_never_reach_the_terminal() {
        let messy = "line\none\ttwo\u{7}three";
        let clean = truncate(messy, 100);
        assert!(!clean.contains('\n') && !clean.contains('\t') && !clean.contains('\u{7}'));
    }
}
