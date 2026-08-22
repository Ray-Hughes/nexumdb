//! Reading documents off disk.
//!
//! Deliberately narrow: this reads text. Binary formats each need their own
//! extractor, and guessing at one produces plausible-looking garbage that then
//! gets embedded and served as evidence — so unreadable files are reported as
//! skipped rather than silently mangled.

use crate::{IngestError, Result};
use nexum_core::ContentHash;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// A document ready to be ingested.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SourceDocument {
    /// Stable identity across re-ingests — this is the versioning key.
    pub source_uri: String,
    pub title: String,
    pub text: String,
    pub content_hash: ContentHash,
    /// Original path, when it came from the filesystem.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
}

impl SourceDocument {
    /// Build from raw text.
    pub fn from_text(
        source_uri: impl Into<String>,
        title: impl Into<String>,
        text: String,
    ) -> Self {
        let content_hash = ContentHash::of(text.as_bytes());
        SourceDocument {
            source_uri: source_uri.into(),
            title: title.into(),
            text,
            content_hash,
            path: None,
        }
    }

    /// Read a file.
    pub fn from_path(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        if looks_binary(&bytes) {
            return Err(IngestError::Unsupported {
                path: path.display().to_string(),
                reason: "file appears to be binary".into(),
            });
        }
        let text = String::from_utf8(bytes).map_err(|_| IngestError::Unsupported {
            path: path.display().to_string(),
            reason: "file is not valid UTF-8".into(),
        })?;

        let text = if is_html(path) {
            strip_html(&text)
        } else {
            text
        };
        let content_hash = ContentHash::of(text.as_bytes());

        Ok(SourceDocument {
            source_uri: path_to_uri(path),
            title: title_for(path, &text),
            text,
            content_hash,
            path: Some(path.to_path_buf()),
        })
    }
}

/// Extensions treated as text.
///
/// An allowlist rather than a denylist: a new binary format showing up should
/// be skipped by default, not ingested as mojibake.
pub const TEXT_EXTENSIONS: &[&str] = &[
    "txt",
    "md",
    "markdown",
    "mdx",
    "rst",
    "org",
    "adoc",
    "asciidoc",
    "text",
    "log",
    "json",
    "jsonl",
    "ndjson",
    "csv",
    "tsv",
    "yaml",
    "yml",
    "toml",
    "ini",
    "cfg",
    "conf",
    "xml",
    "html",
    "htm",
    "rs",
    "py",
    "js",
    "jsx",
    "ts",
    "tsx",
    "go",
    "java",
    "kt",
    "swift",
    "c",
    "h",
    "cpp",
    "hpp",
    "cs",
    "rb",
    "php",
    "sh",
    "bash",
    "zsh",
    "sql",
    "graphql",
    "proto",
    "tf",
    "dockerfile",
];

/// Directories never worth walking into.
const SKIP_DIRS: &[&str] = &[
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "target",
    "dist",
    "build",
    ".venv",
    "venv",
    "__pycache__",
    ".next",
    ".nuxt",
    ".cache",
    ".idea",
    ".vscode",
    "vendor",
];

/// Whether a path looks ingestible.
pub fn is_supported(path: &Path) -> bool {
    // Extensionless files with well-known names are still text.
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default()
        .to_lowercase();
    if matches!(
        name.as_str(),
        "readme" | "license" | "licence" | "changelog" | "makefile" | "dockerfile" | "notice"
    ) {
        return true;
    }
    path.extension()
        .and_then(|e| e.to_str())
        .map(str::to_lowercase)
        .is_some_and(|ext| TEXT_EXTENSIONS.contains(&ext.as_str()))
}

/// Find every ingestible file under a path.
///
/// Results are sorted so that ingesting the same tree twice produces the same
/// order, which keeps chunk indices and IDs stable across runs.
pub fn discover(root: &Path, recursive: bool) -> Result<Vec<PathBuf>> {
    if root.is_file() {
        return Ok(vec![root.to_path_buf()]);
    }
    if !root.exists() {
        return Err(IngestError::Config(format!(
            "no such path: {}",
            root.display()
        )));
    }

    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];

    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(e) => {
                // An unreadable subdirectory should not abort the whole walk.
                tracing::warn!(path = %dir.display(), error = %e, "skipping unreadable directory");
                continue;
            }
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with('.') && path.is_dir() {
                continue;
            }
            if path.is_dir() {
                if recursive && !SKIP_DIRS.contains(&name.as_str()) {
                    stack.push(path);
                }
            } else if is_supported(&path) {
                found.push(path);
            }
        }
    }

    found.sort();
    Ok(found)
}

/// `file://` URI for a path, absolute where possible.
pub fn path_to_uri(path: &Path) -> String {
    let absolute = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    // Windows paths use backslashes, which are not URI separators.
    let normalized = absolute.to_string_lossy().replace('\\', "/");
    if normalized.starts_with('/') {
        format!("file://{normalized}")
    } else {
        format!("file:///{normalized}")
    }
}

/// A human title: the first markdown heading, else the filename.
fn title_for(path: &Path, text: &str) -> String {
    for line in text.lines().take(20) {
        let trimmed = line.trim();
        if let Some(heading) = trimmed.strip_prefix("# ") {
            let heading = heading.trim();
            if !heading.is_empty() {
                return heading.to_string();
            }
        }
    }
    path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("untitled")
        .to_string()
}

fn is_html(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("html") || e.eq_ignore_ascii_case("htm"))
}

/// A file is binary if it contains a NUL in its first few kilobytes.
///
/// Crude, and the same test `grep` uses. It costs nothing and catches the
/// cases that matter: images, archives, compiled objects.
fn looks_binary(bytes: &[u8]) -> bool {
    bytes.iter().take(8_192).any(|b| *b == 0)
}

/// Strip tags, scripts, and styles from HTML, leaving readable text.
pub fn strip_html(html: &str) -> String {
    let mut out = String::with_capacity(html.len() / 2);
    let mut chars = html.chars().peekable();
    let mut in_tag = false;
    let mut skip_until: Option<&str> = None;
    let mut tag = String::new();

    while let Some(c) = chars.next() {
        match c {
            '<' => {
                in_tag = true;
                tag.clear();
            }
            '>' if in_tag => {
                in_tag = false;
                let name = tag.trim().to_lowercase();
                // Script and style bodies are code, not prose; dropping the
                // tags alone would leave their contents in the text.
                if name.starts_with("script") {
                    skip_until = Some("script");
                } else if name.starts_with("style") {
                    skip_until = Some("style");
                } else if let Some(closing) = skip_until
                    && name == format!("/{closing}")
                {
                    skip_until = None;
                } else if matches!(
                    name.trim_start_matches('/'),
                    "p" | "br" | "div" | "li" | "tr" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6"
                ) {
                    out.push('\n');
                }
            }
            _ if in_tag => tag.push(c),
            _ if skip_until.is_some() => {}
            _ => out.push(c),
        }
        // Consume nothing else; the peek is only to keep the borrow checker
        // from complaining about an unused peekable.
        let _ = chars.peek();
    }

    decode_entities(&out)
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn decode_entities(text: &str) -> String {
    text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write(dir: &Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn reads_a_text_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(dir.path(), "notes.txt", "hello world");
        let doc = SourceDocument::from_path(&path).unwrap();
        assert_eq!(doc.text, "hello world");
        assert_eq!(doc.title, "notes.txt");
        assert!(doc.source_uri.starts_with("file://"));
        assert_eq!(doc.content_hash, ContentHash::of(b"hello world"));
    }

    #[test]
    fn a_markdown_heading_becomes_the_title() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(dir.path(), "doc.md", "# Annual Report\n\nBody text.");
        assert_eq!(
            SourceDocument::from_path(&path).unwrap().title,
            "Annual Report"
        );
    }

    #[test]
    fn binary_files_are_refused_not_mangled() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("image.txt");
        fs::write(&path, [0x89, 0x50, 0x00, 0x01, 0x02]).unwrap();
        let err = SourceDocument::from_path(&path).unwrap_err().to_string();
        assert!(err.contains("binary"), "got: {err}");
    }

    #[test]
    fn invalid_utf8_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.txt");
        fs::write(&path, [0xff, 0xfe, 0x41]).unwrap();
        assert!(SourceDocument::from_path(&path).is_err());
    }

    #[test]
    fn the_same_content_hashes_the_same() {
        let a = SourceDocument::from_text("file:///a", "A", "identical".into());
        let b = SourceDocument::from_text("file:///b", "B", "identical".into());
        assert_eq!(a.content_hash, b.content_hash);
    }

    #[test]
    fn discovery_finds_supported_files_and_sorts_them() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "b.md", "b");
        write(dir.path(), "a.txt", "a");
        write(dir.path(), "image.png", "not really");
        write(dir.path(), "nested/c.rs", "fn main() {}");

        let flat = discover(dir.path(), false).unwrap();
        let names: Vec<String> = flat
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert_eq!(names, vec!["a.txt", "b.md"], "png excluded, sorted");

        let deep = discover(dir.path(), true).unwrap();
        assert_eq!(deep.len(), 3, "recursive should reach nested/c.rs");
    }

    #[test]
    fn discovery_skips_noise_directories() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "keep.md", "yes");
        write(dir.path(), "node_modules/pkg/index.js", "no");
        write(dir.path(), "target/debug/build.rs", "no");
        write(dir.path(), ".git/config", "no");

        let found = discover(dir.path(), true).unwrap();
        assert_eq!(found.len(), 1, "found {found:?}");
    }

    #[test]
    fn discovery_on_a_single_file_returns_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(dir.path(), "one.txt", "x");
        assert_eq!(discover(&path, true).unwrap(), vec![path]);
    }

    #[test]
    fn a_missing_path_is_an_error() {
        assert!(discover(Path::new("/definitely/not/here"), true).is_err());
    }

    #[test]
    fn extensionless_conventional_files_are_supported() {
        assert!(is_supported(Path::new("README")));
        assert!(is_supported(Path::new("Dockerfile")));
        assert!(is_supported(Path::new("notes.md")));
        assert!(!is_supported(Path::new("photo.jpg")));
        assert!(!is_supported(Path::new("archive.zip")));
    }

    #[test]
    fn html_is_reduced_to_readable_text() {
        let html = r#"<html><head><style>body{color:red}</style>
            <script>alert('hi')</script></head>
            <body><h1>Title</h1><p>First para.</p><p>Second &amp; last.</p></body></html>"#;
        let text = strip_html(html);
        assert!(text.contains("Title"));
        assert!(text.contains("First para."));
        assert!(text.contains("Second & last."));
        assert!(
            !text.contains("alert"),
            "script body must be dropped: {text}"
        );
        assert!(
            !text.contains("color:red"),
            "style body must be dropped: {text}"
        );
        assert!(!text.contains('<'));
    }

    #[test]
    fn block_tags_become_line_breaks() {
        let text = strip_html("<p>one</p><p>two</p>");
        assert_eq!(text.lines().count(), 2, "got {text:?}");
    }

    #[test]
    fn uris_are_absolute_and_use_forward_slashes() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(dir.path(), "x.txt", "x");
        let uri = path_to_uri(&path);
        assert!(uri.starts_with("file:///"), "got {uri}");
        assert!(!uri.contains('\\'));
    }
}
