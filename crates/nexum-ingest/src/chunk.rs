//! Chunking strategies.
//!
//! Chunking decides what retrieval can return, so the strategy and its
//! parameters are recorded on the `PipelineRun` — two chunks embedded by the
//! same model but split by different strategies are not comparable evidence,
//! and without the record there is no way to tell after the fact.
//!
//! Sizes are in characters, not tokens: characters are exact, model-agnostic,
//! and reproducible, whereas a token count depends on which tokenizer you ask.
//! `token_count` on the resulting chunk is an estimate, and labelled as one.

use serde::{Deserialize, Serialize};
use std::str::FromStr;

use crate::IngestError;

/// How a document is split.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "strategy", rename_all = "snake_case")]
pub enum ChunkerConfig {
    /// Hard character windows with overlap. Predictable, and blind to
    /// structure — it will cut mid-word.
    Fixed {
        #[serde(default = "default_size")]
        size: usize,
        #[serde(default = "default_overlap")]
        overlap: usize,
    },
    /// Split on the largest natural boundary that fits: paragraphs, then
    /// lines, then sentences, then words. The sensible default for prose.
    Recursive {
        #[serde(default = "default_size")]
        size: usize,
        #[serde(default = "default_overlap")]
        overlap: usize,
    },
    /// Pack whole sentences up to a budget, overlapping by whole sentences.
    /// Never cuts mid-sentence, at the cost of uneven chunk sizes.
    Sentence {
        #[serde(default = "default_size")]
        max_size: usize,
        #[serde(default = "default_sentence_overlap")]
        overlap_sentences: usize,
    },
    /// One chunk per document. For short documents where splitting only
    /// destroys context.
    Whole,
}

fn default_size() -> usize {
    1_000
}
fn default_overlap() -> usize {
    150
}
fn default_sentence_overlap() -> usize {
    1
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        ChunkerConfig::Recursive {
            size: default_size(),
            overlap: default_overlap(),
        }
    }
}

impl ChunkerConfig {
    /// Parse a CLI spec: `recursive`, `fixed:800`, `fixed:800:100`,
    /// `sentence:1200`, `whole`.
    pub fn parse(spec: &str) -> Result<Self, IngestError> {
        let mut parts = spec.split(':');
        let kind = parts.next().unwrap_or_default().to_ascii_lowercase();
        let number = |part: Option<&str>, what: &str| -> Result<Option<usize>, IngestError> {
            part.map(|p| {
                p.parse::<usize>()
                    .map_err(|_| IngestError::Config(format!("`{p}` is not a valid {what}")))
            })
            .transpose()
        };
        let first = number(parts.next(), "size")?;
        let second = number(parts.next(), "overlap")?;

        let config = match kind.as_str() {
            "fixed" => ChunkerConfig::Fixed {
                size: first.unwrap_or_else(default_size),
                overlap: second.unwrap_or_else(default_overlap),
            },
            "recursive" | "" => ChunkerConfig::Recursive {
                size: first.unwrap_or_else(default_size),
                overlap: second.unwrap_or_else(default_overlap),
            },
            "sentence" => ChunkerConfig::Sentence {
                max_size: first.unwrap_or_else(default_size),
                overlap_sentences: second.unwrap_or_else(default_sentence_overlap),
            },
            "whole" | "none" => ChunkerConfig::Whole,
            other => {
                return Err(IngestError::Config(format!(
                    "unknown chunker `{other}` (expected fixed, recursive, sentence, or whole)"
                )));
            }
        };
        config.validate()?;
        Ok(config)
    }

    /// Reject parameter combinations that cannot terminate or cannot help.
    pub fn validate(&self) -> Result<(), IngestError> {
        match self {
            ChunkerConfig::Fixed { size, overlap } | ChunkerConfig::Recursive { size, overlap } => {
                if *size == 0 {
                    return Err(IngestError::Config("chunk size must be at least 1".into()));
                }
                // Overlap >= size means each window starts where the last one
                // did, and chunking never advances.
                if overlap >= size {
                    return Err(IngestError::Config(format!(
                        "overlap ({overlap}) must be smaller than chunk size ({size})"
                    )));
                }
            }
            ChunkerConfig::Sentence { max_size, .. } => {
                if *max_size == 0 {
                    return Err(IngestError::Config(
                        "max chunk size must be at least 1".into(),
                    ));
                }
            }
            ChunkerConfig::Whole => {}
        }
        Ok(())
    }

    /// Stable name recorded on the `PipelineRun`.
    pub fn name(&self) -> String {
        match self {
            ChunkerConfig::Fixed { size, overlap } => format!("fixed:{size}:{overlap}"),
            ChunkerConfig::Recursive { size, overlap } => format!("recursive:{size}:{overlap}"),
            ChunkerConfig::Sentence {
                max_size,
                overlap_sentences,
            } => format!("sentence:{max_size}:{overlap_sentences}"),
            ChunkerConfig::Whole => "whole".to_string(),
        }
    }

    /// Split text into chunks.
    pub fn split(&self, text: &str) -> Vec<TextChunk> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Vec::new();
        }
        let pieces = match self {
            ChunkerConfig::Whole => vec![trimmed.to_string()],
            ChunkerConfig::Fixed { size, overlap } => fixed(trimmed, *size, *overlap),
            ChunkerConfig::Recursive { size, overlap } => recursive(trimmed, *size, *overlap),
            ChunkerConfig::Sentence {
                max_size,
                overlap_sentences,
            } => by_sentence(trimmed, *max_size, *overlap_sentences),
        };

        pieces
            .into_iter()
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
            .enumerate()
            .map(|(index, text)| TextChunk {
                estimated_tokens: estimate_tokens(&text),
                index: index as u32,
                text,
            })
            .collect()
    }
}

impl FromStr for ChunkerConfig {
    type Err = IngestError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        ChunkerConfig::parse(s)
    }
}

/// One split piece of a document.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextChunk {
    pub index: u32,
    pub text: String,
    /// Approximate token count. Real tokenization depends on the model, so
    /// this is a portable estimate for display and budgeting, not a promise.
    pub estimated_tokens: u32,
}

/// Rough token count: English averages ~4 characters per token, with a
/// per-word floor so that many short words are not under-counted.
pub fn estimate_tokens(text: &str) -> u32 {
    let chars = text.chars().count();
    let words = text.split_whitespace().count();
    (chars / 4).max(words) as u32
}

/// Character-window splitting, respecting char boundaries.
fn fixed(text: &str, size: usize, overlap: usize) -> Vec<String> {
    // Index by chars, not bytes: slicing a UTF-8 string at an arbitrary byte
    // offset panics, and documents are not all ASCII.
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= size {
        return vec![text.to_string()];
    }
    let step = size.saturating_sub(overlap).max(1);
    let mut out = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let end = (start + size).min(chars.len());
        out.push(chars[start..end].iter().collect());
        if end == chars.len() {
            break;
        }
        start += step;
    }
    out
}

/// Split on the largest separator that produces pieces within budget.
fn recursive(text: &str, size: usize, overlap: usize) -> Vec<String> {
    const SEPARATORS: &[&str] = &["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "];
    let pieces = split_recursive(text, size, SEPARATORS);
    merge_with_overlap(pieces, size, overlap)
}

fn split_recursive(text: &str, size: usize, separators: &[&str]) -> Vec<String> {
    if text.chars().count() <= size {
        return vec![text.to_string()];
    }
    let Some((separator, rest)) = separators.split_first() else {
        // Out of separators: fall back to hard windows so an unbroken run of
        // characters still gets split rather than returned oversized.
        return fixed(text, size, 0);
    };

    let parts: Vec<&str> = text.split(separator).collect();
    if parts.len() == 1 {
        return split_recursive(text, size, rest);
    }

    let mut out = Vec::new();
    for (i, part) in parts.iter().enumerate() {
        // Put the separator back, so reassembly is lossless.
        let piece = if i + 1 < parts.len() {
            format!("{part}{separator}")
        } else {
            (*part).to_string()
        };
        if piece.trim().is_empty() {
            continue;
        }
        if piece.chars().count() > size {
            out.extend(split_recursive(&piece, size, rest));
        } else {
            out.push(piece);
        }
    }
    out
}

/// Greedily pack pieces up to `size`, carrying `overlap` characters forward.
fn merge_with_overlap(pieces: Vec<String>, size: usize, overlap: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();

    for piece in pieces {
        if !current.is_empty() && current.chars().count() + piece.chars().count() > size {
            let tail = tail_chars(&current, overlap);
            out.push(std::mem::take(&mut current));
            current.push_str(&tail);
        }
        current.push_str(&piece);
    }
    if !current.trim().is_empty() {
        out.push(current);
    }
    out
}

/// The last `n` characters, starting at a word boundary where one is close by,
/// so overlap does not begin mid-word.
fn tail_chars(text: &str, n: usize) -> String {
    if n == 0 {
        return String::new();
    }
    let chars: Vec<char> = text.chars().collect();
    let start = chars.len().saturating_sub(n);
    let tail: String = chars[start..].iter().collect();
    match tail.find(char::is_whitespace) {
        Some(offset) if offset < n / 2 => tail[offset..].trim_start().to_string(),
        _ => tail,
    }
}

/// Pack whole sentences up to a budget.
fn by_sentence(text: &str, max_size: usize, overlap_sentences: usize) -> Vec<String> {
    let sentences = split_sentences(text);
    if sentences.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let mut current_len = 0usize;

    for sentence in sentences {
        let len = sentence.chars().count();
        if !current.is_empty() && current_len + len > max_size {
            out.push(current.join(" "));
            // Carry the trailing sentences forward so a chunk boundary does
            // not sever the context a following sentence depends on.
            let keep = overlap_sentences.min(current.len().saturating_sub(1));
            current = current.split_off(current.len() - keep);
            current_len = current.iter().map(|s| s.chars().count()).sum();
        }
        current_len += len;
        current.push(sentence);
    }
    if !current.is_empty() {
        out.push(current.join(" "));
    }
    out
}

/// Abbreviations that end in a period without ending a sentence.
///
/// Without these, "Dr. Hopper" and "Acme Corp. grew" get cut in half, which
/// breaks sentence chunking and severs entity names mid-phrase.
const ABBREVIATIONS: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "rev", "hon", "gen", "col", "lt", "sgt",
    "capt", "inc", "corp", "ltd", "co", "llc", "plc", "dept", "univ", "assn", "bros", "etc", "eg",
    "ie", "vs", "approx", "est", "fig", "vol", "no", "pp", "ed", "al", "jan", "feb", "mar", "apr",
    "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "mon", "tue", "wed", "thu", "fri",
    "sat", "sun", "am", "pm", "usa", "uk", "eu",
];

/// Whether a period following `word` genuinely ends a sentence.
fn is_sentence_end(word: &str, next: Option<char>) -> bool {
    // A following lowercase letter or digit is strong evidence the period was
    // an abbreviation or a decimal point, not a full stop.
    if next.is_some_and(|c| c.is_lowercase() || c.is_ascii_digit()) {
        return false;
    }
    let bare = word
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase();
    if ABBREVIATIONS.contains(&bare.as_str()) {
        return false;
    }
    // A single letter is an initial: "J. R. R. Tolkien".
    if bare.chars().count() == 1 && bare.chars().all(char::is_alphabetic) {
        return false;
    }
    true
}

/// Split into sentences on terminal punctuation followed by whitespace.
///
/// Handles the common abbreviation and initial cases, which matters because
/// both the sentence chunker and entity extraction build on this. It is still
/// a heuristic, not a trained segmenter.
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        current.push(c);
        if !matches!(c, '.' | '!' | '?') {
            continue;
        }
        if !chars.peek().is_some_and(|next| next.is_whitespace()) {
            continue;
        }
        if current.trim().is_empty() {
            continue;
        }

        // Look past the whitespace to see what follows.
        let following = {
            let mut lookahead = chars.clone();
            loop {
                match lookahead.next() {
                    Some(c) if c.is_whitespace() => continue,
                    other => break other,
                }
            }
        };

        // Question and exclamation marks are unambiguous; only periods need
        // the abbreviation check.
        let last_word = current
            .trim_end_matches('.')
            .split_whitespace()
            .next_back()
            .unwrap_or_default()
            .to_string();
        if c != '.' || is_sentence_end(&last_word, following) {
            out.push(current.trim().to_string());
            current.clear();
        }
    }
    if !current.trim().is_empty() {
        out.push(current.trim().to_string());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_of(chunks: &[TextChunk]) -> Vec<&str> {
        chunks.iter().map(|c| c.text.as_str()).collect()
    }

    #[test]
    fn empty_input_yields_no_chunks() {
        for config in [
            ChunkerConfig::Whole,
            ChunkerConfig::Fixed {
                size: 10,
                overlap: 2,
            },
            ChunkerConfig::Recursive {
                size: 10,
                overlap: 2,
            },
            ChunkerConfig::Sentence {
                max_size: 10,
                overlap_sentences: 1,
            },
        ] {
            assert!(config.split("").is_empty(), "{config:?}");
            assert!(config.split("   \n  ").is_empty(), "{config:?}");
        }
    }

    #[test]
    fn whole_returns_one_chunk() {
        let chunks = ChunkerConfig::Whole.split("anything at all");
        assert_eq!(text_of(&chunks), vec!["anything at all"]);
        assert_eq!(chunks[0].index, 0);
    }

    #[test]
    fn fixed_chunks_respect_size_and_overlap() {
        let text: String = ('a'..='z').collect();
        let chunks = ChunkerConfig::Fixed {
            size: 10,
            overlap: 3,
        }
        .split(&text);
        for chunk in &chunks {
            assert!(chunk.text.chars().count() <= 10, "{:?}", chunk.text);
        }
        // Step is size - overlap = 7.
        assert_eq!(chunks[0].text, "abcdefghij");
        assert_eq!(chunks[1].text, "hijklmnopq");
    }

    #[test]
    fn fixed_covers_the_whole_input() {
        let text: String = (0..500)
            .map(|i| char::from(b'a' + (i % 26) as u8))
            .collect();
        let chunks = ChunkerConfig::Fixed {
            size: 64,
            overlap: 8,
        }
        .split(&text);
        let rejoined: String = chunks.iter().map(|c| c.text.as_str()).collect();
        // Every original character must appear somewhere.
        for window in text.as_bytes().chunks(32) {
            let piece = std::str::from_utf8(window).unwrap();
            assert!(rejoined.contains(piece), "lost `{piece}`");
        }
    }

    #[test]
    fn chunk_indices_are_sequential_from_zero() {
        let text = "word ".repeat(500);
        let chunks = ChunkerConfig::Recursive {
            size: 100,
            overlap: 20,
        }
        .split(&text);
        assert!(chunks.len() > 3);
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.index, i as u32);
        }
    }

    #[test]
    fn recursive_prefers_paragraph_boundaries() {
        let text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here.";
        let chunks = ChunkerConfig::Recursive {
            size: 30,
            overlap: 0,
        }
        .split(text);
        // Each paragraph fits in 30 chars, so none should be cut mid-sentence.
        assert!(chunks.iter().all(|c| c.text.contains("paragraph")));
        assert_eq!(chunks.len(), 3);
    }

    #[test]
    fn recursive_splits_text_with_no_separators_at_all() {
        let text = "x".repeat(250);
        let chunks = ChunkerConfig::Recursive {
            size: 100,
            overlap: 0,
        }
        .split(&text);
        assert!(chunks.len() >= 3);
        assert!(chunks.iter().all(|c| c.text.chars().count() <= 100));
    }

    #[test]
    fn recursive_never_returns_oversized_chunks() {
        let text = "Some sentence here. ".repeat(200);
        for size in [50usize, 120, 400] {
            let chunks = ChunkerConfig::Recursive { size, overlap: 10 }.split(&text);
            for chunk in &chunks {
                assert!(
                    chunk.text.chars().count() <= size + 10,
                    "size {size}: chunk of {} chars",
                    chunk.text.chars().count()
                );
            }
        }
    }

    #[test]
    fn sentence_chunking_never_cuts_a_sentence() {
        let text = "One sentence. Two sentence. Three sentence. Four sentence. Five sentence.";
        let chunks = ChunkerConfig::Sentence {
            max_size: 30,
            overlap_sentences: 0,
        }
        .split(text);
        for chunk in &chunks {
            assert!(
                chunk.text.ends_with('.'),
                "chunk should end at a sentence: {:?}",
                chunk.text
            );
        }
    }

    #[test]
    fn sentence_overlap_repeats_the_previous_sentence() {
        let text = "Alpha one. Bravo two. Charlie three. Delta four.";
        let chunks = ChunkerConfig::Sentence {
            max_size: 25,
            overlap_sentences: 1,
        }
        .split(text);
        assert!(chunks.len() >= 2);
        // The tail of chunk N should reappear at the head of chunk N+1.
        for pair in chunks.windows(2) {
            let previous_last = pair[0]
                .text
                .rsplit_once(". ")
                .map_or(pair[0].text.as_str(), |(_, t)| t);
            assert!(
                pair[1].text.starts_with(previous_last.trim()),
                "expected {:?} to start with {:?}",
                pair[1].text,
                previous_last
            );
        }
    }

    #[test]
    fn multibyte_text_is_never_split_mid_character() {
        let text = "日本語のテキストです。".repeat(40);
        for config in [
            ChunkerConfig::Fixed {
                size: 17,
                overlap: 5,
            },
            ChunkerConfig::Recursive {
                size: 17,
                overlap: 5,
            },
        ] {
            let chunks = config.split(&text);
            assert!(!chunks.is_empty());
            for chunk in &chunks {
                // Reaching here at all means no panic; confirm it is valid text.
                assert!(chunk.text.chars().all(|c| c != '\u{FFFD}'), "{config:?}");
            }
        }
    }

    #[test]
    fn emoji_survive_chunking() {
        let text = "🎉 party ".repeat(60);
        let chunks = ChunkerConfig::Fixed {
            size: 20,
            overlap: 4,
        }
        .split(&text);
        let rejoined: String = chunks.iter().map(|c| c.text.as_str()).collect();
        assert!(rejoined.contains("🎉"));
    }

    #[test]
    fn overlap_at_least_as_large_as_size_is_rejected() {
        // This would make chunking loop forever without advancing.
        assert!(
            ChunkerConfig::Fixed {
                size: 10,
                overlap: 10
            }
            .validate()
            .is_err()
        );
        assert!(
            ChunkerConfig::Fixed {
                size: 10,
                overlap: 99
            }
            .validate()
            .is_err()
        );
        assert!(
            ChunkerConfig::Recursive {
                size: 10,
                overlap: 10
            }
            .validate()
            .is_err()
        );
        assert!(
            ChunkerConfig::Fixed {
                size: 0,
                overlap: 0
            }
            .validate()
            .is_err()
        );
        assert!(
            ChunkerConfig::Fixed {
                size: 10,
                overlap: 9
            }
            .validate()
            .is_ok()
        );
    }

    #[test]
    fn specs_parse_and_round_trip_through_their_name() {
        let cases = [
            (
                "fixed:800:100",
                ChunkerConfig::Fixed {
                    size: 800,
                    overlap: 100,
                },
            ),
            (
                "recursive",
                ChunkerConfig::Recursive {
                    size: 1000,
                    overlap: 150,
                },
            ),
            (
                "sentence:500:2",
                ChunkerConfig::Sentence {
                    max_size: 500,
                    overlap_sentences: 2,
                },
            ),
            ("whole", ChunkerConfig::Whole),
        ];
        for (spec, expected) in cases {
            let parsed = ChunkerConfig::parse(spec).unwrap();
            assert_eq!(parsed, expected, "{spec}");
            assert_eq!(ChunkerConfig::parse(&parsed.name()).unwrap(), expected);
        }
    }

    #[test]
    fn bad_specs_are_rejected_with_guidance() {
        let err = ChunkerConfig::parse("nonsense").unwrap_err().to_string();
        assert!(
            err.contains("fixed, recursive, sentence, or whole"),
            "got: {err}"
        );
        assert!(ChunkerConfig::parse("fixed:abc").is_err());
        assert!(ChunkerConfig::parse("fixed:10:10").is_err());
    }

    #[test]
    fn sentence_splitting_handles_terminal_punctuation() {
        let sentences = split_sentences("Hi there! How are you? I am fine. Really");
        assert_eq!(
            sentences,
            vec!["Hi there!", "How are you?", "I am fine.", "Really"]
        );
    }

    #[test]
    fn token_estimates_are_in_a_sane_range() {
        // ~9 words, 52 chars.
        let text = "the quick brown fox jumps over the lazy dog again";
        let estimate = estimate_tokens(text);
        assert!((9..=20).contains(&estimate), "estimate was {estimate}");
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn every_chunk_carries_a_token_estimate() {
        let chunks = ChunkerConfig::default().split(&"some words here. ".repeat(300));
        assert!(chunks.iter().all(|c| c.estimated_tokens > 0));
    }
}
