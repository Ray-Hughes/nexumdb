//! Entity canonicalisation.
//!
//! Two layers, because they solve different problems:
//!
//! - **Exact**, via content-addressed IDs. An entity's node ID is derived from
//!   its normalised name and type, so the same entity mentioned in fifty
//!   documents lands on one node without any merge pass at all. This is the
//!   layer that does the real work.
//! - **Fuzzy**, via string similarity, for the residue: "Ada Lovelace" and
//!   "A. Lovelace" normalise differently but mean the same person. These are
//!   linked with `canonical_id` rather than merged, so the alias stays
//!   queryable and a wrong merge stays reversible.

use crate::extract::normalize_name;
use nexum_core::NodeId;
use serde::{Deserialize, Serialize};

/// How aggressively to fold similar entities together.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DedupConfig {
    /// Run the fuzzy pass at all.
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    /// Similarity at or above which two names are treated as the same entity.
    #[serde(default = "default_threshold")]
    pub threshold: f32,
    /// Allow folding entities that were assigned different types. Off by
    /// default: a person and an organisation sharing a name are usually two
    /// things, not one.
    #[serde(default)]
    pub across_types: bool,
}

fn default_enabled() -> bool {
    true
}
fn default_threshold() -> f32 {
    0.92
}

impl Default for DedupConfig {
    fn default() -> Self {
        DedupConfig {
            enabled: default_enabled(),
            threshold: default_threshold(),
            across_types: false,
        }
    }
}

/// The stable node ID for an entity.
///
/// Deriving the ID from the content is what makes exact dedup free: two
/// ingests of the same name produce the same ID, so the second is an update
/// rather than a duplicate.
pub fn canonical_entity_id(name: &str, entity_type: &str) -> NodeId {
    let key = format!("{}|{}", normalize_name(name), entity_type.to_lowercase());
    NodeId::derived("entity", key.as_bytes())
}

/// One entity folded into another.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Alias {
    pub alias_id: NodeId,
    pub alias_name: String,
    pub canonical_id: NodeId,
    pub canonical_name: String,
    pub similarity: f32,
}

/// An entity considered by the dedup pass.
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub id: NodeId,
    pub name: String,
    pub entity_type: String,
    /// Total mentions. The most-mentioned form wins when two names merge,
    /// which keeps the canonical node the one users actually recognise.
    pub mentions: u32,
}

/// Find alias relationships among candidates.
///
/// The most-mentioned entity in each cluster becomes canonical; ties break on
/// name length and then on ID, so the outcome does not depend on input order.
pub fn find_aliases(candidates: &[Candidate], config: DedupConfig) -> Vec<Alias> {
    if !config.enabled || candidates.len() < 2 {
        return Vec::new();
    }

    // Strongest candidates first, so they become cluster heads.
    let mut ordered: Vec<&Candidate> = candidates.iter().collect();
    ordered.sort_by(|a, b| {
        b.mentions
            .cmp(&a.mentions)
            .then_with(|| b.name.len().cmp(&a.name.len()))
            .then_with(|| a.id.cmp(&b.id))
    });

    let mut aliases = Vec::new();
    let mut heads: Vec<&Candidate> = Vec::new();

    for candidate in ordered {
        let matched = heads.iter().find(|head| {
            if !config.across_types && head.entity_type != candidate.entity_type {
                return false;
            }
            similarity(&head.name, &candidate.name) >= config.threshold
        });

        match matched {
            Some(head) => aliases.push(Alias {
                alias_id: candidate.id,
                alias_name: candidate.name.clone(),
                canonical_id: head.id,
                canonical_name: head.name.clone(),
                similarity: similarity(&head.name, &candidate.name),
            }),
            None => heads.push(candidate),
        }
    }
    aliases
}

/// Similarity between two entity names, in 0..=1.
pub fn similarity(a: &str, b: &str) -> f32 {
    let a = normalize_name(a);
    let b = normalize_name(b);
    if a == b {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    // One name's tokens fully contained in the other's: "Lovelace" inside
    // "Ada Lovelace". Strong, but not certain — plenty of people share a
    // surname — so it lands just under an exact match.
    let a_tokens: Vec<&str> = a.split_whitespace().collect();
    let b_tokens: Vec<&str> = b.split_whitespace().collect();
    let (shorter, longer) = if a_tokens.len() <= b_tokens.len() {
        (&a_tokens, &b_tokens)
    } else {
        (&b_tokens, &a_tokens)
    };
    if shorter.iter().all(|t| longer.contains(t)) {
        return 0.95;
    }

    // Initials against full names: "a lovelace" vs "ada lovelace".
    if shorter.len() == longer.len()
        && shorter.iter().zip(longer.iter()).all(|(s, l)| {
            s == l || (s.len() == 1 && l.starts_with(s)) || (l.len() == 1 && s.starts_with(l))
        })
    {
        return 0.93;
    }

    jaro_winkler(&a, &b)
}

/// Jaro-Winkler similarity.
///
/// Chosen over plain edit distance because it weights a shared prefix, which
/// is exactly the signal in names: "Lovelace" and "Lovelacce" are the same
/// person, "Lovelace" and "Babbage" are not.
pub fn jaro_winkler(a: &str, b: &str) -> f32 {
    let jaro = jaro(a, b);
    if jaro < 0.7 {
        return jaro;
    }
    let prefix = a
        .chars()
        .zip(b.chars())
        .take(4)
        .take_while(|(x, y)| x == y)
        .count() as f32;
    jaro + prefix * 0.1 * (1.0 - jaro)
}

fn jaro(a: &str, b: &str) -> f32 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    if a == b {
        return 1.0;
    }

    // Characters can only match within this window of each other.
    let window = (a.len().max(b.len()) / 2).saturating_sub(1);
    let mut a_matched = vec![false; a.len()];
    let mut b_matched = vec![false; b.len()];
    let mut matches = 0usize;

    for (i, ch) in a.iter().enumerate() {
        let start = i.saturating_sub(window);
        let end = (i + window + 1).min(b.len());
        for j in start..end {
            if b_matched[j] || b[j] != *ch {
                continue;
            }
            a_matched[i] = true;
            b_matched[j] = true;
            matches += 1;
            break;
        }
    }
    if matches == 0 {
        return 0.0;
    }

    // Count matched characters that appear in a different order.
    let mut transpositions = 0usize;
    let mut k = 0usize;
    for i in 0..a.len() {
        if !a_matched[i] {
            continue;
        }
        while !b_matched[k] {
            k += 1;
        }
        if a[i] != b[k] {
            transpositions += 1;
        }
        k += 1;
    }

    let m = matches as f32;
    (m / a.len() as f32 + m / b.len() as f32 + (m - transpositions as f32 / 2.0) / m) / 3.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(name: &str, entity_type: &str, mentions: u32) -> Candidate {
        Candidate {
            id: canonical_entity_id(name, entity_type),
            name: name.to_string(),
            entity_type: entity_type.to_string(),
            mentions,
        }
    }

    #[test]
    fn derived_ids_dedupe_exact_matches_for_free() {
        assert_eq!(
            canonical_entity_id("Ada Lovelace", "person"),
            canonical_entity_id("ada lovelace", "person")
        );
        assert_eq!(
            canonical_entity_id("The Acme Corp.", "org"),
            canonical_entity_id("acme corp", "org")
        );
    }

    #[test]
    fn different_types_get_different_ids() {
        assert_ne!(
            canonical_entity_id("Mercury", "person"),
            canonical_entity_id("Mercury", "place")
        );
    }

    #[test]
    fn different_names_get_different_ids() {
        assert_ne!(
            canonical_entity_id("Ada Lovelace", "person"),
            canonical_entity_id("Alan Turing", "person")
        );
    }

    #[test]
    fn identical_names_score_one() {
        assert_eq!(similarity("Ada Lovelace", "ada lovelace"), 1.0);
    }

    #[test]
    fn unrelated_names_score_low() {
        assert!(similarity("Ada Lovelace", "Charles Babbage") < 0.7);
        assert!(similarity("Acme Corp", "Globex Industries") < 0.7);
    }

    #[test]
    fn a_surname_matches_the_full_name() {
        assert!(similarity("Lovelace", "Ada Lovelace") > 0.9);
    }

    #[test]
    fn initials_match_full_first_names() {
        assert!(similarity("A. Lovelace", "Ada Lovelace") > 0.9);
        assert!(similarity("J. R. Tolkien", "John R. Tolkien") > 0.9);
    }

    #[test]
    fn typos_score_high_but_distinct_names_do_not() {
        assert!(similarity("Lovelace", "Lovelacce") > 0.9);
        // Sharing a prefix is not enough on its own.
        assert!(similarity("Microsoft", "Micron") < 0.92);
    }

    #[test]
    fn jaro_winkler_matches_known_values() {
        // Classic reference pairs.
        assert!((jaro_winkler("martha", "marhta") - 0.961).abs() < 0.01);
        assert!((jaro_winkler("dixon", "dicksonx") - 0.813).abs() < 0.01);
        assert_eq!(jaro_winkler("abc", "abc"), 1.0);
        assert_eq!(jaro("", "x"), 0.0);
    }

    #[test]
    fn the_most_mentioned_form_becomes_canonical() {
        let candidates = vec![
            candidate("Lovelace", "person", 2),
            candidate("Ada Lovelace", "person", 9),
        ];
        let aliases = find_aliases(&candidates, DedupConfig::default());
        assert_eq!(aliases.len(), 1);
        assert_eq!(aliases[0].alias_name, "Lovelace");
        assert_eq!(aliases[0].canonical_name, "Ada Lovelace");
    }

    #[test]
    fn dedup_is_independent_of_input_order() {
        let a = candidate("Ada Lovelace", "person", 9);
        let b = candidate("Lovelace", "person", 2);
        let forward = find_aliases(&[a.clone(), b.clone()], DedupConfig::default());
        let backward = find_aliases(&[b, a], DedupConfig::default());
        assert_eq!(forward, backward);
    }

    #[test]
    fn unrelated_entities_are_left_alone() {
        let candidates = vec![
            candidate("Ada Lovelace", "person", 5),
            candidate("Charles Babbage", "person", 4),
            candidate("Acme Corp", "org", 3),
        ];
        assert!(find_aliases(&candidates, DedupConfig::default()).is_empty());
    }

    #[test]
    fn types_are_not_crossed_by_default() {
        let candidates = vec![
            candidate("Mercury", "person", 5),
            candidate("Mercury", "place", 3),
        ];
        assert!(find_aliases(&candidates, DedupConfig::default()).is_empty());

        let across = DedupConfig {
            across_types: true,
            ..Default::default()
        };
        assert_eq!(find_aliases(&candidates, across).len(), 1);
    }

    #[test]
    fn dedup_can_be_switched_off() {
        let candidates = vec![
            candidate("Lovelace", "person", 2),
            candidate("Ada Lovelace", "person", 9),
        ];
        let off = DedupConfig {
            enabled: false,
            ..Default::default()
        };
        assert!(find_aliases(&candidates, off).is_empty());
    }

    #[test]
    fn a_stricter_threshold_folds_less() {
        let candidates = vec![
            candidate("Lovelace", "person", 2),
            candidate("Ada Lovelace", "person", 9),
        ];
        let strict = DedupConfig {
            threshold: 0.99,
            ..Default::default()
        };
        assert!(find_aliases(&candidates, strict).is_empty());
    }

    #[test]
    fn a_cluster_folds_into_one_head_not_a_chain() {
        let candidates = vec![
            candidate("Ada Lovelace", "person", 10),
            candidate("Lovelace", "person", 5),
            candidate("A. Lovelace", "person", 2),
        ];
        let aliases = find_aliases(&candidates, DedupConfig::default());
        assert_eq!(aliases.len(), 2);
        // Both must point at the same canonical entity, not at each other.
        assert!(aliases.iter().all(|a| a.canonical_name == "Ada Lovelace"));
    }

    #[test]
    fn empty_and_single_inputs_are_handled() {
        assert!(find_aliases(&[], DedupConfig::default()).is_empty());
        assert!(find_aliases(&[candidate("Solo", "person", 1)], DedupConfig::default()).is_empty());
    }
}
