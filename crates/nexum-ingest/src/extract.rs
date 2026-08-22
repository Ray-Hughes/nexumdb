//! Entity and relation extraction.
//!
//! The default extractor is rule-based, and deliberately so: it needs no
//! model, no network, and no Python, which means entity edges exist from the
//! first ingest rather than after a separate setup step. It is genuinely
//! weaker than a trained NER model — it recognises *named things* and guesses
//! their type from surface form — so [`EntityExtractor`] is the seam where a
//! spaCy service or an LLM extractor slots in without touching the pipeline.

use crate::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

/// A named thing found in text.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExtractedEntity {
    /// Surface form, as written.
    pub name: String,
    /// person / org / place / concept.
    pub entity_type: String,
    /// How many times it appeared.
    pub mentions: u32,
    /// Confidence in 0..=1. Rule-based extraction is not certain, and
    /// pretending otherwise makes downstream filtering impossible.
    pub confidence: f32,
}

/// A relation between two entities.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExtractedRelation {
    pub from: String,
    pub to: String,
    pub relation_type: String,
    pub confidence: f32,
}

/// Everything one extraction pass found.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Extraction {
    pub entities: Vec<ExtractedEntity>,
    pub relations: Vec<ExtractedRelation>,
}

impl Extraction {
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty() && self.relations.is_empty()
    }
}

/// Pulls entities and relations out of text.
#[async_trait]
pub trait EntityExtractor: Send + Sync {
    /// Recorded on the `PipelineRun`, so extractions are attributable to the
    /// thing that produced them.
    fn name(&self) -> &str;

    async fn extract(&self, text: &str) -> Result<Extraction>;
}

/// Entity types this extractor can assign.
pub mod entity_type {
    pub const PERSON: &str = "person";
    pub const ORG: &str = "org";
    pub const PLACE: &str = "place";
    pub const CONCEPT: &str = "concept";
}

/// Words that are capitalised for grammatical reasons rather than because they
/// name something. Without this every sentence-initial "The" becomes an entity.
const STOPWORDS: &[&str] = &[
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "we",
    "our",
    "you",
    "your",
    "they",
    "their",
    "he",
    "she",
    "his",
    "her",
    "i",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "can",
    "could",
    "should",
    "may",
    "might",
    "must",
    "not",
    "no",
    "yes",
    "there",
    "here",
    "when",
    "where",
    "why",
    "how",
    "what",
    "which",
    "who",
    "whom",
    "all",
    "any",
    "some",
    "each",
    "every",
    "both",
    "more",
    "most",
    "other",
    "such",
    "only",
    "own",
    "same",
    "than",
    "too",
    "very",
    "just",
    "also",
    "however",
    "therefore",
    "because",
    "while",
    "after",
    "before",
    "during",
    "since",
    "until",
    "although",
    "though",
    // Sentence-initial adverbs and connectives. Capitalised by position, not
    // because they name anything.
    "later",
    "meanwhile",
    "finally",
    "first",
    "second",
    "third",
    "next",
    "today",
    "yesterday",
    "tomorrow",
    "now",
    "thus",
    "hence",
    "moreover",
    "furthermore",
    "additionally",
    "subsequently",
    "previously",
    "recently",
    "currently",
    "overall",
    "instead",
    "otherwise",
    "similarly",
    "likewise",
    "indeed",
    "still",
    "yet",
    "once",
    "again",
    "often",
    "always",
    "never",
    "sometimes",
    "usually",
    "generally",
    "specifically",
    "particularly",
    "especially",
    "unlike",
    "despite",
    "besides",
    "accordingly",
    "consequently",
    "nevertheless",
    "nonetheless",
];

/// Suffixes that mark an organisation.
const ORG_SUFFIXES: &[&str] = &[
    "inc",
    "inc.",
    "corp",
    "corp.",
    "corporation",
    "ltd",
    "ltd.",
    "llc",
    "plc",
    "gmbh",
    "company",
    "co",
    "co.",
    "foundation",
    "institute",
    "university",
    "college",
    "school",
    "hospital",
    "laboratory",
    "labs",
    "group",
    "association",
    "society",
    "agency",
    "department",
    "ministry",
    "committee",
    "council",
    "bureau",
    "office",
    "bank",
    "partners",
    "holdings",
    "systems",
    "technologies",
    "solutions",
];

/// Honorifics that mark a person.
const PERSON_PREFIXES: &[&str] = &[
    "mr",
    "mrs",
    "ms",
    "miss",
    "dr",
    "prof",
    "professor",
    "sir",
    "dame",
    "lord",
    "lady",
    "rev",
    "president",
    "senator",
    "governor",
    "judge",
    "captain",
    "general",
    "colonel",
];

/// Words that mark a place when they end a name.
const PLACE_SUFFIXES: &[&str] = &[
    "street",
    "avenue",
    "road",
    "boulevard",
    "lane",
    "city",
    "county",
    "state",
    "province",
    "river",
    "mountain",
    "lake",
    "island",
    "valley",
    "bay",
    "park",
    "beach",
    "desert",
    "ocean",
    "sea",
];

/// Rule-based extractor.
#[derive(Debug, Clone)]
pub struct RuleExtractor {
    /// Ignore entities appearing fewer than this many times in one chunk.
    min_mentions: u32,
    /// Emit `co_occurs_with` relations between entities in the same text.
    relations: bool,
}

impl Default for RuleExtractor {
    fn default() -> Self {
        RuleExtractor {
            min_mentions: 1,
            relations: true,
        }
    }
}

impl RuleExtractor {
    pub fn new(min_mentions: u32, relations: bool) -> Self {
        RuleExtractor {
            min_mentions,
            relations,
        }
    }

    /// Find candidate names: runs of capitalised words, and acronyms.
    ///
    /// Runs are bounded by line breaks as well as sentence ends. A heading
    /// carries no terminal punctuation, so without the line boundary it merges
    /// into the sentence below it and the two run together into one nonsense
    /// entity.
    fn find_candidates(&self, text: &str) -> Vec<String> {
        let mut out = Vec::new();

        for line in text.lines() {
            // Markdown heading, list, and quote markers are punctuation, not
            // part of any name.
            let line = line.trim_start_matches(['#', '-', '*', '>', ' ', '\t']);

            for sentence in crate::chunk::split_sentences(line) {
                let words: Vec<&str> = sentence.split_whitespace().collect();
                let mut run: Vec<String> = Vec::new();

                for (position, raw) in words.iter().enumerate() {
                    let word = raw.trim_matches(|c: char| {
                        !c.is_alphanumeric() && c != '.' && c != '&' && c != '-' && c != '\''
                    });
                    let bare = word.trim_end_matches('.');

                    if bare.is_empty() {
                        flush(&mut run, &mut out);
                        continue;
                    }

                    let capitalised = bare.chars().next().is_some_and(char::is_uppercase);
                    // A capitalised word at the start of a sentence carries no
                    // signal on its own — most of them are ordinary words.
                    let informative = capitalised
                        && !(position == 0 && STOPWORDS.contains(&bare.to_lowercase().as_str()));
                    // Lowercase connectors stay inside a run: "Bank of England".
                    let connector = !run.is_empty()
                        && matches!(
                            bare.to_lowercase().as_str(),
                            "of" | "and" | "for" | "the" | "de" | "van" | "von"
                        );

                    if informative || connector {
                        run.push(word.to_string());
                        // Punctuation after the word ends the name: in
                        // "Later, Dr. Grace Hopper" the comma separates the
                        // adverb from the name that follows it.
                        if raw.ends_with([',', ';', ':', ')', ']', '"']) {
                            flush(&mut run, &mut out);
                        }
                    } else {
                        flush(&mut run, &mut out);
                    }
                }
                flush(&mut run, &mut out);
            }
        }
        out
    }

    /// Guess a type from the surface form.
    ///
    /// This is genuinely uncertain, and the returned confidence says so: a
    /// suffix or honorific is strong evidence, bare capitalisation is not.
    fn classify(name: &str) -> (&'static str, f32) {
        let words: Vec<String> = name.split_whitespace().map(str::to_lowercase).collect();
        let Some(last) = words.last() else {
            return (entity_type::CONCEPT, 0.3);
        };
        let first = words.first().map(String::as_str).unwrap_or_default();
        let first_bare = first.trim_end_matches('.');

        if ORG_SUFFIXES.contains(&last.as_str()) {
            return (entity_type::ORG, 0.9);
        }
        if PLACE_SUFFIXES.contains(&last.as_str()) {
            return (entity_type::PLACE, 0.8);
        }
        if PERSON_PREFIXES.contains(&first_bare) {
            return (entity_type::PERSON, 0.9);
        }
        // An acronym: 2-6 uppercase letters, e.g. NASA, IBM.
        if name.len() >= 2
            && name.len() <= 6
            && name.chars().all(|c| c.is_ascii_uppercase() || c == '&')
        {
            return (entity_type::ORG, 0.7);
        }
        // Two or three capitalised words with no connectives reads like a
        // personal name more often than not, but this is a coin-flip's worth
        // of evidence and the confidence reflects that.
        if (2..=3).contains(&words.len())
            && !words
                .iter()
                .any(|w| matches!(w.as_str(), "of" | "and" | "for" | "the"))
        {
            return (entity_type::PERSON, 0.45);
        }
        (entity_type::CONCEPT, 0.4)
    }
}

fn flush(run: &mut Vec<String>, out: &mut Vec<String>) {
    if run.is_empty() {
        return;
    }
    // A run must not end on a connector: "Bank of" is not a name.
    while run.last().is_some_and(|w| {
        matches!(
            w.to_lowercase().as_str(),
            "of" | "and" | "for" | "the" | "de" | "van" | "von"
        )
    }) {
        run.pop();
    }
    if !run.is_empty() {
        out.push(run.join(" "));
    }
    run.clear();
}

#[async_trait]
impl EntityExtractor for RuleExtractor {
    fn name(&self) -> &str {
        "rule-v1"
    }

    async fn extract(&self, text: &str) -> Result<Extraction> {
        let candidates = self.find_candidates(text);

        // Count by normalised form so "the Acme Corp" and "Acme Corp" merge,
        // but keep the most common surface form as the display name.
        let mut counts: BTreeMap<String, (String, u32)> = BTreeMap::new();
        for candidate in candidates {
            let normalized = normalize_name(&candidate);
            if normalized.is_empty() || STOPWORDS.contains(&normalized.as_str()) {
                continue;
            }
            let entry = counts
                .entry(normalized)
                .or_insert_with(|| (candidate.clone(), 0));
            entry.1 += 1;
            // Prefer the longer surface form: "Ada Lovelace" over "Ada".
            if candidate.len() > entry.0.len() {
                entry.0 = candidate;
            }
        }

        let entities: Vec<ExtractedEntity> = counts
            .into_iter()
            .filter(|(_, (_, count))| *count >= self.min_mentions)
            .map(|(_, (name, mentions))| {
                let (entity_type, confidence) = RuleExtractor::classify(&name);
                ExtractedEntity {
                    name,
                    entity_type: entity_type.to_string(),
                    mentions,
                    confidence,
                }
            })
            .collect();

        let relations = if self.relations && entities.len() > 1 {
            let mut out = Vec::new();
            for (i, a) in entities.iter().enumerate() {
                for b in entities.iter().skip(i + 1) {
                    out.push(ExtractedRelation {
                        from: a.name.clone(),
                        to: b.name.clone(),
                        // Co-occurrence is the honest claim. Anything more
                        // specific needs a model that reads the sentence.
                        relation_type: "co_occurs_with".into(),
                        confidence: 0.3,
                    });
                }
            }
            out
        } else {
            Vec::new()
        };

        Ok(Extraction {
            entities,
            relations,
        })
    }
}

/// Canonical form of a name, used for dedup and for deriving stable IDs.
///
/// Lowercases, strips punctuation and a leading article, and collapses
/// whitespace, so "The Acme Corp." and "acme corp" agree.
pub fn normalize_name(name: &str) -> String {
    let lowered = name.to_lowercase();
    let cleaned: String = lowered
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '&' {
                c
            } else {
                ' '
            }
        })
        .collect();
    let words: Vec<&str> = cleaned.split_whitespace().collect();
    let words = match words.split_first() {
        Some((first, rest)) if *first == "the" && !rest.is_empty() => rest,
        _ => &words[..],
    };
    words.join(" ")
}

/// An extractor that finds nothing, for when entity extraction is switched off.
#[derive(Debug, Clone, Default)]
pub struct NullExtractor;

#[async_trait]
impl EntityExtractor for NullExtractor {
    fn name(&self) -> &str {
        "none"
    }

    async fn extract(&self, _text: &str) -> Result<Extraction> {
        Ok(Extraction::default())
    }
}

/// Names found, as a set — used by tests and by the dedup pass.
pub fn names(extraction: &Extraction) -> HashSet<String> {
    extraction.entities.iter().map(|e| e.name.clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn extract(text: &str) -> Extraction {
        RuleExtractor::default().extract(text).await.unwrap()
    }

    #[tokio::test]
    async fn finds_multi_word_names() {
        let e = extract("Ada Lovelace worked with Charles Babbage on the engine.").await;
        let found = names(&e);
        assert!(found.contains("Ada Lovelace"), "got {found:?}");
        assert!(found.contains("Charles Babbage"), "got {found:?}");
    }

    #[tokio::test]
    async fn a_heading_does_not_merge_into_the_line_below_it() {
        let e = extract("# Ada Lovelace and the Analytical Engine\nAda Lovelace wrote it.").await;
        let found = names(&e);
        assert!(
            !found.iter().any(|n| n.split_whitespace().count() > 6),
            "a heading ran into the next line: {found:?}"
        );
        assert!(found.contains("Ada Lovelace"), "got {found:?}");
        assert!(found.iter().all(|n| !n.starts_with('#')), "got {found:?}");
    }

    #[tokio::test]
    async fn list_markers_are_stripped() {
        let e = extract("- Acme Corp. is listed\n* Globex Industries too").await;
        let found = names(&e);
        assert!(
            found.iter().all(|n| !n.starts_with(['-', '*'])),
            "got {found:?}"
        );
    }

    #[tokio::test]
    async fn sentence_initial_stopwords_are_not_entities() {
        let e = extract("The report was late. However, it arrived. This is fine.").await;
        let found = names(&e);
        for noise in ["The", "However", "This", "It"] {
            assert!(
                !found.contains(noise),
                "{noise} should not be an entity: {found:?}"
            );
        }
    }

    #[tokio::test]
    async fn organisation_suffixes_are_classified_as_orgs() {
        let e = extract("Acme Corp. reported earnings. Globex Industries did not.").await;
        let acme = e
            .entities
            .iter()
            .find(|x| x.name.starts_with("Acme"))
            .unwrap();
        assert_eq!(acme.entity_type, entity_type::ORG);
        assert!(acme.confidence > 0.8);
    }

    #[tokio::test]
    async fn acronyms_are_classified_as_orgs() {
        let e = extract("Researchers at NASA published the data.").await;
        let nasa = e.entities.iter().find(|x| x.name == "NASA").unwrap();
        assert_eq!(nasa.entity_type, entity_type::ORG);
    }

    #[tokio::test]
    async fn honorifics_are_classified_as_people() {
        let e = extract("Later, Dr. Grace Hopper spoke about compilers.").await;
        let hopper = e
            .entities
            .iter()
            .find(|x| x.name.contains("Hopper"))
            .unwrap();
        assert_eq!(hopper.entity_type, entity_type::PERSON);
        assert!(hopper.confidence > 0.8);
    }

    #[tokio::test]
    async fn connectives_are_kept_inside_a_name() {
        let e = extract("Officials from the Bank of England commented today.").await;
        assert!(
            names(&e).iter().any(|n| n.contains("Bank of England")),
            "got {:?}",
            names(&e)
        );
    }

    #[tokio::test]
    async fn a_name_never_ends_on_a_connector() {
        let e = extract("Reports from Acme and other firms arrived.").await;
        for name in names(&e) {
            assert!(
                !name.to_lowercase().ends_with(" and") && !name.to_lowercase().ends_with(" of"),
                "dangling connector in {name:?}"
            );
        }
    }

    #[tokio::test]
    async fn repeated_mentions_are_counted() {
        let e =
            extract("Acme Corp. grew. Later, Acme Corp. grew again. Acme Corp. is large.").await;
        let acme = e
            .entities
            .iter()
            .find(|x| x.name.starts_with("Acme"))
            .unwrap();
        assert_eq!(acme.mentions, 3);
    }

    #[tokio::test]
    async fn surface_variants_merge_and_keep_the_longest_form() {
        let e = extract("The Acme Corp. is here. Acme Corp. is also there.").await;
        let matching: Vec<&ExtractedEntity> = e
            .entities
            .iter()
            .filter(|x| x.name.to_lowercase().contains("acme"))
            .collect();
        assert_eq!(matching.len(), 1, "variants should merge: {matching:?}");
        assert_eq!(matching[0].mentions, 2);
    }

    #[tokio::test]
    async fn co_occurrence_relations_are_emitted_pairwise() {
        let e = extract("Ada Lovelace met Charles Babbage at NASA.").await;
        assert!(e.entities.len() >= 3);
        let expected = e.entities.len() * (e.entities.len() - 1) / 2;
        assert_eq!(e.relations.len(), expected);
        assert!(
            e.relations
                .iter()
                .all(|r| r.relation_type == "co_occurs_with")
        );
    }

    #[tokio::test]
    async fn relations_can_be_switched_off() {
        let e = RuleExtractor::new(1, false)
            .extract("Ada Lovelace met Charles Babbage.")
            .await
            .unwrap();
        assert!(e.relations.is_empty());
        assert!(!e.entities.is_empty());
    }

    #[tokio::test]
    async fn a_mention_threshold_filters_one_off_names() {
        let text = "Acme Corp. appears twice. Acme Corp. again. Globex appears once.";
        let e = RuleExtractor::new(2, false).extract(text).await.unwrap();
        let found = names(&e);
        assert!(found.iter().any(|n| n.contains("Acme")));
        assert!(!found.iter().any(|n| n.contains("Globex")));
    }

    #[tokio::test]
    async fn empty_and_lowercase_text_yield_nothing() {
        assert!(extract("").await.is_empty());
        assert!(
            extract("all lowercase words with no names at all")
                .await
                .entities
                .is_empty()
        );
    }

    #[tokio::test]
    async fn the_null_extractor_finds_nothing() {
        let e = NullExtractor
            .extract("Ada Lovelace met Charles Babbage.")
            .await
            .unwrap();
        assert!(e.is_empty());
    }

    #[test]
    fn normalization_folds_articles_case_and_punctuation() {
        assert_eq!(normalize_name("The Acme Corp."), "acme corp");
        assert_eq!(normalize_name("acme corp"), "acme corp");
        assert_eq!(normalize_name("  ACME   CORP!  "), "acme corp");
        assert_eq!(normalize_name("AT&T"), "at&t");
        // A bare "The" must survive rather than normalising to nothing.
        assert_eq!(normalize_name("The"), "the");
    }

    #[tokio::test]
    async fn confidence_is_always_a_probability() {
        let e = extract("Dr. Grace Hopper visited Acme Corp. and NASA in Boston.").await;
        assert!(
            e.entities
                .iter()
                .all(|x| (0.0..=1.0).contains(&x.confidence))
        );
        assert!(
            e.relations
                .iter()
                .all(|r| (0.0..=1.0).contains(&r.confidence))
        );
    }
}
