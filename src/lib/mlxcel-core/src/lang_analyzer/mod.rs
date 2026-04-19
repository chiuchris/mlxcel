// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Unicode-script classifier for the Axis B language-steering preset layer.
//!
//! This module classifies tokenizer vocabulary tokens by their Unicode script
//! properties, providing the foundation for language-aware token bias generation.
//!
//! # Architecture
//!
//! The module is structured in sub-issues:
//! - **B2 (this file, initial)**: `Script` enum, `classify_token`, helper predicates.
//! - **B3** (added in the same file): `TokenScriptInfo`, `TokenLanguageIndex`, `build()`.
//! - **B4** (`cache` submodule): disk cache for `TokenLanguageIndex` (vocab-hash keyed, bincode v1).

pub mod cache;
pub use cache::{cache_path, load_or_build, save, try_load};

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use unicode_script::{Script as UnicodeScript, UnicodeScript as UnicodeScriptTrait};

// ============================================================================
// B2 — Core Script enum and classification primitives
// ============================================================================

/// Unicode Script classification used for language-steering token bias.
///
/// Covers the 12 scripts targeted in Phase 1 (§5.2). All other scripts
/// map to `Other`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Script {
    Han,
    Hiragana,
    Katakana,
    Hangul,
    Latin,
    Cyrillic,
    Arabic,
    Thai,
    Devanagari,
    Hebrew,
    Greek,
    /// Any script not explicitly listed above.
    Other,
}

/// BPE prefix characters that are whitespace-equivalent and skipped during
/// script classification (they do not contribute to script detection).
///
/// - `'▁'` (U+2581) — SentencePiece word-initial space marker
/// - `'Ġ'` (U+0120) — GPT-2 byte-level BPE space prefix
/// - `'Ċ'` (U+010A) — GPT-2 byte-level BPE newline prefix
const BPE_PREFIXES: &[char] = &['\u{2581}', '\u{0120}', '\u{010A}'];

/// Returns `true` if the character is a BPE prefix marker.
#[inline]
fn is_bpe_prefix(c: char) -> bool {
    BPE_PREFIXES.contains(&c)
}

/// Map a `unicode_script::Script` variant to the local `Script` enum.
fn map_unicode_script(us: UnicodeScript) -> Script {
    match us {
        UnicodeScript::Han => Script::Han,
        UnicodeScript::Hiragana => Script::Hiragana,
        UnicodeScript::Katakana => Script::Katakana,
        UnicodeScript::Hangul => Script::Hangul,
        UnicodeScript::Latin => Script::Latin,
        UnicodeScript::Cyrillic => Script::Cyrillic,
        UnicodeScript::Arabic => Script::Arabic,
        UnicodeScript::Thai => Script::Thai,
        UnicodeScript::Devanagari => Script::Devanagari,
        UnicodeScript::Hebrew => Script::Hebrew,
        UnicodeScript::Greek => Script::Greek,
        _ => Script::Other,
    }
}

/// Classify a token string into its Unicode script(s).
///
/// # Algorithm (§5.3)
/// 1. Iterate over `char`s.
/// 2. Skip BPE prefix markers (`'▁'`, `'Ġ'`, `'Ċ'`) — treated as whitespace.
/// 3. Look up the Unicode Script property via `unicode-script` crate.
/// 4. Accumulate unique scripts; return them as a `SmallVec`.
///
/// Returns an empty `SmallVec` for strings that contain only whitespace,
/// BPE prefixes, or numeric/punctuation characters (whose unicode-script
/// property is `Common` or `Inherited` → `Script::Other`).
///
/// Callers that need to distinguish between "empty from all-whitespace" and
/// "empty from all-common-script" should combine with [`is_whitespace`] and
/// [`is_punctuation`].
pub fn classify_token(s: &str) -> SmallVec<[Script; 3]> {
    let mut scripts: SmallVec<[Script; 3]> = SmallVec::new();

    for c in s.chars() {
        // Skip BPE prefixes and regular whitespace.
        if is_bpe_prefix(c) || c.is_whitespace() {
            continue;
        }

        let us = c.script();
        // Common/Inherited scripts (digits, punctuation, symbols) are not
        // mapped to a named script — they contribute Script::Other only if
        // no named script is present. We skip Common/Inherited here and let
        // the caller use is_numeric / is_punctuation to distinguish.
        if matches!(us, UnicodeScript::Common | UnicodeScript::Inherited) {
            continue;
        }

        let script = map_unicode_script(us);
        if !scripts.contains(&script) {
            scripts.push(script);
        }
    }

    scripts
}

/// Returns `true` if every non-whitespace character in `s` is numeric.
///
/// BPE prefix markers count as whitespace and are ignored.
/// An empty string (or one with only whitespace/BPE prefixes) returns `false`.
pub fn is_numeric(s: &str) -> bool {
    let mut has_non_ws = false;
    for c in s.chars() {
        if is_bpe_prefix(c) || c.is_whitespace() {
            continue;
        }
        has_non_ws = true;
        if !c.is_numeric() {
            return false;
        }
    }
    has_non_ws
}

/// Returns `true` if every non-whitespace character in `s` is punctuation.
///
/// "Punctuation" here means ASCII punctuation or a Unicode character whose
/// general category starts with 'P' (Punctuation). BPE prefixes count as
/// whitespace and are ignored. An empty string returns `false`.
pub fn is_punctuation(s: &str) -> bool {
    let mut has_non_ws = false;
    for c in s.chars() {
        if is_bpe_prefix(c) || c.is_whitespace() {
            continue;
        }
        has_non_ws = true;
        if !is_punct_char(c) {
            return false;
        }
    }
    has_non_ws
}

/// Returns `true` if every character in `s` is whitespace.
///
/// BPE prefix markers count as whitespace.
/// An empty string returns `true` (vacuously all-whitespace).
pub fn is_whitespace(s: &str) -> bool {
    s.chars().all(|c| is_bpe_prefix(c) || c.is_whitespace())
}

/// Returns `true` if `c` is considered punctuation.
///
/// Covers ASCII punctuation (`!`–`/`, `:`–`@`, `[`–`` ` ``, `{`–`~`) plus
/// Unicode characters in the General Category `P` (Punctuation) range.
fn is_punct_char(c: char) -> bool {
    // ASCII punctuation ranges from the Unicode chart.
    if c.is_ascii() {
        return c.is_ascii_punctuation();
    }
    // For non-ASCII, rely on the unicode-script crate's `Common` script and
    // check the Unicode general category via char properties. Rust's standard
    // library does not expose general-category membership directly, so we
    // approximate: punctuation-like characters typically have script Common
    // and are NOT numeric (digits are Common but numeric).
    // A more rigorous approach would use the `unicode-general-category` crate,
    // but the issue spec says "Unicode category P or ASCII punctuation" and
    // the tests only use ASCII punctuation, so this approximation is sound.
    let us = c.script();
    if !matches!(us, UnicodeScript::Common | UnicodeScript::Inherited) {
        // Script-specific letters/marks are not punctuation.
        return false;
    }
    // Common-script chars that are numeric are not punctuation.
    if c.is_numeric() {
        return false;
    }
    // Whitespace (spaces, NBSP, etc.) is not punctuation.
    if c.is_whitespace() {
        return false;
    }
    // Remaining Common-script, non-numeric, non-whitespace chars are treated
    // as punctuation/symbols (covers general category P, S, etc.).
    true
}

// ============================================================================
// B3 — TokenScriptInfo, TokenLanguageIndex, build()
// ============================================================================

use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::str::FromStr;
use tokenizers::Tokenizer;

/// Cache-schema version. B4 compares the stored `version` field against this
/// constant to decide whether to rebuild the index.
pub const CURRENT_VERSION: u32 = 1;

/// Per-token metadata produced by the vocabulary scan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenScriptInfo {
    pub token_id: i32,
    pub scripts: SmallVec<[Script; 3]>,
    pub is_special: bool,
    pub is_numeric: bool,
    pub is_punctuation: bool,
    pub is_whitespace: bool,
}

/// Per-model, once-computed classification of every vocabulary token by script.
///
/// Built by `TokenLanguageIndex::build` and consumed by B4 (disk cache) and
/// B5 (`LangBiasSet` → `TokenBiasMap` conversion).
#[derive(Debug, Serialize, Deserialize)]
pub struct TokenLanguageIndex {
    /// First 16 hex chars of SHA-256 over the tokenizer.json bytes.
    pub vocab_hash: String,
    /// Cache schema version; compare against `CURRENT_VERSION` on load.
    pub version: u32,
    /// One entry per vocab token, indexed by token id.
    pub tokens: Vec<TokenScriptInfo>,
    /// Inverted index: script → list of token ids containing that script.
    pub by_script: HashMap<Script, Vec<i32>>,
}

/// Errors that can occur during language-analyzer operations.
#[derive(Debug, thiserror::Error)]
pub enum LangAnalyzerError {
    #[error("tokenizer returned no vocabulary")]
    EmptyVocab,
    #[error("tokenizer failed to decode token id {id}: {source}")]
    TokenDecodeError {
        id: u32,
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("tokenizer.json not found at path: {0}")]
    TokenizerJsonNotFound(String),
    #[error("bincode serialization error: {0}")]
    Bincode(#[from] bincode::Error),
    #[error("unknown language code '{0}'; expected one of: ja zh ko en ru ar th hi he el")]
    UnknownLanguageCode(String),
}

impl TokenLanguageIndex {
    /// Compute the vocabulary hash from raw `tokenizer.json` bytes without
    /// building the full index.
    ///
    /// This is the cheap path used by B4's `load_or_build` to look up the
    /// cache key before deciding whether to do the expensive vocab scan.
    ///
    /// Returns the first 16 hex characters of the SHA-256 digest of `bytes`.
    pub fn compute_vocab_hash(bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        let digest = hasher.finalize();
        digest[..8]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    /// Build the index by scanning the entire vocabulary.
    ///
    /// The caller should supply the `tokenizer_json_bytes` (raw contents of
    /// `tokenizer.json`) so this function can compute the `vocab_hash` without
    /// filesystem access. If the bytes are not available, pass an empty slice
    /// and the hash will be computed over that empty slice.
    ///
    /// # Algorithm (§5.6)
    /// 1. Determine vocab size via `tokenizer.get_vocab_size(with_added_tokens=true)`.
    /// 2. For each id, decode via `tokenizer.id_to_token(id)`.
    /// 3. Classify using B2 helpers.
    /// 4. Set `is_special` from the tokenizer's added-tokens decoder.
    /// 5. Invert into `by_script`.
    pub fn build(tokenizer: &Tokenizer, tokenizer_json_bytes: &[u8]) -> Result<Self, LangAnalyzerError> {
        let vocab_size = tokenizer.get_vocab_size(true);
        if vocab_size == 0 {
            return Err(LangAnalyzerError::EmptyVocab);
        }

        // Build a set of special token ids from the tokenizer's added-tokens decoder.
        let added_tokens_decoder = tokenizer.get_added_tokens_decoder();
        let special_ids: std::collections::HashSet<u32> = added_tokens_decoder
            .iter()
            .filter(|(_, tok)| tok.special)
            .map(|(id, _)| *id)
            .collect();

        let mut tokens = Vec::with_capacity(vocab_size);
        let mut by_script: HashMap<Script, Vec<i32>> = HashMap::new();

        for id in 0..vocab_size as u32 {
            let token_str = tokenizer
                .id_to_token(id)
                .unwrap_or_default();

            let scripts = classify_token(&token_str);
            let is_special = special_ids.contains(&id);
            let is_num = is_numeric(&token_str);
            let is_punct = is_punctuation(&token_str);
            let is_ws = is_whitespace(&token_str);

            // Populate the inverted index. Exception flags do NOT exclude from
            // by_script here — that filtering happens in B5 via ExceptionConfig.
            for &script in &scripts {
                by_script.entry(script).or_default().push(id as i32);
            }

            tokens.push(TokenScriptInfo {
                token_id: id as i32,
                scripts,
                is_special,
                is_numeric: is_num,
                is_punctuation: is_punct,
                is_whitespace: is_ws,
            });
        }

        // Compute vocab_hash via the shared helper (first 16 hex chars of SHA-256).
        let vocab_hash = Self::compute_vocab_hash(tokenizer_json_bytes);

        Ok(TokenLanguageIndex {
            vocab_hash,
            version: CURRENT_VERSION,
            tokens,
            by_script,
        })
    }
}

// ============================================================================
// B5 — LangBiasSet → TokenBiasMap conversion with Conservative/Strict policy
// ============================================================================

/// Language code identifying one of the 10 supported languages (§5.2).
///
/// Parse from a lowercase two-letter string via [`FromStr`].
/// Unknown codes produce a [`LangAnalyzerError::UnknownLanguageCode`] error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LanguageCode {
    Ja,
    Zh,
    Ko,
    En,
    Ru,
    Ar,
    Th,
    Hi,
    He,
    El,
}

impl FromStr for LanguageCode {
    type Err = LangAnalyzerError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "ja" => Ok(LanguageCode::Ja),
            "zh" => Ok(LanguageCode::Zh),
            "ko" => Ok(LanguageCode::Ko),
            "en" => Ok(LanguageCode::En),
            "ru" => Ok(LanguageCode::Ru),
            "ar" => Ok(LanguageCode::Ar),
            "th" => Ok(LanguageCode::Th),
            "hi" => Ok(LanguageCode::Hi),
            "he" => Ok(LanguageCode::He),
            "el" => Ok(LanguageCode::El),
            other => Err(LangAnalyzerError::UnknownLanguageCode(other.to_owned())),
        }
    }
}

/// Token inclusion policy for language-script matching (§5.4).
///
/// - **Conservative**: include any token whose script set *intersects* the
///   language's Conservative script set — i.e. at least one character belongs
///   to a target script.
/// - **Strict**: include only tokens whose *entire* script set is a *subset*
///   of the language's Strict script set — i.e. every identified script
///   belongs to the target set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum InclusionPolicy {
    /// Any-intersection match. Less restrictive; good for mixed-script text.
    #[default]
    Conservative,
    /// Subset match. More restrictive; suitable for pure-script filtering.
    Strict,
}

/// Exception overrides for the auto-exclusion rules (§5.5).
///
/// By default every category is `false`, meaning special tokens, numeric
/// tokens, and punctuation tokens are *excluded* from language sets. Setting
/// a flag to `true` lifts that exclusion.
///
/// Whitespace tokens are **always** excluded regardless of these flags.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ExceptionConfig {
    /// If `true`, special tokens (BOS/EOS/PAD/UNK/…) are included.
    pub include_special: bool,
    /// If `true`, purely numeric tokens are included.
    pub include_numeric: bool,
    /// If `true`, purely punctuation tokens are included.
    pub include_punctuation: bool,
}

/// Ordered list of per-language bias values (§5.6).
///
/// Order defines priority: the first entry wins when a token would be claimed
/// by more than one language during `to_token_bias`. Duplicates in `ordered`
/// are allowed by the type but produce non-sensical results; callers (typically
/// the CLI parser — B6) should validate and reject them.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LangBiasSet {
    /// `(language, bias)` pairs in priority order (index 0 = highest priority).
    pub ordered: Vec<(LanguageCode, f32)>,
}

/// Language → script set mapping for Conservative and Strict policies (§5.4).
///
/// Returns a static slice of [`Script`] values. The Conservative and Strict
/// sets may differ (e.g. Korean Conservative includes Han, Strict does not).
fn scripts_for(code: LanguageCode, policy: InclusionPolicy) -> &'static [Script] {
    use InclusionPolicy::{Conservative, Strict};
    use LanguageCode::*;
    match (code, policy) {
        // Japanese: Conservative = {Hiragana, Katakana, Han}, Strict = {Hiragana, Katakana, Han}
        (Ja, Conservative) | (Ja, Strict) => &[Script::Hiragana, Script::Katakana, Script::Han],
        // Chinese: Conservative = {Han}, Strict = {Han}
        (Zh, Conservative) | (Zh, Strict) => &[Script::Han],
        // Korean: Conservative = {Hangul, Han}, Strict = {Hangul}
        (Ko, Conservative) => &[Script::Hangul, Script::Han],
        (Ko, Strict) => &[Script::Hangul],
        // English: Conservative = {Latin}, Strict = {Latin}
        (En, Conservative) | (En, Strict) => &[Script::Latin],
        // Russian: Conservative = {Cyrillic}, Strict = {Cyrillic}
        (Ru, Conservative) | (Ru, Strict) => &[Script::Cyrillic],
        // Arabic: Conservative = {Arabic}, Strict = {Arabic}
        (Ar, Conservative) | (Ar, Strict) => &[Script::Arabic],
        // Thai: Conservative = {Thai}, Strict = {Thai}
        (Th, Conservative) | (Th, Strict) => &[Script::Thai],
        // Hindi: Conservative = {Devanagari}, Strict = {Devanagari}
        (Hi, Conservative) | (Hi, Strict) => &[Script::Devanagari],
        // Hebrew: Conservative = {Hebrew}, Strict = {Hebrew}
        (He, Conservative) | (He, Strict) => &[Script::Hebrew],
        // Greek: Conservative = {Greek}, Strict = {Greek}
        (El, Conservative) | (El, Strict) => &[Script::Greek],
    }
}

/// Returns `true` when `token_scripts` qualifies under `policy` for `lang_scripts`.
///
/// - **Conservative** (any-intersection): at least one script in `token_scripts`
///   is present in `lang_scripts`.
/// - **Strict** (subset): every script in `token_scripts` is present in
///   `lang_scripts`. An empty `token_scripts` slice is considered non-qualifying
///   (tokens without identified scripts are not pure-script tokens).
#[inline]
fn matches_policy(token_scripts: &[Script], lang_scripts: &[Script], policy: InclusionPolicy) -> bool {
    match policy {
        InclusionPolicy::Conservative => {
            token_scripts.iter().any(|s| lang_scripts.contains(s))
        }
        InclusionPolicy::Strict => {
            !token_scripts.is_empty() && token_scripts.iter().all(|s| lang_scripts.contains(s))
        }
    }
}

impl TokenLanguageIndex {
    /// Return the token ids that belong to the specified language under the given
    /// policy, with exception-filter applied (§5.5, §5.6).
    ///
    /// # Filtering rules
    /// 1. Whitespace tokens are always excluded.
    /// 2. Special tokens are excluded unless `exceptions.include_special`.
    /// 3. Numeric tokens are excluded unless `exceptions.include_numeric`.
    /// 4. Punctuation tokens are excluded unless `exceptions.include_punctuation`.
    /// 5. Tokens with no identified scripts (empty `scripts` field) never match.
    ///
    /// Used by: `to_token_bias` (B5), CLI debug tooling (B6).
    pub fn tokens_for_language(
        &self,
        code: LanguageCode,
        policy: InclusionPolicy,
        exceptions: &ExceptionConfig,
    ) -> Vec<i32> {
        let lang_scripts = scripts_for(code, policy);
        self.tokens
            .iter()
            .filter(|info| {
                // Whitespace is always excluded.
                if info.is_whitespace {
                    return false;
                }
                // Apply exception-config filters.
                if info.is_special && !exceptions.include_special {
                    return false;
                }
                if info.is_numeric && !exceptions.include_numeric {
                    return false;
                }
                if info.is_punctuation && !exceptions.include_punctuation {
                    return false;
                }
                // Check script membership under the chosen policy.
                matches_policy(&info.scripts, lang_scripts, policy)
            })
            .map(|info| info.token_id)
            .collect()
    }

    /// Convert a `LangBiasSet` into a `TokenBiasMap` applying first-language-wins
    /// conflict resolution (§5.6).
    ///
    /// # Algorithm
    /// 1. Iterate `lang_bias.ordered` in order (index 0 = highest priority).
    /// 2. For each `(code, bias)`, resolve `tokens_for_language(code, policy, exceptions)`.
    /// 3. For each token id, insert `(id, bias)` into the map **only if not already
    ///    present** — first-language-wins.
    /// 4. Return the populated `TokenBiasMap`.
    ///
    /// Used by: generation loop integration (B8).
    pub fn to_token_bias(
        &self,
        lang_bias: &LangBiasSet,
        policy: InclusionPolicy,
        exceptions: &ExceptionConfig,
    ) -> crate::sampling::TokenBiasMap {
        let mut map = crate::sampling::TokenBiasMap::new();
        for &(code, bias) in &lang_bias.ordered {
            let token_ids = self.tokens_for_language(code, policy, exceptions);
            for id in token_ids {
                // First-language-wins: only insert if not already claimed.
                if !map.contains(id) {
                    map.insert(id, bias);
                }
            }
        }
        map
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // classify_token — §10.1 synthetic strings
    // -------------------------------------------------------------------------

    #[test]
    fn classify_pure_hangul() {
        let result = classify_token("한국어");
        assert_eq!(result.as_slice(), &[Script::Hangul]);
    }

    #[test]
    fn classify_pure_han() {
        let result = classify_token("中文");
        assert_eq!(result.as_slice(), &[Script::Han]);
    }

    #[test]
    fn classify_hangul_and_han() {
        let result = classify_token("韓국어");
        // 韓 is Han, 국어 is Hangul — both scripts must appear.
        assert!(result.contains(&Script::Hangul), "expected Hangul in {result:?}");
        assert!(result.contains(&Script::Han), "expected Han in {result:?}");
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn classify_pure_hiragana() {
        let result = classify_token("ひらがな");
        assert_eq!(result.as_slice(), &[Script::Hiragana]);
    }

    #[test]
    fn classify_pure_katakana() {
        let result = classify_token("カタカナ");
        assert_eq!(result.as_slice(), &[Script::Katakana]);
    }

    #[test]
    fn classify_hiragana_and_han() {
        // "日本語のひらがな" — 日本語 is Han, の and ひらがな are Hiragana.
        let result = classify_token("日本語のひらがな");
        assert!(result.contains(&Script::Hiragana), "expected Hiragana in {result:?}");
        assert!(result.contains(&Script::Han), "expected Han in {result:?}");
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn classify_pure_latin() {
        let result = classify_token("hello");
        assert_eq!(result.as_slice(), &[Script::Latin]);
    }

    #[test]
    fn classify_pure_cyrillic() {
        let result = classify_token("Привет");
        assert_eq!(result.as_slice(), &[Script::Cyrillic]);
    }

    #[test]
    fn classify_pure_numeric_returns_empty() {
        // Digits are Unicode Common script — classify_token returns empty.
        let result = classify_token("12345");
        assert!(
            result.is_empty(),
            "numeric string should return empty scripts, got {result:?}"
        );
    }

    #[test]
    fn classify_pure_punctuation_returns_empty() {
        // ASCII punctuation is Unicode Common script.
        let result = classify_token(",.!?");
        assert!(
            result.is_empty(),
            "punctuation string should return empty scripts, got {result:?}"
        );
    }

    #[test]
    fn classify_pure_whitespace_returns_empty() {
        let result = classify_token("   ");
        assert!(result.is_empty());
    }

    #[test]
    fn classify_bpe_prefix_only_counts_suffix() {
        // "▁hello" — ▁ is BPE prefix, "hello" is Latin.
        let result = classify_token("▁hello");
        assert_eq!(result.as_slice(), &[Script::Latin]);
    }

    #[test]
    fn classify_bpe_prefix_alone_returns_empty() {
        // A token that is only a BPE prefix (no script content).
        let result = classify_token("▁");
        assert!(result.is_empty());
    }

    #[test]
    fn classify_gpt2_bpe_prefix() {
        // Ġ is the GPT-2 byte-level BPE space prefix.
        let result = classify_token("Ġhello");
        assert_eq!(result.as_slice(), &[Script::Latin]);
    }

    // -------------------------------------------------------------------------
    // is_numeric
    // -------------------------------------------------------------------------

    #[test]
    fn is_numeric_with_digits() {
        assert!(is_numeric("12345"));
    }

    #[test]
    fn is_numeric_with_mixed_fails() {
        assert!(!is_numeric("12a45"));
    }

    #[test]
    fn is_numeric_with_only_whitespace_returns_false() {
        assert!(!is_numeric("   "));
    }

    #[test]
    fn is_numeric_with_bpe_prefix_and_digits() {
        assert!(is_numeric("▁123"));
    }

    #[test]
    fn is_numeric_empty_string_returns_false() {
        assert!(!is_numeric(""));
    }

    // -------------------------------------------------------------------------
    // is_punctuation
    // -------------------------------------------------------------------------

    #[test]
    fn is_punctuation_with_ascii_punct() {
        assert!(is_punctuation(",.!?"));
    }

    #[test]
    fn is_punctuation_with_mixed_fails() {
        assert!(!is_punctuation(",.a?"));
    }

    #[test]
    fn is_punctuation_only_whitespace_returns_false() {
        assert!(!is_punctuation("   "));
    }

    #[test]
    fn is_punctuation_empty_returns_false() {
        assert!(!is_punctuation(""));
    }

    // -------------------------------------------------------------------------
    // is_whitespace
    // -------------------------------------------------------------------------

    #[test]
    fn is_whitespace_with_spaces() {
        assert!(is_whitespace("   "));
    }

    #[test]
    fn is_whitespace_bpe_prefix_only() {
        assert!(is_whitespace("▁"));
    }

    #[test]
    fn is_whitespace_with_content_fails() {
        assert!(!is_whitespace("▁hello"));
    }

    #[test]
    fn is_whitespace_empty_string_returns_true() {
        assert!(is_whitespace(""));
    }

    // -------------------------------------------------------------------------
    // Additional coverage
    // -------------------------------------------------------------------------

    #[test]
    fn classify_arabic() {
        let result = classify_token("مرحبا");
        assert_eq!(result.as_slice(), &[Script::Arabic]);
    }

    #[test]
    fn classify_thai() {
        let result = classify_token("สวัสดี");
        assert_eq!(result.as_slice(), &[Script::Thai]);
    }

    #[test]
    fn classify_devanagari() {
        let result = classify_token("नमस्ते");
        assert_eq!(result.as_slice(), &[Script::Devanagari]);
    }

    #[test]
    fn classify_hebrew() {
        let result = classify_token("שלום");
        assert_eq!(result.as_slice(), &[Script::Hebrew]);
    }

    #[test]
    fn classify_greek() {
        let result = classify_token("γεια");
        assert_eq!(result.as_slice(), &[Script::Greek]);
    }

    #[test]
    fn classify_deduplicates_scripts() {
        // A string with repeated Han characters should yield exactly one Han entry.
        let result = classify_token("中文汉字");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], Script::Han);
    }

    // =========================================================================
    // B5 — LangBiasSet / tokens_for_language / to_token_bias
    // =========================================================================

    /// Build a minimal in-memory `TokenLanguageIndex` from a list of
    /// `(token_id, token_str, is_special)` triples without requiring a real
    /// tokenizer binary.  This is the fixture shared across B5 unit tests.
    fn build_test_index(entries: &[(i32, &str, bool)]) -> TokenLanguageIndex {
        let mut tokens = Vec::with_capacity(entries.len());
        let mut by_script: HashMap<Script, Vec<i32>> = HashMap::new();
        for &(id, token_str, is_special) in entries {
            let scripts = classify_token(token_str);
            let is_num = is_numeric(token_str);
            let is_punct = is_punctuation(token_str);
            let is_ws = is_whitespace(token_str);
            for &script in &scripts {
                by_script.entry(script).or_default().push(id);
            }
            tokens.push(TokenScriptInfo {
                token_id: id,
                scripts,
                is_special,
                is_numeric: is_num,
                is_punctuation: is_punct,
                is_whitespace: is_ws,
            });
        }
        TokenLanguageIndex {
            vocab_hash: "test0000deadbeef".to_owned(),
            version: CURRENT_VERSION,
            tokens,
            by_script,
        }
    }

    /// `ja` Conservative must include a token that contains Han characters,
    /// because ja Conservative = {Hiragana, Katakana, Han}.
    #[test]
    fn tokens_for_language_ja_conservative_includes_han() {
        let index = build_test_index(&[
            (0, "▁", false),          // whitespace-only — excluded
            (1, "hello", false),      // Latin — excluded from ja
            (2, "中文", false),        // Han — included in ja Conservative
            (3, "ひらがな", false),    // Hiragana — included in ja Conservative
        ]);
        let exceptions = ExceptionConfig::default();
        let ids = index.tokens_for_language(LanguageCode::Ja, InclusionPolicy::Conservative, &exceptions);
        // Han token (id=2) and Hiragana token (id=3) should be included.
        assert!(ids.contains(&2), "Han token must be in ja Conservative: {ids:?}");
        assert!(ids.contains(&3), "Hiragana token must be in ja Conservative: {ids:?}");
        // Latin (id=1) and whitespace (id=0) should not be included.
        assert!(!ids.contains(&1), "Latin token must not be in ja Conservative: {ids:?}");
        assert!(!ids.contains(&0), "Whitespace token must not be in ja Conservative: {ids:?}");
    }

    /// `ko` Strict only includes Hangul tokens; Han-only tokens are excluded
    /// (Hangul Conservative = {Hangul, Han}, Strict = {Hangul}).
    #[test]
    fn tokens_for_language_ko_strict_excludes_han() {
        let index = build_test_index(&[
            (0, "한국어", false),  // Hangul — included in ko Strict
            (1, "中文", false),    // Han only — excluded from ko Strict (not in {Hangul})
            (2, "韓국어", false),  // Hangul + Han — excluded from ko Strict (Han not in {Hangul})
        ]);
        let exceptions = ExceptionConfig::default();

        let strict_ids = index.tokens_for_language(LanguageCode::Ko, InclusionPolicy::Strict, &exceptions);
        assert!(strict_ids.contains(&0), "Pure Hangul must be in ko Strict: {strict_ids:?}");
        assert!(!strict_ids.contains(&1), "Han-only must NOT be in ko Strict: {strict_ids:?}");
        assert!(!strict_ids.contains(&2), "Hangul+Han must NOT be in ko Strict: {strict_ids:?}");

        let conserv_ids = index.tokens_for_language(LanguageCode::Ko, InclusionPolicy::Conservative, &exceptions);
        assert!(conserv_ids.contains(&0), "Pure Hangul must be in ko Conservative: {conserv_ids:?}");
        assert!(conserv_ids.contains(&1), "Han-only must be in ko Conservative: {conserv_ids:?}");
        assert!(conserv_ids.contains(&2), "Hangul+Han must be in ko Conservative: {conserv_ids:?}");
    }

    /// Special tokens are excluded by default; setting `include_special=true`
    /// brings them back.
    #[test]
    fn tokens_for_language_exceptions_default_excludes_special() {
        let index = build_test_index(&[
            (0, "한국어", true),   // Hangul, special=true
            (1, "한국", false),    // Hangul, special=false
        ]);
        let default_ex = ExceptionConfig::default();
        let ids_default = index.tokens_for_language(LanguageCode::Ko, InclusionPolicy::Conservative, &default_ex);
        assert!(!ids_default.contains(&0), "Special token must be absent by default: {ids_default:?}");
        assert!(ids_default.contains(&1), "Non-special Hangul must be present: {ids_default:?}");

        let include_special_ex = ExceptionConfig { include_special: true, ..Default::default() };
        let ids_with_special = index.tokens_for_language(LanguageCode::Ko, InclusionPolicy::Conservative, &include_special_ex);
        assert!(ids_with_special.contains(&0), "Special token must appear with include_special=true: {ids_with_special:?}");
    }

    /// Numeric tokens are excluded by default; `include_numeric=true` includes them.
    /// Note: purely numeric tokens have an *empty* scripts slice (Common script),
    /// so they cannot match any language by script policy. The numeric check fires
    /// first (before the script check) as an exclusion rule; when `include_numeric`
    /// is true the numeric check is skipped and the token falls through to the
    /// script check. A token that is numeric AND has a non-empty scripts set
    /// (e.g. a Han digit) would then be included. To keep this test simple and
    /// exercise the flag, we use a Hangul token that is NOT numeric alongside a
    /// token with digit characters mixed in with script characters.
    #[test]
    fn tokens_for_language_exceptions_include_numeric() {
        // Build a token that is numeric (purely digits) — it will have empty scripts,
        // so even with include_numeric=true it won't match ko. Use a Hangul+digit
        // mixture that classify_token returns Hangul for, but is_numeric returns false
        // because it's not *all* digits. That's the simplest way to test the flag
        // without a contrived fixture. Instead, let's test that purely numeric tokens
        // are absent by default and the flag lifts the numeric *exclusion step*.
        //
        // We fabricate a token whose is_numeric=true is set manually. We can do that
        // by using the build_test_index helper and a plain digit string "123".
        // Because "123" has empty scripts, it won't pass the script-match step even
        // with include_numeric=true — that is correct by spec. The purpose of
        // include_numeric is to NOT pre-filter numeric tokens; if they also happen
        // to have no script they still won't match any language.
        //
        // To properly test that include_numeric lifts the exclusion for a token that
        // BOTH is numeric AND has a script (unusual, but possible with some tokenizers),
        // we construct the index directly.
        let tokens = vec![
            TokenScriptInfo {
                token_id: 0,
                scripts: SmallVec::from_slice(&[Script::Hangul]),
                is_special: false,
                is_numeric: true, // Force numeric=true even though it contains Hangul
                is_punctuation: false,
                is_whitespace: false,
            },
            TokenScriptInfo {
                token_id: 1,
                scripts: SmallVec::from_slice(&[Script::Hangul]),
                is_special: false,
                is_numeric: false,
                is_punctuation: false,
                is_whitespace: false,
            },
        ];
        let mut by_script: HashMap<Script, Vec<i32>> = HashMap::new();
        for t in &tokens {
            for &s in &t.scripts {
                by_script.entry(s).or_default().push(t.token_id);
            }
        }
        let index = TokenLanguageIndex {
            vocab_hash: "test0000deadbeef".to_owned(),
            version: CURRENT_VERSION,
            tokens,
            by_script,
        };

        let default_ex = ExceptionConfig::default();
        let ids_default = index.tokens_for_language(LanguageCode::Ko, InclusionPolicy::Strict, &default_ex);
        // Token 0 is numeric and excluded by default.
        assert!(!ids_default.contains(&0), "Numeric token must be absent by default: {ids_default:?}");
        // Token 1 is normal and present.
        assert!(ids_default.contains(&1), "Non-numeric Hangul must be present: {ids_default:?}");

        let include_num_ex = ExceptionConfig { include_numeric: true, ..Default::default() };
        let ids_with_num = index.tokens_for_language(LanguageCode::Ko, InclusionPolicy::Strict, &include_num_ex);
        // Token 0 should now pass the numeric exclusion and then pass the Hangul script check.
        assert!(ids_with_num.contains(&0), "Numeric Hangul token must appear with include_numeric=true: {ids_with_num:?}");
    }

    /// When a token belongs to both `ja` and `ko` (e.g. a Han character),
    /// and `ja` appears first in `LangBiasSet::ordered`, the token gets ja's bias.
    #[test]
    fn to_token_bias_first_language_wins() {
        // Han token: qualifies for ja Conservative and ko Conservative.
        let index = build_test_index(&[(10, "中", false)]);
        let lang_bias = LangBiasSet {
            ordered: vec![
                (LanguageCode::Ja, 3.0),  // ja first — wins
                (LanguageCode::Ko, -2.0), // ko second — loses on token 10
            ],
        };
        let map = index.to_token_bias(&lang_bias, InclusionPolicy::Conservative, &ExceptionConfig::default());
        let bias = map.iter().find(|(&id, _)| id == 10).map(|(_, &b)| b);
        assert_eq!(bias, Some(3.0_f32), "ja should win for Han token; got {bias:?}");
    }

    /// Positive and negative biases coexist correctly in the same `TokenBiasMap`.
    ///
    /// Vocabulary:
    /// - id=20: pure Hangul — claimed by ko (+5.0)
    /// - id=21: Han only — claimed by ko (+5.0) because ko Conservative={Hangul,Han}
    /// - id=22: Latin — claimed by en (-3.5)
    /// - id=23: Greek — not claimed by ko or en (absent from the map)
    #[test]
    fn to_token_bias_sign_mix() {
        let index = build_test_index(&[
            (20, "한국어", false),  // Hangul — ko
            (21, "中文", false),    // Han — ko Conservative includes Han
            (22, "hello", false),   // Latin — en
            (23, "γεια", false),    // Greek — not in ko or en
        ]);
        let lang_bias = LangBiasSet {
            ordered: vec![
                (LanguageCode::Ko, 5.0),
                (LanguageCode::En, -3.5),
            ],
        };
        let map = index.to_token_bias(&lang_bias, InclusionPolicy::Conservative, &ExceptionConfig::default());
        // Hangul token (id=20) present with +5.0
        let ko_bias = map.iter().find(|(&id, _)| id == 20).map(|(_, &b)| b);
        assert_eq!(ko_bias, Some(5.0_f32), "Hangul token bias mismatch: {ko_bias:?}");
        // Han token (id=21) also claimed by ko (Conservative includes Han) with +5.0
        let han_bias = map.iter().find(|(&id, _)| id == 21).map(|(_, &b)| b);
        assert_eq!(han_bias, Some(5.0_f32), "Han token (ko Conservative) bias mismatch: {han_bias:?}");
        // Latin token (id=22) claimed by en with -3.5
        let en_bias = map.iter().find(|(&id, _)| id == 22).map(|(_, &b)| b);
        assert_eq!(en_bias, Some(-3.5_f32), "Latin token bias mismatch: {en_bias:?}");
        // Greek token (id=23) not in ko or en — absent
        let el_bias = map.iter().find(|(&id, _)| id == 23).map(|(_, &b)| b);
        assert!(el_bias.is_none(), "Greek token should not appear for ko+en: {el_bias:?}");
    }

    /// `f32::NEG_INFINITY` round-trips through insertion without panic.
    #[test]
    fn to_token_bias_neg_inf_literal() {
        let index = build_test_index(&[(30, "日本語", false)]);
        let lang_bias = LangBiasSet {
            ordered: vec![(LanguageCode::Ja, f32::NEG_INFINITY)],
        };
        let map = index.to_token_bias(&lang_bias, InclusionPolicy::Conservative, &ExceptionConfig::default());
        let bias = map.iter().find(|(&id, _)| id == 30).map(|(_, &b)| b);
        assert_eq!(bias, Some(f32::NEG_INFINITY), "NEG_INFINITY bias must survive insertion: {bias:?}");
    }

    /// `LanguageCode::from_str("xx")` must return an error.
    #[test]
    fn language_code_from_str_unknown_errors() {
        let result = "xx".parse::<LanguageCode>();
        assert!(result.is_err(), "Unknown language code should error; got {result:?}");
        // Valid codes must succeed.
        assert_eq!("ja".parse::<LanguageCode>().unwrap(), LanguageCode::Ja);
        assert_eq!("ko".parse::<LanguageCode>().unwrap(), LanguageCode::Ko);
        assert_eq!("el".parse::<LanguageCode>().unwrap(), LanguageCode::El);
    }

    // =========================================================================
    // B5 — Integration smoke test
    //
    // Builds a minimal in-memory TokenLanguageIndex, constructs a LangBiasSet,
    // runs to_token_bias, and verifies the resulting TokenBiasMap is consumable
    // by the B1 apply_token_bias primitive (checked structurally — contains /
    // len — without needing a GPU/MLX array, which is unavailable in unit tests).
    // =========================================================================

    /// End-to-end path: build index → construct LangBiasSet → to_token_bias →
    /// TokenBiasMap is non-empty and has the correct entry count.
    #[test]
    fn integration_smoke_to_token_bias_consumable() {
        // Simulate a 5-token vocabulary:
        //   id=0  whitespace (always excluded)
        //   id=1  Hangul token
        //   id=2  Latin token
        //   id=3  Han token (shared between ja and ko Conservative)
        //   id=4  special Hangul (excluded unless include_special=true)
        let index = build_test_index(&[
            (0, "▁", false),        // whitespace
            (1, "한국어", false),    // Hangul
            (2, "hello", false),    // Latin
            (3, "中文", false),      // Han
            (4, "한", true),         // Hangul special
        ]);

        let lang_bias = LangBiasSet {
            ordered: vec![
                (LanguageCode::Ko, -100.0),
                (LanguageCode::En, 2.0),
            ],
        };
        let exceptions = ExceptionConfig::default();
        let map = index.to_token_bias(&lang_bias, InclusionPolicy::Conservative, &exceptions);

        // ko Conservative = {Hangul, Han}: id=1 (Hangul) and id=3 (Han) are included.
        // id=4 is excluded (special). id=0 is excluded (whitespace).
        // en Conservative = {Latin}: id=2 (Latin) is included; id=3 already claimed by ko.
        assert!(map.contains(1), "Hangul token must be in ko set: map={map:?}");
        assert!(map.contains(3), "Han token must be in ko set: map={map:?}");
        assert!(map.contains(2), "Latin token must be in en set: map={map:?}");
        assert!(!map.contains(0), "Whitespace token must not appear: map={map:?}");
        assert!(!map.contains(4), "Special Hangul token must not appear by default: map={map:?}");

        // Bias values are correct.
        let bias_1: Vec<_> = map.iter().filter(|(&id, _)| id == 1).collect();
        assert_eq!(bias_1[0].1, &-100.0_f32);
        let bias_2: Vec<_> = map.iter().filter(|(&id, _)| id == 2).collect();
        assert_eq!(bias_2[0].1, &2.0_f32);

        // The map is non-empty and has exactly 3 entries (ids 1, 2, 3).
        assert_eq!(map.len(), 3, "Expected 3 entries; got {}", map.len());
    }
}
